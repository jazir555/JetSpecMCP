"""
VGB: Value-Guided Sampling with Stochastic Backtracking
Reimplementation of: "Taming Imperfect Process Verifiers:
A Sampling Perspective on Backtracking" (Rohatgi et al., 2025)

Fixes over the previous MCP server:
  1. Theorem-aware step count T computed from ε_V, H, δ (Thm 4.1 / 4.2)
  2. ε_V estimation and assumption verification (Assumption 4.1 / 4.2)
  3. VGB-Momentum (Hayes & Sinclair 2010) — Appendix E.1
  4. Chat-template support for instruct-tuned local models
  5. Auto-fallback to sampled candidate mode when API lacks logprobs
  6. Token-accurate remaining-token estimation via tokenizer
  7. Separate (small) verifier model option — keeps large base models usable
  8. Default value_type="tilt" for true plug-and-play (Section 5.3)
  9. KL-regularized value via Q̂ (Algorithm 2)
 10. Ground-truth τ at leaves (Algorithm 1, Line 1)
 11. Bug fixes: _default_config shadowing, run_theoretical attempt leak,
     OutcomeLevelRS completion check, tree-walk during mutation

PATCHES APPLIED (this revision):
  1A. Chat-template continuation: assistant-message prefill instead of
      concatenating partial into the user prompt.
  1B. Momentum direction is now persisted on VGBSession (and the Node's
      `direction` slot is actually used), so vgb_step no longer resets it.
  1C. top_logprobs clamped to OpenAI's [0, 20] bound.
  2A. _get_sampled_blocks falls back to K parallel n=1 requests when the
      endpoint rejects n>1 (Anthropic / OpenRouter / Ollama / vLLM).
  2B. Authorization header omitted entirely when api_key is empty.
  2C. ε_V estimation breaks early on empty / invalid candidates.
"""

import os
import json
import uuid
import math
import random
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("VGB-Engine")

# =====================================================================
# §1  Mathematical Utilities
# =====================================================================

def logsumexp(values: List[float]) -> float:
    if not values:
        return float('-inf')
    max_val = max(values)
    if max_val == float('-inf'):
        return float('-inf')
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def sample_from_log_weights(log_weights: List[float]) -> int:
    if not log_weights:
        raise ValueError("Empty log_weights")
    max_lw = max(log_weights)
    if max_lw == float('-inf'):
        return random.randint(0, len(log_weights) - 1)
    weights = [math.exp(lw - max_lw) for lw in log_weights]
    total = sum(weights)
    if total <= 0:
        return random.randint(0, len(log_weights) - 1)
    return random.choices(range(len(log_weights)), weights=weights, k=1)[0]


def safe_log(x: float, eps: float = 1e-30) -> float:
    return math.log(max(x, eps))


# =====================================================================
# §1.1  Theorem-aware step count T  (Theorems 4.1, 4.2)
# =====================================================================

def compute_T_theoretical(
    H: int,
    epsilon_V: float,
    delta: float,
    mode: str = "uniform",
    safety_constant: float = 64.0,
) -> int:
    kappa = 1.0 + max(epsilon_V, 0.0)
    if mode == "uniform":
        T = int(math.ceil(safety_constant * (H ** 2) * (kappa ** 4) * math.log(max(1.0 / delta, 1.0))))
    elif mode == "average":
        T = int(math.ceil(safety_constant * (H ** 5) * (kappa ** 6) * (max(delta, 1e-6) ** -4)))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return max(T, 2 * H)


def max_reruns_for_leaf(H: int, kappa: float) -> int:
    return max(3, int(math.ceil(8 * kappa * H * math.log(2.0))))


# =====================================================================
# §1.2  ε_V estimation  (Assumption 4.1 / 4.2 verification)
# =====================================================================

async def estimate_epsilon_v_uniform(
    value_fn: 'ValueFunction',
    reward_fn: 'RewardFunction',
    base_model: 'BaseModel',
    prompt: str,
    horizon: int,
    num_probes: int = 8,
    num_rollouts_per_probe: int = 4,
) -> Tuple[float, Dict[str, Any]]:
    ratios: List[float] = []
    diagnostics: Dict[str, Any] = {"probes": []}

    for _ in range(num_probes):
        h = random.randint(1, max(1, horizon - 1))
        partial = ""
        for _ in range(h):
            cands = await base_model.get_candidates(
                prompt, partial, num_candidates=4, temperature=0.7,
                block_size=1, mode="sampled",
            )
            # FIX 2C: break on empty / invalid candidate lists so we don't
            # spin burning rate-limits on a failing endpoint.
            if not cands or not any(x[0] for x in cands):
                break
            cand = random.choice(cands)[0]
            if not cand:
                break
            partial += cand

        if not partial:
            continue

        v_hat = await value_fn.evaluate(prompt, partial, h)

        gen_len = max(1, horizon * 2)
        tasks = [base_model.complete(prompt, partial, gen_len) for _ in range(num_rollouts_per_probe)]
        completions = await asyncio.gather(*tasks)
        rewards = await asyncio.gather(*[reward_fn.evaluate(prompt, c) for c in completions])
        v_star_hat = sum(rewards) / len(rewards) if rewards else 0.0

        if v_star_hat > 1e-6 and v_hat > 1e-6:
            r = max(v_hat / v_star_hat, v_star_hat / v_hat)
            ratios.append(r)
            diagnostics["probes"].append({
                "h": h, "v_hat": v_hat, "v_star_hat": v_star_hat, "ratio": r,
            })

    if not ratios:
        return float("inf"), {**diagnostics, "warning": "No valid probes; ε_V unbounded."}

    epsilon_V_hat = max(ratios) - 1.0
    diagnostics["max_ratio"] = max(ratios)
    diagnostics["mean_ratio"] = sum(ratios) / len(ratios)
    diagnostics["epsilon_V_hat"] = epsilon_V_hat
    return max(epsilon_V_hat, 0.0), diagnostics


async def estimate_epsilon_v_average(
    value_fn: 'ValueFunction',
    reward_fn: 'RewardFunction',
    base_model: 'BaseModel',
    prompt: str,
    horizon: int,
    num_samples: int = 32,
    num_rollouts_per_sample: int = 2,
) -> Tuple[float, Dict[str, Any]]:
    fwd_ratios: List[float] = []
    inv_ratios: List[float] = []

    for _ in range(num_samples):
        h = random.randint(1, max(1, horizon - 1))
        partial = ""
        for _ in range(h):
            cands = await base_model.get_candidates(
                prompt, partial, num_candidates=4, temperature=0.7,
                block_size=1, mode="sampled",
            )
            # FIX 2C: same early-break protection.
            if not cands or not any(x[0] for x in cands):
                break
            cand = random.choice(cands)[0]
            if not cand:
                break
            partial += cand
        if not partial:
            continue

        v_hat = await value_fn.evaluate(prompt, partial, h)
        tasks = [base_model.complete(prompt, partial, horizon * 2) for _ in range(num_rollouts_per_sample)]
        completions = await asyncio.gather(*tasks)
        rewards = await asyncio.gather(*[reward_fn.evaluate(prompt, c) for c in completions])
        v_star_hat = sum(rewards) / len(rewards) if rewards else 0.0

        if v_star_hat > 1e-6 and v_hat > 1e-6:
            fwd_ratios.append(v_hat / v_star_hat)
            inv_ratios.append(v_star_hat / v_hat)

    if not fwd_ratios:
        return float("inf"), {"warning": "No valid samples."}

    eps_fwd = max(0.0, (sum(fwd_ratios) / len(fwd_ratios)) - 1.0)
    eps_inv = max(0.0, (sum(inv_ratios) / len(inv_ratios)) - 1.0)
    eps = max(eps_fwd, eps_inv)
    return eps, {
        "epsilon_V_hat": eps,
        "eps_fwd": eps_fwd,
        "eps_inv": eps_inv,
        "n_samples": len(fwd_ratios),
    }


# =====================================================================
# §2  Reward Functions
# =====================================================================

class RewardFunction:
    async def evaluate(self, prompt: str, completion: str) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return "base"


def _auth_headers(api_key: str) -> Dict[str, str]:
    """FIX 2B: only emit Authorization when an api_key is present.

    Empty Bearer headers ('Authorization: Bearer ') are rejected by some
    strict proxies / web servers (HTTP 400 / 401) even when the local
    endpoint (Ollama, vLLM) does not require auth.
    """
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class LLMJudgeReward(RewardFunction):
    def __init__(self, model: str, base_url: str, api_key: str, reward_prompt: str = ""):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0)
        self.reward_prompt = reward_prompt

    async def evaluate(self, prompt: str, completion: str) -> float:
        custom = f"\nAdditional evaluation criteria: {self.reward_prompt}\n" if self.reward_prompt else ""
        system_prompt = (
            "You are an outcome verifier. Evaluate whether the completion "
            "correctly and completely satisfies the prompt."
            f"{custom}"
            "\nFirst, reason step-by-step about correctness. "
            "Then on the FINAL line write EXACTLY: Score: X  "
            "where X is 1.0 if correct, 0.0 if incorrect, "
            "or a value in [0,1] for partial credit."
        )
        user_prompt = f"Prompt: {prompt}\n\nCompletion:\n{completion}\n\nEvaluate:"
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=_auth_headers(self.api_key),  # FIX 2B
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return 0.0

        for line in reversed(text.strip().split("\n")):
            m = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", line)
            if m:
                return max(0.0, min(1.0, float(m.group(1))))
        m = re.search(r"Score:\s*([0-9]*\.?[0-9]+)", text)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        m = re.search(r"([0-9]*\.?[0-9]+)", text.split("\n")[-1])
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
        return 0.0

    def describe(self) -> str:
        return f"LLM-judge ({self.model})"


class RegexReward(RewardFunction):
    def __init__(self, pattern: str):
        self.pattern = pattern

    async def evaluate(self, prompt: str, completion: str) -> float:
        return 1.0 if re.fullmatch(self.pattern, completion, re.DOTALL) else 0.0

    def describe(self) -> str:
        return f"regex({self.pattern!r})"


class ContainsReward(RewardFunction):
    def __init__(self, substring: str, case_sensitive: bool = False):
        self.substring = substring
        self.case_sensitive = case_sensitive

    async def evaluate(self, prompt: str, completion: str) -> float:
        c = completion if self.case_sensitive else completion.lower()
        s = self.substring if self.case_sensitive else self.substring.lower()
        return 1.0 if s in c else 0.0

    def describe(self) -> str:
        return f"contains({self.substring!r})"


class NotContainsReward(RewardFunction):
    def __init__(self, substring: str, case_sensitive: bool = False):
        self.substring = substring
        self.case_sensitive = case_sensitive

    async def evaluate(self, prompt: str, completion: str) -> float:
        c = completion if self.case_sensitive else completion.lower()
        s = self.substring if self.case_sensitive else self.substring.lower()
        return 1.0 if s not in c else 0.0

    def describe(self) -> str:
        return f"not_contains({self.substring!r})"


class CompositeReward(RewardFunction):
    def __init__(self, rewards: List[RewardFunction]):
        self.rewards = rewards

    async def evaluate(self, prompt: str, completion: str) -> float:
        vals = await asyncio.gather(*[r.evaluate(prompt, completion) for r in self.rewards])
        return math.prod(vals)

    def describe(self) -> str:
        return " ∧ ".join(r.describe() for r in self.rewards)


# =====================================================================
# §3  Base Models
# =====================================================================

class BaseModel:
    """Common interface for base models (sample π_ref, compute densities)."""

    async def get_candidates(
        self, prompt: str, context: str, num_candidates: int,
        temperature: float, block_size: int = 1, mode: str = "topk",
    ) -> List[Tuple[str, float]]:
        raise NotImplementedError

    async def complete(self, prompt: str, context: str, max_tokens: int) -> str:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


class APIBaseModel(BaseModel):
    """
    OpenAI-compatible API base model.

    Auto-detects logprobs support; if unavailable, transparently switches
    to sampled candidate mode (block-level, no logprob dependency).
    """

    # OpenAI Chat Completions hard limit on top_logprobs.
    _MAX_TOP_LOGPROBS = 20

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str = "", use_chat_template: bool = True):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=180.0)
        self.use_chat_template = use_chat_template
        self._logprobs_supported: Optional[bool] = None
        self._approx_tokens_per_word = 1.3

    def _headers(self) -> Dict[str, str]:
        """FIX 2B: shared header builder — omit Authorization if no api_key."""
        return _auth_headers(self.api_key)

    def _build_messages(self, prompt: str, context: str) -> List[Dict[str, str]]:
        """FIX 1A: prefill assistant generation instead of merging context
        into the user prompt.

        For chat / instruct-tuned models, ending the messages list with an
        assistant message whose content is the partial `context` causes the
        API to treat `context` as the model's own in-progress generation and
        to continue from it — exactly the behaviour we need for VGB.

        For raw base models we fall back to plain concatenation since there
        is no role structure to exploit.
        """
        if self.use_chat_template:
            messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
            if context:
                messages.append({"role": "assistant", "content": context})
            return messages
        # Raw continuation for base models
        return [{"role": "user", "content": f"{prompt}\n{context}"}]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * self._approx_tokens_per_word))

    async def _probe_logprobs(self) -> bool:
        if self._logprobs_supported is not None:
            return self._logprobs_supported
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),  # FIX 2B
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            )
            data = resp.json()
            content = data["choices"][0].get("logprobs", {}).get("content", [])
            self._logprobs_supported = bool(content) and bool(content[0].get("top_logprobs"))
        except Exception:
            self._logprobs_supported = False
        return self._logprobs_supported

    async def get_candidates(
        self, prompt: str, context: str, num_candidates: int,
        temperature: float, block_size: int = 1, mode: str = "topk",
    ) -> List[Tuple[str, float]]:
        if mode == "topk" and block_size <= 1:
            if await self._probe_logprobs():
                return await self._get_topk_tokens(prompt, context, num_candidates)
            return await self._get_sampled_blocks(prompt, context, num_candidates, temperature, 1)
        return await self._get_sampled_blocks(prompt, context, num_candidates, temperature, max(block_size, 1))

    async def _get_topk_tokens(self, prompt: str, context: str, K: int) -> List[Tuple[str, float]]:
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),  # FIX 2B
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, context),
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    # FIX 1C: OpenAI caps top_logprobs at 20. Anything larger
                    # returns HTTP 400 and kills the candidate step entirely.
                    "top_logprobs": min(K, self._MAX_TOP_LOGPROBS),
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return [("", -100.0)]

        content = data["choices"][0].get("logprobs", {}).get("content", [])
        if not content:
            text = data["choices"][0]["message"].get("content", "")
            return [(text, -1.0)] if text else []

        top_lp = content[0].get("top_logprobs", [])
        seen: Dict[str, float] = {}
        for item in top_lp:
            tok = item.get("token", "")
            lp = item.get("logprob", -100.0)
            if tok in seen:
                seen[tok] = logsumexp([seen[tok], lp])
            else:
                seen[tok] = lp
        candidates = list(seen.items())
        return candidates if candidates else [("", -100.0)]

    def _parse_sampled_choices(
        self, choices: List[Dict[str, Any]], K: int,
    ) -> List[Tuple[str, float]]:
        """Parse a list of completion choices into (text, sum_logprob) tuples.

        Shared between the batch (n=K) path and the parallel (n=1) fallback.
        """
        result: List[Tuple[str, float]] = []
        for choice in choices:
            block_text = (choice.get("message") or {}).get("content", "") or ""
            if not block_text:
                continue
            lp_content = (choice.get("logprobs") or {}).get("content", [])
            if lp_content:
                sum_lp = sum(
                    (it.get("logprob", -10.0) if it else -10.0)
                    for it in lp_content
                )
            else:
                # No logprobs returned: uniform prior (still usable for VGB-sampled
                # since π_ref is implicitly encoded by sampling frequency).
                sum_lp = -math.log(max(K, 1))
            result.append((block_text, sum_lp))
        return result

    async def _get_sampled_blocks(
        self, prompt: str, context: str, K: int,
        temperature: float, block_size: int,
    ) -> List[Tuple[str, float]]:
        """FIX 2A: try batch (n=K) first; on failure, fall back to K parallel
        n=1 requests. Many providers (Anthropic, OpenRouter, Ollama, some
        vLLM gateways) reject n>1 or logprobs+n>1 outright — the previous
        code would silently return [("", -100.0)] and stall VGB.
        """
        headers = self._headers()  # FIX 2B
        messages = self._build_messages(prompt, context)
        payload_base: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": block_size,
            "temperature": temperature,
            "logprobs": True,
        }

        # --- Attempt 1: batch n=K ---------------------------------------
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={**payload_base, "n": K},
            )
            resp.raise_for_status()
            data = resp.json()
            result = self._parse_sampled_choices(data.get("choices", []), K)
            if result:
                return result
            # Empty result (e.g. all choices blank) — fall through to parallel.
        except Exception:
            # Batch path rejected (4xx/5xx or parse error) — fall through.
            pass

        # --- Attempt 2: K parallel n=1 requests -------------------------
        tasks = [
            self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={**payload_base, "n": 1},
            )
            for _ in range(K)
        ]
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            return [("", -100.0)]

        result: List[Tuple[str, float]] = []
        for resp in responses:
            if isinstance(resp, Exception):
                continue
            try:
                resp.raise_for_status()
                data = resp.json()
                parsed = self._parse_sampled_choices(data.get("choices", []), 1)
                result.extend(parsed)
            except Exception:
                continue
        return result if result else [("", -100.0)]

    async def complete(self, prompt: str, context: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),  # FIX 2B
                json={
                    "model": self.model,
                    "messages": self._build_messages(prompt, context),
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""


class LocalBaseModel(BaseModel):
    """
    HuggingFace local model. Applies the tokenizer's chat template
    for instruct-tuned models when `apply_chat_template=True`.
    """

    def __init__(self, model_name: str, apply_chat_template: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.apply_chat_template = apply_chat_template
        self._has_chat_template = (
            apply_chat_template
            and getattr(self.tokenizer, "chat_template", None) is not None
        )

    def _format_input(self, prompt: str, context: str) -> str:
        """FIX 1A (local side): apply chat template with an assistant prefill
        message when `context` is non-empty, instead of jamming context into
        the user message. With `add_generation_prompt=False` the rendered
        template ends exactly at the assistant message's content, so the
        model continues from `context` as a genuine continuation.
        """
        if self._has_chat_template:
            msgs: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
            if context:
                msgs.append({"role": "assistant", "content": context})
                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return prompt + context

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    async def get_candidates(
        self, prompt: str, context: str, num_candidates: int,
        temperature: float, block_size: int = 1, mode: str = "topk",
    ) -> List[Tuple[str, float]]:
        full_text = self._format_input(prompt, context)
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        if block_size <= 1 and mode == "topk":
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]
                log_probs = self.torch.nn.functional.log_softmax(logits, dim=-1)
                topk = self.torch.topk(log_probs, min(num_candidates, log_probs.shape[0]))
            candidates = []
            for i in range(topk.indices.shape[0]):
                tok_id = topk.indices[i].item()
                tok_str = self.tokenizer.decode([tok_id])
                candidates.append((tok_str, topk.values[i].item()))
            return candidates
        else:
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=block_size,
                    num_return_sequences=num_candidates,
                    do_sample=True,
                    temperature=max(temperature, 1e-6),
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            candidates = []
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            for i in range(num_candidates):
                gen = outputs.sequences[i][input_len:]
                end = len(gen)
                for j in range(len(gen)):
                    t = gen[j].item()
                    if (eos_id is not None and t == eos_id) or \
                       (pad_id is not None and t == pad_id):
                        end = j if t == pad_id else j + 1
                        break
                actual = gen[:end]
                text = self.tokenizer.decode(actual, skip_special_tokens=True)
                if not text:
                    continue
                num_steps = min(len(actual), len(outputs.scores))
                lp = 0.0
                for s in range(num_steps):
                    lp_dist = self.torch.nn.functional.log_softmax(
                        outputs.scores[s][i], dim=-1
                    )
                    lp += lp_dist[actual[s]].item()
                candidates.append((text, lp))
            return candidates if candidates else [("", -100.0)]

    async def complete(self, prompt: str, context: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        full_text = self._format_input(prompt, context)
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# =====================================================================
# §4  Value Functions
# =====================================================================

class ValueFunction:
    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        raise NotImplementedError

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        return await self.evaluate(prompt, completion, depth=-1)


class MCRolloutValue(ValueFunction):
    def __init__(self, base_model: BaseModel, reward_fn: RewardFunction,
                 num_rollouts: int, generation_length: int):
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.num_rollouts = max(1, num_rollouts)
        self.generation_length = generation_length

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        partial_tokens = self.base_model.count_tokens(partial) if partial else 0
        remaining = max(1, self.generation_length - partial_tokens)
        tasks = [self.base_model.complete(prompt, partial, remaining) for _ in range(self.num_rollouts)]
        completions = await asyncio.gather(*tasks)
        rewards = await asyncio.gather(*[self.reward_fn.evaluate(prompt, c) for c in completions])
        return sum(rewards) / len(rewards) if rewards else 0.0

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        return await self.reward_fn.evaluate(prompt, completion)


class TiltValue(ValueFunction):
    def __init__(self, reward_fn: RewardFunction, alpha: float, horizon: int):
        self.reward_fn = reward_fn
        self.alpha = alpha
        self.horizon = horizon

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        r = await self.reward_fn.evaluate(prompt, partial)
        decay = self.alpha ** max(0, self.horizon - depth)
        return r * decay

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        return await self.reward_fn.evaluate(prompt, completion)


class KLRegularizedValue(ValueFunction):
    def __init__(self, q_fn: 'QFunction', beta: float):
        self.q_fn = q_fn
        self.beta = max(beta, 1e-6)

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        q = await self.q_fn.evaluate(prompt, partial, depth)
        return math.exp(q / self.beta)

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        q = await self.q_fn.evaluate(prompt, completion, depth=-1)
        return math.exp(q / self.beta)


class QFunction:
    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        raise NotImplementedError


class LLMJudgeQFunction(QFunction):
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0)

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        system_prompt = (
            "Estimate how promising this partial completion is "
            "for correctly answering the prompt. "
            "Respond with a single number in [0, 10]."
        )
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=_auth_headers(self.api_key),  # FIX 2B
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Prompt: {prompt}\nPartial:\n{partial}\nScore:"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 10,
                },
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"([0-9]*\.?[0-9]+)", text)
            return min(10.0, max(0.0, float(m.group(1)))) if m else 5.0
        except Exception:
            return 5.0


# =====================================================================
# §5  VGB Tree and Session
# =====================================================================

class Node:
    __slots__ = ("text", "parent", "children", "depth",
                 "value_score", "log_ref_prob", "direction")

    def __init__(self, text: str, parent: Optional['Node'] = None,
                 depth: int = 0, direction: int = 0):
        self.text = text
        self.parent = parent
        self.children: Dict[str, 'Node'] = {}
        self.depth = depth
        self.value_score: Optional[float] = None
        self.log_ref_prob: float = 0.0
        # FIX 1B: Momentum direction (Hayes-Sinclair): +1 descending,
        # -1 ascending. Now actually used and persisted via the session.
        self.direction = direction


class VGBSession:
    """Holds all state for one VGB generation session."""

    def __init__(self, session_id: str, prompt: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.prompt = prompt
        self.config = config
        self.root = Node(text="", depth=0)
        self.current_node = self.root
        self.step_count = 0
        self.history: List[str] = []
        self.estimated_epsilon_v: Optional[float] = None

        # FIX 1B: Session-level momentum direction so vgb_step does not
        # reset it on every call. +1 = descending by default.
        self.momentum_direction: int = 1

        horizon = config["horizon"]
        gen_len = config.get("generation_length", horizon * config.get("block_size", 1))

        self.reward_fn = self._build_reward_fn(config)
        self.base_model = self._build_base_model(config)

        value_type = config.get("value_type", "tilt")
        if value_type == "mc_rollout":
            value_base = self._build_value_model(config) or self.base_model
            self.value_fn: ValueFunction = MCRolloutValue(
                value_base, self.reward_fn,
                config.get("num_rollouts", 2), gen_len,
            )
        elif value_type == "tilt":
            self.value_fn = TiltValue(
                self.reward_fn,
                config.get("tilt_alpha", 0.3),
                horizon,
            )
        elif value_type == "kl_regularized":
            q_fn = LLMJudgeQFunction(
                config.get("value_model") or config["base_model"],
                config.get("base_url", "https://api.openai.com/v1"),
                os.getenv("OPENAI_API_KEY", ""),
            )
            self.value_fn = KLRegularizedValue(q_fn, config.get("beta", 1.0))
        else:
            self.value_fn = TiltValue(self.reward_fn, config.get("tilt_alpha", 0.3), horizon)

    @staticmethod
    def _build_base_model(config: Dict[str, Any]) -> BaseModel:
        if config.get("use_local_model", False):
            return LocalBaseModel(
                config["model_name"],
                apply_chat_template=config.get("apply_chat_template", True),
            )
        return APIBaseModel(
            config["base_model"],
            config.get("base_url", "https://api.openai.com/v1"),
            use_chat_template=config.get("apply_chat_template", True),
        )

    @staticmethod
    def _build_value_model(config: Dict[str, Any]) -> Optional[BaseModel]:
        vm = config.get("value_model", "")
        if not vm:
            return None
        if config.get("value_model_local", False):
            return LocalBaseModel(vm, apply_chat_template=True)
        return APIBaseModel(vm, config.get("base_url", "https://api.openai.com/v1"))

    @staticmethod
    def _build_reward_fn(config: Dict[str, Any]) -> RewardFunction:
        rt = config.get("reward_type", "llm_judge")
        if rt == "llm_judge":
            return LLMJudgeReward(
                config.get("value_model") or config["base_model"],
                config.get("base_url", "https://api.openai.com/v1"),
                os.getenv("OPENAI_API_KEY", ""),
                config.get("reward_prompt", ""),
            )
        elif rt == "regex":
            return RegexReward(config.get("reward_regex", ".*"))
        elif rt == "contains":
            return ContainsReward(config.get("reward_substring", ""))
        elif rt == "not_contains":
            return NotContainsReward(config.get("reward_substring", ""))
        elif rt == "composite":
            parts = config.get("reward_parts", [])
            fns = [VGBSession._build_reward_fn(p) for p in parts]
            return CompositeReward(fns)
        else:
            return LLMJudgeReward(
                config["base_model"],
                config.get("base_url", "https://api.openai.com/v1"),
                os.getenv("OPENAI_API_KEY", ""),
            )


# =====================================================================
# §6  VGB Engine  (Algorithm 1 / Algorithm 3)
# =====================================================================

class VGBEngine:
    """
    Implements the VGB random walk (Algorithms 1 & 3).
    """

    def __init__(self, session: VGBSession):
        self.session = session

    async def _get_node_value(self, node: Node) -> float:
        if node.value_score is not None:
            return node.value_score
        if node.depth == 0:
            node.value_score = 0.5
            return 0.5
        cfg = self.session.config
        is_leaf = node.depth >= cfg["horizon"]
        if is_leaf and cfg.get("use_ground_truth_leaf", True):
            score = await self.session.value_fn.evaluate_leaf(self.session.prompt, node.text)
        else:
            score = await self.session.value_fn.evaluate(self.session.prompt, node.text, node.depth)
        node.value_score = max(score, 0.0)
        return node.value_score

    async def step(self) -> str:
        s = self.session
        cfg = s.config
        current = s.current_node
        h = current.depth
        H = cfg["horizon"]

        if cfg.get("use_laziness", False):
            if random.random() < 0.5:
                s.step_count += 1
                s.history.append(f"Step {s.step_count}: lazy stay at depth {h}")
                return "stay"

        log_weights: List[float] = []
        neighbor_nodes: List[Node] = []
        action_types: List[str] = []

        if h > 0 and current.parent is not None:
            val = await self._get_node_value(current)
            lw = safe_log(val)
            if cfg.get("candidate_mode", "topk") == "sampled":
                lw += safe_log(cfg["num_candidates"])
            log_weights.append(lw)
            neighbor_nodes.append(current.parent)
            action_types.append("backtrack")

        if h < H:
            candidates = await s.base_model.get_candidates(
                prompt=s.prompt,
                context=current.text,
                num_candidates=cfg["num_candidates"],
                temperature=cfg["temperature"],
                block_size=cfg.get("block_size", 1),
                mode=cfg.get("candidate_mode", "topk"),
            )

            candidate_mode = cfg.get("candidate_mode", "topk")
            for action_text, log_pi_ref in candidates:
                if not action_text:
                    continue
                child_key = action_text
                if child_key not in current.children:
                    child = Node(
                        text=current.text + action_text,
                        parent=current,
                        depth=h + 1,
                    )
                    child.log_ref_prob = log_pi_ref
                    current.children[child_key] = child

                child_node = current.children[child_key]
                child_val = await self._get_node_value(child_node)
                if child_val <= 0:
                    continue

                if candidate_mode == "sampled":
                    fw = safe_log(child_val)
                else:
                    fw = log_pi_ref + safe_log(child_val)

                log_weights.append(fw)
                neighbor_nodes.append(child_node)
                action_types.append("forward")

        if not neighbor_nodes:
            s.step_count += 1
            s.history.append(f"Step {s.step_count}: stuck at depth {h}")
            return "stay"

        idx = sample_from_log_weights(log_weights)
        selected_node = neighbor_nodes[idx]
        selected_action = action_types[idx]

        s.current_node = selected_node
        s.step_count += 1
        v = selected_node.value_score
        s.history.append(
            f"Step {s.step_count}: {selected_action} → depth "
            f"{selected_node.depth}  val={v:.4f}" if v is not None else
            f"Step {s.step_count}: {selected_action} → depth "
            f"{selected_node.depth}  val=?"
        )
        return selected_action

    async def run(self) -> Dict[str, Any]:
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        max_steps = cfg.get("max_steps", max(200, H * 40))

        while s.current_node.depth < H and s.step_count < max_steps:
            await self.step()

        reached_leaf = s.current_node.depth >= H
        return {
            "session_id": s.session_id,
            "generation": s.current_node.text,
            "total_steps": s.step_count,
            "final_depth": s.current_node.depth,
            "reached_leaf": reached_leaf,
        }

    async def run_theoretical(
        self,
        T: Optional[int] = None,
        delta: float = 0.1,
        epsilon_V: Optional[float] = None,
        error_mode: str = "uniform",
    ) -> Dict[str, Any]:
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        cfg["use_laziness"] = True

        if epsilon_V is None:
            if s.estimated_epsilon_v is None:
                eps, diag = await estimate_epsilon_v_uniform(
                    s.value_fn, s.reward_fn, s.base_model, s.prompt, H,
                    num_probes=cfg.get("eps_probes", 6),
                    num_rollouts_per_probe=cfg.get("eps_rollouts", 2),
                )
                s.estimated_epsilon_v = eps
            epsilon_V = s.estimated_epsilon_v

        kappa = 1.0 + max(epsilon_V, 0.0)

        if T is None or T <= 0:
            T = compute_T_theoretical(H, epsilon_V, delta, mode=error_mode)

        max_attempts = max_reruns_for_leaf(H, kappa)
        attempts_used = 0
        for attempt in range(max_attempts):
            attempts_used = attempt + 1
            s.current_node = s.root
            s.step_count = 0
            for _ in range(T):
                await self.step()
            if s.current_node.depth >= H:
                break
            for n in _walk_tree(s.root):
                n.value_score = None

        reached_leaf = s.current_node.depth >= H
        return {
            "session_id": s.session_id,
            "generation": s.current_node.text,
            "total_steps": s.step_count,
            "final_depth": s.current_node.depth,
            "reached_leaf": reached_leaf,
            "attempts": attempts_used,
            "T_used": T,
            "epsilon_V": epsilon_V,
            "kappa": kappa,
            "delta": delta,
            "error_mode": error_mode,
            "max_attempts": max_attempts,
        }


# =====================================================================
# §6.1  VGB-Momentum  (Hayes & Sinclair 2010; Appendix E.1)
# =====================================================================

class VGBMomentumEngine(VGBEngine):
    """
    VGB with momentum (Hayes-Sinclair lifting). Each step carries a
    "direction" bit: +1 (descending / forward) or -1 (ascending /
    backtracking). The chain is a directed cycle over (node, direction)
    pairs; crossing flows cancel, reducing diffusive behavior while
    preserving the same stationary distribution µ over nodes.

    FIX 1B: direction state is now read from / written to
    `session.momentum_direction` (and mirrored onto `Node.direction`
    for inspection) so that the per-call `VGBMomentumEngine` instances
    created by `vgb_step` no longer reset momentum to +1 on every call.
    """

    def __init__(self, session: VGBSession):
        super().__init__(session)
        # Initialise from session state (default +1 descending).
        self.session.momentum_direction = self.session.momentum_direction or 1

    @property
    def direction(self) -> int:
        return self.session.momentum_direction

    @direction.setter
    def direction(self, value: int) -> None:
        self.session.momentum_direction = value
        # Mirror onto current node for inspection / debugging.
        if self.session.current_node is not None:
            self.session.current_node.direction = value

    async def step(self) -> str:
        s = self.session
        cfg = s.config
        current = s.current_node
        h = current.depth
        H = cfg["horizon"]

        if cfg.get("use_laziness", False):
            if random.random() < 0.5:
                s.step_count += 1
                s.history.append(f"Step {s.step_count}: lazy stay at depth {h}")
                return "stay"

        log_weights: List[float] = []
        neighbor_nodes: List[Node] = []
        action_types: List[str] = []
        next_directions: List[int] = []

        if h > 0 and current.parent is not None:
            val = await self._get_node_value(current)
            lw = safe_log(val)
            if cfg.get("candidate_mode", "topk") == "sampled":
                lw += safe_log(cfg["num_candidates"])
            # Momentum: if currently ascending, boost backtrack weight.
            if self.direction == -1:
                lw += math.log(2.0)
            log_weights.append(lw)
            neighbor_nodes.append(current.parent)
            action_types.append("backtrack")
            next_directions.append(-1)

        if h < H:
            candidates = await s.base_model.get_candidates(
                prompt=s.prompt,
                context=current.text,
                num_candidates=cfg["num_candidates"],
                temperature=cfg["temperature"],
                block_size=cfg.get("block_size", 1),
                mode=cfg.get("candidate_mode", "topk"),
            )

            candidate_mode = cfg.get("candidate_mode", "topk")
            for action_text, log_pi_ref in candidates:
                if not action_text:
                    continue
                child_key = action_text
                if child_key not in current.children:
                    child = Node(
                        text=current.text + action_text,
                        parent=current,
                        depth=h + 1,
                    )
                    child.log_ref_prob = log_pi_ref
                    current.children[child_key] = child

                child_node = current.children[child_key]
                child_val = await self._get_node_value(child_node)
                if child_val <= 0:
                    continue

                if candidate_mode == "sampled":
                    fw = safe_log(child_val)
                else:
                    fw = log_pi_ref + safe_log(child_val)
                # Momentum: if currently descending, boost forward weight.
                if self.direction == 1:
                    fw += math.log(2.0)
                log_weights.append(fw)
                neighbor_nodes.append(child_node)
                action_types.append("forward")
                next_directions.append(1)

        if not neighbor_nodes:
            s.step_count += 1
            s.history.append(f"Step {s.step_count}: stuck at depth {h}")
            return "stay"

        idx = sample_from_log_weights(log_weights)
        selected_node = neighbor_nodes[idx]
        selected_action = action_types[idx]
        # FIX 1B: persist direction on the session (and mirror onto the node).
        self.direction = next_directions[idx]

        s.current_node = selected_node
        s.step_count += 1
        v = selected_node.value_score
        s.history.append(
            f"Step {s.step_count}: {selected_action} → depth "
            f"{selected_node.depth}  val={v:.4f}  dir={self.direction}"
            if v is not None else
            f"Step {s.step_count}: {selected_action} → depth "
            f"{selected_node.depth}  val=?  dir={self.direction}"
        )
        return selected_action


# =====================================================================
# §7  Baselines
# =====================================================================

class ActionLevelRS:
    """Action-Level Rejection Sampling with V̂ (Algorithm 7)."""

    def __init__(self, session: VGBSession):
        self.session = session

    async def generate(self) -> Dict[str, Any]:
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        max_restarts = cfg.get("max_restarts", 10)
        text = ""
        total_steps = 0
        success = False
        h_final = 0

        for restart in range(max_restarts):
            text = ""
            success = True
            for h in range(H):
                h_final = h
                candidates = await s.base_model.get_candidates(
                    prompt=s.prompt,
                    context=text,
                    num_candidates=cfg["num_candidates"],
                    temperature=cfg["temperature"],
                    block_size=cfg.get("block_size", 1),
                    mode=cfg.get("candidate_mode", "topk"),
                )
                log_weights = []
                valid = []
                for action, log_pi_ref in candidates:
                    if not action:
                        continue
                    child_text = text + action
                    is_leaf = (h + 1 >= H)
                    if is_leaf and cfg.get("use_ground_truth_leaf", True):
                        val = await s.value_fn.evaluate_leaf(s.prompt, child_text)
                    else:
                        val = await s.value_fn.evaluate(s.prompt, child_text, h + 1)
                    if val <= 0:
                        continue
                    mode = cfg.get("candidate_mode", "topk")
                    if mode == "sampled":
                        lw = safe_log(val)
                    else:
                        lw = log_pi_ref + safe_log(val)
                    log_weights.append(lw)
                    valid.append((action, child_text))

                if not valid:
                    success = False
                    break

                idx = sample_from_log_weights(log_weights)
                text = valid[idx][1]
                total_steps += 1

            if success:
                break

        return {
            "generation": text,
            "total_steps": total_steps,
            "final_depth": H if success else h_final,
            "reached_leaf": success,
        }


class OutcomeLevelRS:
    """Outcome-Level Rejection Sampling (Algorithm 6)."""

    def __init__(self, session: VGBSession):
        self.session = session

    async def generate(self, max_attempts: int = 200, threshold: float = 0.5) -> Dict[str, Any]:
        s = self.session
        gen_len = s.config.get(
            "generation_length",
            s.config["horizon"] * s.config.get("block_size", 1),
        )
        completion = ""
        for attempt in range(max_attempts):
            completion = await s.base_model.complete(s.prompt, "", max(gen_len, 1))
            reward = await s.reward_fn.evaluate(s.prompt, completion)
            if reward >= threshold:
                return {
                    "generation": completion,
                    "attempts": attempt + 1,
                    "reward": reward,
                    "reached_leaf": True,
                }
        return {
            "generation": completion,
            "attempts": max_attempts,
            "reward": 0.0,
            "reached_leaf": False,
        }


# =====================================================================
# §8  Server State
# =====================================================================

SESSIONS: Dict[str, VGBSession] = {}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "horizon": 5,
    "block_size": 1,
    "generation_length": 75,
    "temperature": 0.7,
    "num_candidates": 8,
    "max_steps": 200,

    "value_type": "tilt",
    "num_rollouts": 2,
    "tilt_alpha": 0.3,
    "beta": 1.0,
    "use_ground_truth_leaf": True,

    "candidate_mode": "topk",
    "use_laziness": False,
    # FIX 1B: persist use_momentum so vgb_step picks the right engine.
    "use_momentum": False,

    "base_model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1",
    "use_local_model": False,
    "model_name": "",
    "apply_chat_template": True,

    "value_model": "",
    "value_model_local": False,

    "reward_type": "llm_judge",
    "reward_prompt": "",
    "reward_regex": ".*",
    "reward_substring": "",

    "eps_probes": 6,
    "eps_rollouts": 2,
}


def _default_config() -> Dict[str, Any]:
    return dict(_DEFAULT_CONFIG)


def _set_default_config(updates: Dict[str, Any]) -> None:
    _DEFAULT_CONFIG.update(updates)


def _build_engine_for_session(session: VGBSession) -> VGBEngine:
    """FIX 1B: centralised engine factory so vgb_step honours use_momentum
    and inherits persisted momentum direction from the session."""
    if session.config.get("use_momentum", False):
        return VGBMomentumEngine(session)
    return VGBEngine(session)


# =====================================================================
# §9  MCP Tools
# =====================================================================

@mcp.tool()
async def vgb_generate(
    prompt: str,
    horizon: int = 5,
    block_size: int = 1,
    generation_length: int = 75,
    temperature: float = 0.7,
    num_candidates: int = 8,
    num_rollouts: int = 2,
    value_type: str = "tilt",
    tilt_alpha: float = 0.3,
    beta: float = 1.0,
    use_ground_truth_leaf: bool = True,
    candidate_mode: str = "topk",
    base_model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
    run_mode: str = "practical",
    step_count_T: int = 0,
    delta: float = 0.1,
    epsilon_V: float = -1.0,
    error_mode: str = "uniform",
    use_momentum: bool = False,
    apply_chat_template: bool = True,
    value_model: str = "",
    value_model_local: bool = False,
) -> str:
    """
    Generate text using VGB (Value-Guided Sampling with Stochastic Backtracking).

    Implements Algorithm 1 from "Taming Imperfect Process Verifiers"
    (Rohatgi et al., 2025). Theorem-aware T is computed when run_mode="theoretical".

    Defaults:
      - value_type="tilt" — no rollouts, works with any model instantly (§5.3).
      - candidate_mode="topk" — auto-falls-back to "sampled" if API lacks logprobs.
      - apply_chat_template=True — applies tokenizer chat template for instruct models.

    Args:
        prompt:              Input prompt.
        horizon:             H — tree depth (number of actions).
        block_size:          Tokens per action (1 = token-level).
        generation_length:   Approx total tokens.
        temperature:         Sampling temperature for π_ref.
        num_candidates:      K — candidates per forward step.
        num_rollouts:        MC rollouts per value estimate (mc_rollout only).
        value_type:          "tilt" (default) | "mc_rollout" | "kl_regularized".
        tilt_alpha:          α for tilt value fn (§5.3).
        beta:                Temperature for KL-regularized setting.
        use_ground_truth_leaf: Use τ at leaves (Algorithm 1 Line 1).
        candidate_mode:      "topk" | "sampled". Auto-fallback if logprobs missing.
        base_model:          Model name for API calls.
        base_url:            API base URL.
        reward_type:         "llm_judge" | "regex" | "contains" | "not_contains" | "composite".
        reward_prompt:       Extra instructions for LLM judge.
        reward_regex:        Regex pattern for regex reward.
        reward_substring:    Substring for contains/not_contains reward.
        run_mode:            "practical" (run-to-leaf) | "theoretical" (fixed T, re-run).
        step_count_T:        If >0 and run_mode="theoretical", run exactly T steps.
        delta:               Target error for theoretical mode.
        epsilon_V:           Known ε_V. If <0, attempt empirical estimation.
        error_mode:          "uniform" (Thm 4.1) | "average" (Thm 4.2).
        use_momentum:        Use VGB-Momentum (Hayes-Sinclair, Appendix E.1).
        apply_chat_template: Apply chat template for instruct-tuned models.
        value_model:         Optional separate (small) verifier model name.
        value_model_local:   If True, load value_model as a local HF model.
    """
    session_id = str(uuid.uuid4())
    cfg = _default_config()
    cfg.update({
        "horizon": horizon,
        "block_size": block_size,
        "generation_length": generation_length,
        "temperature": temperature,
        "num_candidates": num_candidates,
        "num_rollouts": num_rollouts,
        "value_type": value_type,
        "tilt_alpha": tilt_alpha,
        "beta": beta,
        "use_ground_truth_leaf": use_ground_truth_leaf,
        "candidate_mode": candidate_mode,
        "base_model": base_model,
        "base_url": base_url,
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
        "apply_chat_template": apply_chat_template,
        "value_model": value_model,
        "value_model_local": value_model_local,
        # FIX 1B: persist use_momentum on the session config.
        "use_momentum": use_momentum,
    })

    session = VGBSession(session_id, prompt, cfg)
    engine = _build_engine_for_session(session)

    if run_mode == "theoretical":
        eps_arg = epsilon_V if epsilon_V >= 0 else None
        T_arg = step_count_T if step_count_T > 0 else None
        result = await engine.run_theoretical(
            T=T_arg, delta=delta, epsilon_V=eps_arg, error_mode=error_mode,
        )
    else:
        result = await engine.run()

    SESSIONS[session_id] = session
    return json.dumps(result, indent=2)


@mcp.tool()
async def vgb_step(session_id: str) -> str:
    """Execute a single step of the VGB random walk for an active session.

    FIX 1B: the engine is now built via _build_engine_for_session, which
    honours `use_momentum` and reads the persisted `momentum_direction`
    from the session — so momentum state survives across calls.
    """
    if session_id not in SESSIONS:
        return json.dumps({"error": "Session not found."})
    session = SESSIONS[session_id]
    engine = _build_engine_for_session(session)  # FIX 1B
    action = await engine.step()
    node = session.current_node
    v = node.value_score
    return json.dumps({
        "session_id": session_id,
        "action_taken": action,
        "current_depth": node.depth,
        "current_text": node.text[-200:],
        "value": round(v, 4) if v is not None else None,
        "momentum_direction": session.momentum_direction,  # FIX 1B (visibility)
        "use_momentum": session.config.get("use_momentum", False),
    }, indent=2)


@mcp.tool()
async def vgb_status(session_id: str) -> str:
    """Get the current status of a VGB session."""
    if session_id not in SESSIONS:
        return json.dumps({"error": "Session not found."})
    s = SESSIONS[session_id]
    node = s.current_node
    v = node.value_score
    return json.dumps({
        "session_id": session_id,
        "prompt": s.prompt[:200],
        "current_depth": node.depth,
        "horizon": s.config["horizon"],
        "current_text": node.text,
        "value": round(v, 4) if v is not None else None,
        "step_count": s.step_count,
        "num_children_explored": sum(len(n.children) for n in _walk_tree(s.root)),
        "reward_type": s.config.get("reward_type"),
        "value_type": s.config.get("value_type"),
        "block_size": s.config.get("block_size"),
        "candidate_mode": s.config.get("candidate_mode"),
        "use_ground_truth_leaf": s.config.get("use_ground_truth_leaf"),
        "use_momentum": s.config.get("use_momentum", False),         # FIX 1B
        "momentum_direction": s.momentum_direction,                   # FIX 1B
        "estimated_epsilon_v": s.estimated_epsilon_v,
        "recent_history": s.history[-5:],
    }, indent=2)


@mcp.tool()
async def verify_assumptions(
    prompt: str,
    horizon: int = 5,
    base_model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
    value_type: str = "tilt",
    tilt_alpha: float = 0.3,
    num_rollouts: int = 2,
    generation_length: int = 75,
    num_probes: int = 8,
    num_rollouts_per_probe: int = 4,
    num_average_samples: int = 32,
    delta: float = 0.1,
    candidate_mode: str = "sampled",
    apply_chat_template: bool = True,
) -> str:
    """
    Empirically estimate ε_V and compute the theorem-required step count T.
    """
    session_id = str(uuid.uuid4())
    cfg = _default_config()
    cfg.update({
        "horizon": horizon,
        "generation_length": generation_length,
        "base_model": base_model,
        "base_url": base_url,
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
        "value_type": value_type,
        "tilt_alpha": tilt_alpha,
        "num_rollouts": num_rollouts,
        "candidate_mode": candidate_mode,
        "apply_chat_template": apply_chat_template,
    })

    session = VGBSession(session_id, prompt, cfg)
    SESSIONS[session_id] = session

    eps_uniform, diag_uniform = await estimate_epsilon_v_uniform(
        session.value_fn, session.reward_fn, session.base_model,
        prompt, horizon, num_probes=num_probes,
        num_rollouts_per_probe=num_rollouts_per_probe,
    )
    eps_avg, diag_avg = await estimate_epsilon_v_average(
        session.value_fn, session.reward_fn, session.base_model,
        prompt, horizon, num_samples=num_average_samples,
        num_rollouts_per_sample=max(1, num_rollouts_per_probe // 2),
    )

    T_uniform = compute_T_theoretical(horizon, eps_uniform, delta, mode="uniform")
    T_average = compute_T_theoretical(horizon, eps_avg, delta, mode="average")
    max_reruns = max_reruns_for_leaf(horizon, 1.0 + max(eps_uniform, 0.0))

    warnings = []
    if eps_uniform == float("inf"):
        warnings.append("Uniform ε_V is unbounded — Assumption 4.1 likely violated.")
    elif eps_uniform > 1.0:
        warnings.append(f"Uniform ε_V ≈ {eps_uniform:.2f} > 1 — value errors may be catastrophic "
                        f"(cf. Example 3.1). Theorem 4.1 still applies but T is large.")
    if eps_avg == float("inf"):
        warnings.append("Average-case ε_V is unbounded — Assumption 4.2 likely violated.")
    elif eps_avg > 1.0:
        warnings.append(f"Average-case ε_V ≈ {eps_avg:.2f} > 1 — coverage bound (Eq. 9) may be loose.")

    return json.dumps({
        "session_id": session_id,
        "epsilon_V_uniform": eps_uniform if eps_uniform != float("inf") else None,
        "epsilon_V_average": eps_avg if eps_avg != float("inf") else None,
        "T_theorem_4_1": T_uniform,
        "T_theorem_4_2": T_average,
        "max_reruns_for_leaf": max_reruns,
        "delta": delta,
        "horizon": horizon,
        "diagnostics_uniform": diag_uniform,
        "diagnostics_average": diag_avg,
        "warnings": warnings,
    }, indent=2)


@mcp.tool()
async def action_level_rs(
    prompt: str,
    horizon: int = 5,
    block_size: int = 1,
    generation_length: int = 75,
    temperature: float = 0.7,
    num_candidates: int = 8,
    num_rollouts: int = 2,
    base_model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
    use_ground_truth_leaf: bool = True,
    max_restarts: int = 10,
    value_type: str = "tilt",
    tilt_alpha: float = 0.3,
    apply_chat_template: bool = True,
) -> str:
    """
    Action-Level Rejection Sampling baseline (Section 2.1, Algorithm 7).
    """
    session_id = str(uuid.uuid4())
    cfg = _default_config()
    cfg.update({
        "horizon": horizon,
        "block_size": block_size,
        "generation_length": generation_length,
        "temperature": temperature,
        "num_candidates": num_candidates,
        "num_rollouts": num_rollouts,
        "base_model": base_model,
        "base_url": base_url,
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
        "use_ground_truth_leaf": use_ground_truth_leaf,
        "max_restarts": max_restarts,
        "value_type": value_type,
        "tilt_alpha": tilt_alpha,
        "candidate_mode": "topk" if block_size <= 1 else "sampled",
        "apply_chat_template": apply_chat_template,
    })

    session = VGBSession(session_id, prompt, cfg)
    rs = ActionLevelRS(session)
    result = await rs.generate()
    result["algorithm"] = "ActionLevelRS"
    result["session_id"] = session_id
    SESSIONS[session_id] = session
    return json.dumps(result, indent=2)


@mcp.tool()
async def outcome_level_rs(
    prompt: str,
    generation_length: int = 75,
    base_model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
    max_attempts: int = 200,
    threshold: float = 0.5,
    apply_chat_template: bool = True,
) -> str:
    """
    Outcome-Level Rejection Sampling baseline (Section 2.1, Algorithm 6).
    """
    session_id = str(uuid.uuid4())
    cfg = _default_config()
    cfg.update({
        "generation_length": generation_length,
        "base_model": base_model,
        "base_url": base_url,
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
        "horizon": 1,
        "block_size": generation_length,
        "apply_chat_template": apply_chat_template,
    })

    session = VGBSession(session_id, prompt, cfg)
    rs = OutcomeLevelRS(session)
    result = await rs.generate(max_attempts=max_attempts, threshold=threshold)
    result["algorithm"] = "OutcomeLevelRS"
    result["session_id"] = session_id
    SESSIONS[session_id] = session
    return json.dumps(result, indent=2)


@mcp.tool()
async def configure_vgb(
    base_model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    horizon: int = 5,
    block_size: int = 1,
    generation_length: int = 75,
    max_steps: int = 200,
    num_candidates: int = 8,
    num_rollouts: int = 2,
    temperature: float = 0.7,
    value_type: str = "tilt",
    tilt_alpha: float = 0.3,
    beta: float = 1.0,
    use_ground_truth_leaf: bool = True,
    candidate_mode: str = "topk",
    use_laziness: bool = False,
    use_local_model: bool = False,
    model_name: str = "",
    apply_chat_template: bool = True,
    value_model: str = "",
    value_model_local: bool = False,
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
    use_momentum: bool = False,
) -> str:
    """Set default configuration for future VGB sessions (does not affect existing ones)."""
    updates = {
        "base_model": base_model,
        "base_url": base_url,
        "horizon": horizon,
        "block_size": block_size,
        "generation_length": generation_length,
        "max_steps": max_steps,
        "num_candidates": num_candidates,
        "num_rollouts": num_rollouts,
        "temperature": temperature,
        "value_type": value_type,
        "tilt_alpha": tilt_alpha,
        "beta": beta,
        "use_ground_truth_leaf": use_ground_truth_leaf,
        "candidate_mode": candidate_mode,
        "use_laziness": use_laziness,
        "use_local_model": use_local_model,
        "model_name": model_name,
        "apply_chat_template": apply_chat_template,
        "value_model": value_model,
        "value_model_local": value_model_local,
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
        "use_momentum": use_momentum,  # FIX 1B
    }
    _set_default_config(updates)
    return json.dumps({"status": "Configured", "config": _default_config()}, indent=2)


# =====================================================================
# Utility
# =====================================================================

def _walk_tree(node: Node) -> List[Node]:
    """BFS walk of the generation tree (non-mutating copy of queue)."""
    result = []
    queue = [node]
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(list(n.children.values()))
    return result


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    mcp.run()
