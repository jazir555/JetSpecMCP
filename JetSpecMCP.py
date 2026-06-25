"""
VGB: Value-Guided Sampling with Stochastic Backtracking
Reimplementation of: "Taming Imperfect Process Verifiers:
A Sampling Perspective on Backtracking" (Rohatgi et al., 2025)

Key fixes over the original MCP server:
  1. Token-level generation (matching Algorithm 1) alongside block-level
  2. Ground-truth outcome rewards at leaf nodes (Algorithm 1, Line 1)
  3. Log-space arithmetic throughout to prevent underflow
  4. Proper duplicate-candidate probability aggregation
  5. Session-scoped config (no global mutable state shared across sessions)
  6. Both theoretical mode (laziness + fixed T) and practical mode (run-to-leaf)
  7. ActionLevelRS and OutcomeLevelRS baselines
  8. KL-regularized setting (Algorithm 2) via Q̂-based value
  9. Tilt value function for constrained sampling (Section 5.3)
  10. Fixed max_tokens=0 bug at leaf rollouts
"""

import os
import json
import uuid
import math
import random
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from mcp.server.fastmcp import FastMCP
import httpx

mcp = FastMCP("VGB-Engine")

# =====================================================================
# §1  Mathematical Utilities
# =====================================================================

def logsumexp(values: List[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        return float('-inf')
    max_val = max(values)
    if max_val == float('-inf'):
        return float('-inf')
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def sample_from_log_weights(log_weights: List[float]) -> int:
    """Sample an index proportional to exp(log_weight)."""
    if not log_weights:
        raise ValueError("Empty log_weights")
    max_lw = max(log_weights)
    weights = [math.exp(lw - max_lw) for lw in log_weights]
    total = sum(weights)
    if total <= 0:
        return random.randint(0, len(log_weights) - 1)
    return random.choices(range(len(log_weights)), weights=weights, k=1)[0]


def safe_log(x: float, eps: float = 1e-30) -> float:
    return math.log(max(x, eps))


# =====================================================================
# §2  Reward Functions  (τ : X × Y → R≥0)
# =====================================================================

class RewardFunction:
    """Base class for outcome-level reward / tilt functions τ(x, y)."""

    async def evaluate(self, prompt: str, completion: str) -> float:
        raise NotImplementedError

    def describe(self) -> str:
        return "base"


class LLMJudgeReward(RewardFunction):
    """
    Improved LLM-as-judge reward.
    Uses chain-of-thought before a forced Score line, reducing noise.
    """

    def __init__(self, model: str, base_url: str, api_key: str,
                 reward_prompt: str = ""):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0)
        self.reward_prompt = reward_prompt

    async def evaluate(self, prompt: str, completion: str) -> float:
        custom = f"\nAdditional evaluation criteria: {self.reward_prompt}\n" if self.reward_prompt else ""
        system_prompt = (
            "You are an outcome verifier.  Evaluate whether the completion "
            "correctly and completely satisfies the prompt."
            f"{custom}"
            "\nFirst, reason step-by-step about correctness.  "
            "Then on the FINAL line write EXACTLY: Score: X  "
            "where X is 1.0 if correct, 0.0 if incorrect, "
            "or a value in [0,1] for partial credit."
        )
        user_prompt = f"Prompt: {prompt}\n\nCompletion:\n{completion}\n\nEvaluate:"
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
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

        # Look for "Score: X" on the last line first, then anywhere
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
    """Binary reward: 1.0 if the full completion matches *pattern*, else 0.0."""

    def __init__(self, pattern: str):
        self.pattern = pattern

    async def evaluate(self, prompt: str, completion: str) -> float:
        return 1.0 if re.fullmatch(self.pattern, completion, re.DOTALL) else 0.0

    def describe(self) -> str:
        return f"regex({self.pattern!r})"


class ContainsReward(RewardFunction):
    """1.0 if completion contains *substring*, 0.0 otherwise."""

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
    """1.0 if completion does NOT contain *substring*, 0.0 if it does."""

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
    """Product of several reward functions (conjunction)."""

    def __init__(self, rewards: List[RewardFunction]):
        self.rewards = rewards

    async def evaluate(self, prompt: str, completion: str) -> float:
        vals = await asyncio.gather(
            *[r.evaluate(prompt, completion) for r in self.rewards]
        )
        return math.prod(vals)

    def describe(self) -> str:
        return " ∧ ".join(r.describe() for r in self.rewards)


# =====================================================================
# §3  Base Models  (sample from π_ref, compute densities)
# =====================================================================

class APIBaseModel:
    """
    OpenAI-compatible API base model.

    Supports two candidate-generation modes:
      - "topk"  (token-level): use top_logprobs for exact π_ref weights
      - "sampled" (block-level): sample K blocks, estimate π_ref from logprobs
    """

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str = ""):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=180.0)

    # ----- candidate generation -----

    async def get_candidates(
        self,
        prompt: str,
        context: str,
        num_candidates: int,
        temperature: float,
        block_size: int = 1,
        mode: str = "topk",
    ) -> List[Tuple[str, float]]:
        """
        Return [(action_text, log_pi_ref), ...] for candidate next-actions.

        mode="topk"  → token-level: top-K next tokens with exact log-probs
        mode="sampled" → block-level: K sampled blocks with summed log-probs
        """
        if block_size <= 1 and mode == "topk":
            return await self._get_topk_tokens(
                prompt, context, num_candidates
            )
        else:
            return await self._get_sampled_blocks(
                prompt, context, num_candidates, temperature, max(block_size, 1)
            )

    async def _get_topk_tokens(
        self, prompt: str, context: str, K: int
    ) -> List[Tuple[str, float]]:
        """Get top-K next tokens with their log-probs (token-level, exact π_ref)."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user",
                                  "content": f"{prompt}\n{context}"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": K,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return [("", -100.0)]

        content = data["choices"][0].get("logprobs", {}).get("content", [])
        if not content:
            # Fallback: sample a single token and use its logprob
            text = data["choices"][0]["message"].get("content", "")
            return [(text, -1.0)] if text else []

        top_lp = content[0].get("top_logprobs", [])
        candidates = []
        seen: Dict[str, float] = {}            # aggregate duplicates
        for item in top_lp:
            tok = item.get("token", "")
            lp = item.get("logprob", -100.0)
            if tok in seen:
                # log(a+b) from log(a), log(b)
                seen[tok] = logsumexp([seen[tok], lp])
            else:
                seen[tok] = lp
        candidates = list(seen.items())
        return candidates if candidates else [("", -100.0)]

    async def _get_sampled_blocks(
        self, prompt: str, context: str, K: int,
        temperature: float, block_size: int,
    ) -> List[Tuple[str, float]]:
        """Sample K multi-token blocks and return with summed log-probs."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user",
                                  "content": f"{prompt}\n{context}"}],
                    "max_tokens": block_size,
                    "temperature": temperature,
                    "n": K,
                    "logprobs": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return [("", -100.0)]

        candidates: Dict[str, float] = {}      # aggregate duplicates in log-space
        for choice in data["choices"]:
            block_text = choice["message"].get("content", "")
            if not block_text:
                continue
            lp_content = choice.get("logprobs", {}).get("content", [])
            sum_lp = sum(
                it.get("logprob", -10.0) for it in lp_content if it is not None
            )
            if block_text in candidates:
                candidates[block_text] = logsumexp([candidates[block_text], sum_lp])
            else:
                candidates[block_text] = sum_lp
        # Preserve duplicates as separate entries (critical for correct π_ref mass)
        result: List[Tuple[str, float]] = []
        for choice in data["choices"]:
            block_text = choice["message"].get("content", "")
            if not block_text:
                continue
            lp_content = choice.get("logprobs", {}).get("content", [])
            sum_lp = sum(
                it.get("logprob", -10.0) for it in lp_content if it is not None
            )
            result.append((block_text, sum_lp))
        return result if result else [("", -100.0)]

    # ----- completion for MC rollouts -----

    async def complete(self, prompt: str, context: str,
                       max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user",
                                  "content": f"{prompt}\n{context}"}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                },
            )
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return ""


class LocalBaseModel:
    """HuggingFace local model (offline generation with exact log-probs)."""

    def __init__(self, model_name: str):
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

    async def get_candidates(
        self, prompt: str, context: str, num_candidates: int,
        temperature: float, block_size: int = 1, mode: str = "topk",
    ) -> List[Tuple[str, float]]:
        full_text = prompt + context
        inputs = self.tokenizer(full_text, return_tensors="pt").to(
            self.model.device
        )
        input_len = inputs["input_ids"].shape[1]

        if block_size <= 1 and mode == "topk":
            # Token-level: get top-K next-token log-probs
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
            # Block-level: sample K sequences
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
                # Trim at EOS / PAD
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

    async def complete(self, prompt: str, context: str,
                       max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        full_text = prompt + context
        inputs = self.tokenizer(full_text, return_tensors="pt").to(
            self.model.device
        )
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
# §4  Value Functions  V̂(x, y_{1:h})
# =====================================================================

class ValueFunction:
    """Base class for approximate value functions."""

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        raise NotImplementedError

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        """At leaf nodes: override to use ground-trrew τ when available."""
        return await self.evaluate(prompt, completion, depth=-1)


class MCRolloutValue(ValueFunction):
    """
    V̂ via Monte-Carlo rollouts:  V̂(x, y_{1:h}) ≈ E[τ(x, y_{1:H}) | y_{1:h}].

    FIX: remaining_tokens is always > 0 for non-leaf nodes.
         At leaf nodes evaluate_leaf is called instead (ground-truth τ).
    """

    def __init__(self, base_model, reward_fn: RewardFunction,
                 num_rollouts: int, generation_length: int):
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.num_rollouts = max(1, num_rollouts)
        self.generation_length = generation_length

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        # Estimate remaining tokens from partial length (rough but safe)
        partial_len = len(partial.split()) if partial else 0
        remaining = max(1, self.generation_length - partial_len)
        tasks = [
            self.base_model.complete(prompt, partial, remaining)
            for _ in range(self.num_rollouts)
        ]
        completions = await asyncio.gather(*tasks)
        rewards = await asyncio.gather(
            *[self.reward_fn.evaluate(prompt, c) for c in completions]
        )
        return sum(rewards) / len(rewards) if rewards else 0.0

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        """Use ground-truth τ at leaves (Algorithm 1, Line 1)."""
        return await self.reward_fn.evaluate(prompt, completion)


class TiltValue(ValueFunction):
    """
    V̂_α(x, y_{1:h}) = α^{H-h} · r*(x, y_{1:h})

    From Section 5.3 (constrained sampling without a trained value function).
    α controls the backtracking probability.
    At leaves, V̂ = r*(x, y_{1:H}) (ground truth).
    """

    def __init__(self, reward_fn: RewardFunction, alpha: float,
                 horizon: int):
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
    """
    V̂(x, y_{1:h}) = exp(β^{-1} Q̂(x, y_{1:h}))
    for the KL-regularized setting (Algorithm 2).

    Q̂ is an approximate regularized value function.
    """

    def __init__(self, q_fn: 'QFunction', beta: float):
        self.q_fn = q_fn
        self.beta = beta

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        q = await self.q_fn.evaluate(prompt, partial, depth)
        return math.exp(q / self.beta)

    async def evaluate_leaf(self, prompt: str, completion: str) -> float:
        q = await self.q_fn.evaluate(prompt, completion, depth=-1)
        return math.exp(q / self.beta)


class QFunction:
    """Placeholder for approximate Q̂ (KL-regularized setting)."""

    async def evaluate(self, prompt: str, partial: str, depth: int) -> float:
        raise NotImplementedError


class LLMJudgeQFunction(QFunction):
    """Use LLM judge as a rough Q̂."""

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
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",
                         "content": f"Prompt: {prompt}\nPartial:\n{partial}\nScore:"},
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
                 "value_score", "log_ref_prob")

    def __init__(self, text: str, parent: Optional['Node'] = None,
                 depth: int = 0):
        self.text = text
        self.parent = parent
        self.children: Dict[str, 'Node'] = {}
        self.depth = depth
        self.value_score: Optional[float] = None
        self.log_ref_prob: float = 0.0


class VGBSession:
    """Holds all state for one VGB generation session."""

    def __init__(self, session_id: str, prompt: str, config: Dict[str, Any]):
        self.session_id = session_id
        self.prompt = prompt
        self.config = config          # session-scoped, not global
        self.root = Node(text="", depth=0)
        self.current_node = self.root
        self.step_count = 0
        self.history: List[str] = []

        horizon = config["horizon"]
        gen_len = config.get("generation_length", horizon * config.get("block_size", 1))

        # --- reward function ---
        self.reward_fn = self._build_reward_fn(config)

        # --- base model ---
        if config.get("use_local_model", False):
            self.base_model = LocalBaseModel(config["model_name"])
        else:
            self.base_model = APIBaseModel(
                config["base_model"],
                config.get("base_url", "https://api.openai.com/v1"),
            )

        # --- value function ---
        value_type = config.get("value_type", "mc_rollout")
        if value_type == "mc_rollout":
            self.value_fn: ValueFunction = MCRolloutValue(
                self.base_model, self.reward_fn,
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
                config["base_model"],
                config.get("base_url", "https://api.openai.com/v1"),
                os.getenv("OPENAI_API_KEY", ""),
            )
            self.value_fn = KLRegularizedValue(
                q_fn, config.get("beta", 1.0)
            )
        else:
            self.value_fn = MCRolloutValue(
                self.base_model, self.reward_fn,
                config.get("num_rollouts", 2), gen_len,
            )

    @staticmethod
    def _build_reward_fn(config: Dict[str, Any]) -> RewardFunction:
        rt = config.get("reward_type", "llm_judge")
        if rt == "llm_judge":
            return LLMJudgeReward(
                config["base_model"],
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
# §6  VGB Engine
# =====================================================================

class VGBEngine:
    """
    Implements the VGB random walk (Algorithms 1 & 2).

    Transition weights at node y_{1:h}:
      backtrack:  ∝ V̂(x, y_{1:h})
      forward_i:  ∝ π_ref(y_{h+1} | x, y_{1:h}) · V̂(x, y_{1:h+1})

    Practical large-|A| approximation (Appendix E.1):
      backtrack:  ∝ K · V̂(current)       [sampled mode]
      forward_i:  ∝ V̂(child_i)           [sampled mode]
      (topk mode uses full π_ref · V̂ weighting)
    """

    def __init__(self, session: VGBSession):
        self.session = session

    async def _get_node_value(self, node: Node) -> float:
        """Evaluate V̂ for *node*.  Uses ground-truth τ at leaves."""
        if node.value_score is not None:
            return node.value_score
        if node.depth == 0:
            node.value_score = 0.5
            return 0.5
        cfg = self.session.config
        is_leaf = node.depth >= cfg["horizon"]
        if is_leaf and cfg.get("use_ground_truth_leaf", True):
            score = await self.session.value_fn.evaluate_leaf(
                self.session.prompt, node.text
            )
        else:
            score = await self.session.value_fn.evaluate(
                self.session.prompt, node.text, node.depth
            )
        node.value_score = max(score, 0.0)
        return node.value_score

    async def step(self) -> str:
        """Execute one step of the VGB random walk."""
        s = self.session
        cfg = s.config
        current = s.current_node
        h = current.depth
        H = cfg["horizon"]

        # --- Laziness (Algorithm 1, Line 5) ---
        if cfg.get("use_laziness", False):
            if random.random() < 0.5:
                s.step_count += 1
                s.history.append(f"Step {s.step_count}: lazy stay at depth {h}")
                return "stay"

        log_weights: List[float] = []
        neighbor_nodes: List[Node] = []
        action_types: List[str] = []

        # --- Backtracking (Algorithm 1, Line 4) ---
        if h > 0 and current.parent is not None:
            val = await self._get_node_value(current)
            lw = safe_log(val)
            # In sampled mode, multiply backtrack weight by K
            # (Appendix E.1: p̂(z[0]) ∝ K · V̂(current))
            if cfg.get("candidate_mode", "topk") == "sampled":
                lw += safe_log(cfg["num_candidates"])
            log_weights.append(lw)
            neighbor_nodes.append(current.parent)
            action_types.append("backtrack")

        # --- Forward transitions ---
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
                else:
                    # Keep the existing child (its value is already cached)
                    pass

                child_node = current.children[child_key]
                child_val = await self._get_node_value(child_node)
                if child_val <= 0:
                    continue        # skip zero-value children

                if candidate_mode == "sampled":
                    # Weight = V̂(child)  (π_ref already accounted for by sampling)
                    fw = safe_log(child_val)
                else:
                    # Weight = π_ref(action) · V̂(child)
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
        """
        Run VGB until a leaf is reached or max_steps exceeded.

        Returns dict with generation, metadata.
        """
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        max_steps = cfg.get("max_steps", max(200, H * 40))

        # Practical mode: run until leaf (Algorithm 3 / Appendix E.1)
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

    async def run_theoretical(self, T: int) -> Dict[str, Any]:
        """
        Theoretical mode: run exactly T steps with laziness,
        then check if the output is a leaf (Algorithm 1).

        Per Theorem 4.1, if the output is not a leaf, re-run
        (up to O(H) times) until a leaf is obtained.
        """
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        cfg["use_laziness"] = True

        max_attempts = max(3, H)
        for attempt in range(max_attempts):
            # Reset to root
            s.current_node = s.root
            for _ in range(T):
                await self.step()
            if s.current_node.depth >= H:
                break
            # Reset cached values for fresh walk
            s.root.value_score = None
            for child in s.root.children.values():
                child.value_score = None

        reached_leaf = s.current_node.depth >= H
        return {
            "session_id": s.session_id,
            "generation": s.current_node.text,
            "total_steps": s.step_count,
            "final_depth": s.current_node.depth,
            "reached_leaf": reached_leaf,
            "attempts": attempt + 1,
        }


# =====================================================================
# §7  Baseline Algorithms
# =====================================================================

class ActionLevelRS:
    """
    Action-level Rejection Sampling with V̂  (Section 2.1 / Algorithm 7).

    At each step h, sample y_h from:
        μ_h(y_h) ∝ π_ref(y_h | x, y_{1:h-1}) · V̂(x, y_{1:h})

    No backtracking.  If all candidates have value 0, restart from root.
    """

    def __init__(self, session: VGBSession):
        self.session = session

    async def generate(self) -> Dict[str, Any]:
        s = self.session
        cfg = s.config
        H = cfg["horizon"]
        max_restarts = cfg.get("max_restarts", 10)
        text = ""
        total_steps = 0

        for restart in range(max_restarts):
            text = ""
            success = True
            for h in range(H):
                candidates = await s.base_model.get_candidates(
                    prompt=s.prompt,
                    context=text,
                    num_candidates=cfg["num_candidates"],
                    temperature=cfg["temperature"],
                    block_size=cfg.get("block_size", 1),
                    mode=cfg.get("candidate_mode", "topk"),
                )
                # Weight by V̂
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
            "final_depth": H if success else h,
            "reached_leaf": success,
        }


class OutcomeLevelRS:
    """
    Outcome-level Rejection Sampling (Section 2.1 / Algorithm 6).

    Repeatedly draw y ~ π_ref(·|x) until τ(x, y) > threshold.
    """

    def __init__(self, session: VGBSession):
        self.session = session

    async def generate(self, max_attempts: int = 200,
                       threshold: float = 0.5) -> Dict[str, Any]:
        s = self.session
        gen_len = s.config.get("generation_length",
                               s.config["horizon"] * s.config.get("block_size", 1))
        for attempt in range(max_attempts):
            completion = await s.base_model.complete(
                s.prompt, "", max(gen_len, 1)
            )
            reward = await s.reward_fn.evaluate(s.prompt, completion)
            if reward >= threshold:
                return {
                    "generation": completion,
                    "attempts": attempt + 1,
                    "reward": reward,
                    "reached_leaf": True,
                }
        return {
            "generation": completion if 'completion' in dir() else "",
            "attempts": max_attempts,
            "reward": 0.0,
            "reached_leaf": False,
        }


# =====================================================================
# §8  Server State
# =====================================================================

SESSIONS: Dict[str, VGBSession] = {}


def _default_config() -> Dict[str, Any]:
    return {
        # Algorithm parameters
        "horizon": 5,                   # H: number of actions (tree depth)
        "block_size": 1,                # tokens per action (1 = token-level)
        "generation_length": 75,        # approximate total tokens
        "temperature": 0.7,
        "num_candidates": 8,            # K: candidates per forward step
        "max_steps": 200,               # max VGB steps before giving up

        # Value function
        "value_type": "mc_rollout",     # "mc_rollout" | "tilt" | "kl_regularized"
        "num_rollouts": 2,
        "tilt_alpha": 0.3,              # for tilt value function
        "beta": 1.0,                    # for KL-regularized
        "use_ground_truth_leaf": True,  # V̂(leaf) = τ  (Algorithm 1, Line 1)

        # Candidate generation mode
        "candidate_mode": "topk",       # "topk" (token-level) | "sampled" (block)
        "use_laziness": False,          # practical mode = False; theoretical = True

        # Model
        "base_model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "use_local_model": False,
        "model_name": "",

        # Reward
        "reward_type": "llm_judge",     # "llm_judge" | "regex" | "contains" | "not_contains" | "composite"
        "reward_prompt": "",
        "reward_regex": ".*",
        "reward_substring": "",
    }


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
    value_type: str = "mc_rollout",
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
) -> str:
    """
    Generate text using VGB (Value-Guided Sampling with Stochastic Backtracking).

    Implements Algorithm 1 from "Taming Imperfect Process Verifiers"
    (Rohatgi et al., 2025).

    Args:
        prompt:            The input prompt / question.
        horizon:           H — number of actions (tree depth).
        block_size:        Tokens per action.  1 = token-level (paper default).
        generation_length: Approx total tokens to generate.
        temperature:       Sampling temperature for π_ref.
        num_candidates:    K — candidates per forward step.
        num_rollouts:      MC rollouts per value estimate.
        value_type:        "mc_rollout" | "tilt" | "kl_regularized".
        tilt_alpha:        α for tilt value fn (Section 5.3).
        beta:              Temperature for KL-regularized setting.
        use_ground_truth_leaf: Use τ at leaves (Algorithm 1 Line 1).
        candidate_mode:    "topk" (token-level, exact π_ref) |
                           "sampled" (block-level, estimated π_ref).
        base_model:        Model name for API calls.
        base_url:          API base URL.
        reward_type:       "llm_judge" | "regex" | "contains" | "not_contains".
        reward_prompt:     Extra instructions for the LLM judge.
        reward_regex:      Regex pattern for regex reward.
        reward_substring:  Substring for contains/not_contains reward.
        run_mode:          "practical" (run-to-leaf) | "theoretical" (fixed T).
        step_count_T:      If > 0 and run_mode="theoretical", run exactly T steps.
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
    })

    session = VGBSession(session_id, prompt, cfg)
    engine = VGBEngine(session)

    if run_mode == "theoretical" and step_count_T > 0:
        result = await engine.run_theoretical(step_count_T)
    else:
        result = await engine.run()

    SESSIONS[session_id] = session
    return json.dumps(result, indent=2)


@mcp.tool()
async def vgb_step(session_id: str) -> str:
    """Execute a single step of the VGB random walk for an active session."""
    if session_id not in SESSIONS:
        return json.dumps({"error": "Session not found."})
    session = SESSIONS[session_id]
    engine = VGBEngine(session)
    action = await engine.step()
    node = session.current_node
    v = node.value_score
    return json.dumps({
        "session_id": session_id,
        "action_taken": action,
        "current_depth": node.depth,
        "current_text": node.text[-200:],      # last 200 chars
        "value": round(v, 4) if v is not None else None,
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
        "num_children_explored": sum(
            len(n.children) for n in _walk_tree(s.root)
        ),
        "reward_type": s.config.get("reward_type"),
        "value_type": s.config.get("value_type"),
        "block_size": s.config.get("block_size"),
        "candidate_mode": s.config.get("candidate_mode"),
        "use_ground_truth_leaf": s.config.get("use_ground_truth_leaf"),
        "recent_history": s.history[-5:],
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
) -> str:
    """
    Action-Level Rejection Sampling baseline (Section 2.1, Algorithm 7).

    Autoregressively samples y_h ∝ π_ref(y_h) · V̂(y_{1:h}).
    No backtracking.  Restarts from root if stuck.
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
        "value_type": "mc_rollout",
        "candidate_mode": "topk" if block_size <= 1 else "sampled",
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
) -> str:
    """
    Outcome-Level Rejection Sampling baseline (Section 2.1, Algorithm 6).

    Repeatedly sample full responses from π_ref, accept if τ(x,y) ≥ threshold.
    Simple but potentially very slow (exponential in H).
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
    value_type: str = "mc_rollout",
    tilt_alpha: float = 0.3,
    beta: float = 1.0,
    use_ground_truth_leaf: bool = True,
    candidate_mode: str = "topk",
    use_laziness: bool = False,
    use_local_model: bool = False,
    model_name: str = "",
    reward_type: str = "llm_judge",
    reward_prompt: str = "",
    reward_regex: str = ".*",
    reward_substring: str = "",
) -> str:
    """
    Set default configuration for future VGB sessions.

    This does NOT affect already-created sessions (each session
    has its own config snapshot).
    """
    defaults = _default_config()
    defaults.update({
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
        "reward_type": reward_type,
        "reward_prompt": reward_prompt,
        "reward_regex": reward_regex,
        "reward_substring": reward_substring,
    })
    # Update the module-level default (used only for new sessions)
    _DEFAULT_CONFIG_REF[0] = defaults
    return json.dumps({"status": "Configured", "config": defaults}, indent=2)


# Mutable default config reference (so configure_vgb can update it)
_DEFAULT_CONFIG_REF = [_default_config()]


# Override _default_config to read from the mutable ref
def _default_config() -> Dict[str, Any]:
    if _DEFAULT_CONFIG_REF:
        return dict(_DEFAULT_CONFIG_REF[0])
    return {
        "horizon": 5, "block_size": 1, "generation_length": 75,
        "temperature": 0.7, "num_candidates": 8, "max_steps": 200,
        "value_type": "mc_rollout", "num_rollouts": 2,
        "tilt_alpha": 0.3, "beta": 1.0,
        "use_ground_truth_leaf": True,
        "candidate_mode": "topk", "use_laziness": False,
        "base_model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "use_local_model": False, "model_name": "",
        "reward_type": "llm_judge", "reward_prompt": "",
        "reward_regex": ".*", "reward_substring": "",
    }


# =====================================================================
# Utility
# =====================================================================

def _walk_tree(node: Node) -> List[Node]:
    """BFS walk of the generation tree."""
    result = []
    queue = [node]
    while queue:
        n = queue.pop(0)
        result.append(n)
        queue.extend(n.children.values())
    return result


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    mcp.run()
