# JetSpecMCP
MCP Implementation of JetSpec

### How to run and use it:

1. **Install dependencies:** Open your terminal and install the required Python libraries. You will need at least the MCP SDK and HTTPX (and optionally HuggingFace Transformers if you plan to use local models):
   ```bash
   pip install mcp httpx transformers torch
   ```
3. **Configure your MCP Client (e.g., Claude Desktop):** Use an MCP client like Claude Desktop/KiloCode/Codex/RooCode/Cline/OpenCode/etc, you will need to edit its settings json file to point to this script. It usually looks something like this:
   ```json
   {
     "mcpServers": {
       "VGB-Engine": {
         "command": "python",
         "args": [
           "/absolute/path/to/your/vgb_server.py"
         ],
         "env": {
           "OPENAI_API_KEY": "your-api-key-here"
         }
       }
     }
   }
   ```
4. **Run it directly (optional):** You can also test that the script runs without errors by executing it directly in your terminal:
   ```bash
   python JetSpecMCP.py
   ```
