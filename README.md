# Multi-Agent Inventory Operations Dashboard (Local AI Stack)

An end-to-end **Agentic AI** project built 100% locally and free using **Ollama**, **LangChain**, **LangGraph**, **RAG**, **Model Context Protocol (MCP)**, and **Streamlit**.

The application audits warehouse inventory (`inventory.csv`) against company Standard Operating Procedures (`company_policies.txt`), pauses for **Human-in-the-Loop (HITL)** approval in a live web dashboard, and dispatches alerts or escalation emails through a standalone **MCP server**.

---

## Architecture Overview

```text
┌───────────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB UI (streamlit_app.py)            │
│   • Live Streaming Output   • Human-in-the-Loop Approve / Reject  │
│   • Interactive CSV Editor  • Real-Time Audit Log Viewer          │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│               LANGGRAPH MULTI-AGENT ORCHESTRATOR                  │
│                                                                   │
│  [START] ──▶ [Node 1: Auditor Agent] ──▶ (HITL Checkpoint)        │
│                   │                            │                  │
│                   ▼                     [Approve / Reject]        │
│          RAG Vector Store                      │                  │
│       (company_policies.txt)                   ▼                  │
│                                     [Node 2: Dispatcher Agent]    │
│                                                │                  │
│                                                ▼                  │
│                                              [END]                │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ (MCP over stdio)
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                  MCP TOOL SERVER (mcp_server.py)                  │
│  • get_all_inventory()       ──▶ Reads inventory.csv              │
│  • send_team_alert()         ──▶ Writes thread-safe audit_log.txt │
│  • send_escalation_email()   ──▶ Gmail SMTP or audit_log.txt      │
└───────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack & Core Concepts

* **LLM & GenAI ([Ollama](https://ollama.com)):** Runs `qwen2.5:3b` locally on CPU with zero API costs.
* **RAG ([LangChain Core](https://langchain.com)):** Loads and chunks `company_policies.txt` using `RecursiveCharacterTextSplitter`, embeds chunks with `nomic-embed-text`, and retrieves rules via `InMemoryVectorStore`.
* **Connectivity ([Model Context Protocol](https://modelcontextprotocol.io)):** Decouples business tools (`mcp_server.py`) from the AI agents using FastMCP over `stdio`.
* **Multi-Agent Orchestration & HITL ([LangGraph](https://langchain.comlanggraph)):** Splits responsibilities between a read-only **Auditor Agent** and an action-taking **Dispatcher Agent**, using `MemorySaver` and `interrupt_before` to require human sign-off before notifications fire.
* **Frontend ([Streamlit](https://streamlit.io)):** Streams agent steps live, supports inline editing of `inventory.csv`, and displays thread-safe execution logs.

---

## Project Structure

```text
ai-stack-demo/
├── streamlit_app.py        # Main Streamlit dashboard + LangGraph HITL multi-agent app
├── multi_agent_app.py      # CLI version of the LangGraph multi-agent workflow
├── agent_app.py            # CLI version of the single-agent RAG + MCP prototype
├── mcp_server.py           # Standalone FastMCP server exposing inventory & alert tools
├── inventory.csv           # Sample warehouse SKU stock dataset
├── company_policies.txt    # SOP rules indexed into the local RAG vector store
├── requirements.txt        # Python dependencies
└── .gitignore              # Excludes venv, cache, and runtime logs
```

---

## Quickstart Guide

### 1. Prerequisites
Install [Ollama](https://ollama.com) and pull the two local models:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Clone & Install Python Dependencies

```bash
git clone https://github.com/urmitpatelgit/ai-stack-demo
cd ai-stack-demo

python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Streamlit Web Dashboard

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser, click **Start Inventory Audit (Node 1)**, review the streamed Auditor Report, and click **Approve & Dispatch Alerts (Node 2)** to trigger the MCP tools.

---

## Optional: Enable Real Gmail Escalation Emails

By default, `send_escalation_email` safely logs simulated emails to `audit_log.txt`. To send real emails via Gmail SMTP, set these environment variables before launching Streamlit:

```powershell
# Windows PowerShell
\$env:GMAIL_USER="your_email@gmail.com"
\$env:GMAIL_APP_PASSWORD="your-16-char-app-password"
\$env:ESCALATION_RECIPIENT="recipient@gmail.com"
streamlit run streamlit_app.py
```

---

## Key Engineering Takeaways

* **Tool-Side Status Classification:** Offloading deterministic threshold math (`0` -> `OUT OF STOCK`, `< 10` -> `LOW STOCK`) to the MCP tool eliminates numeric reasoning errors in compact 3B local LLMs.
* **Thread-Safe MCP File I/O:** Because LLMs invoke multiple MCP tools in parallel, `mcp_server.py` uses `threading.Lock()` to prevent race conditions when writing to `audit_log.txt`.
* **Role Separation via LangGraph:** Splitting read tools (Auditor) from write tools (Dispatcher) behind a human approval checkpoint prevents unintended side effects.
