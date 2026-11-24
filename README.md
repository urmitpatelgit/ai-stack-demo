# Local Agentic AI Stack: Step-by-Step Learning Project

A hands-on, 100% free and local AI engineering project demonstrating how **LLM (GenAI)**, **RAG**, **Model Context Protocol (MCP)**, **LangChain**, **LangGraph**, and **Streamlit** fit together in a real-world warehouse inventory auditing workflow.

Instead of jumping straight into complex code, this repository is organized into **three progressive learning steps**—allowing you to run each milestone and see *why* each architectural layer is added.

---

## System Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│                STEP 3: STREAMLIT WEB UI (streamlit_app.py)        │
│   • Live Streaming Output   • Human-in-the-Loop Approve / Reject  │
│   • Interactive CSV Editor  • Real-Time Audit Log Viewer          │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│           STEP 2: LANGGRAPH MULTI-AGENT ORCHESTRATOR              │
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

## Core Concepts Covered

| Concept | Role in This Project | Implementation |
| :--- | :--- | :--- |
| **LLM & GenAI** | Reasoning, tool selection, and natural-language summaries | [Ollama](https://ollama.com) (`qwen2.5:3b` running locally on CPU) |
| **RAG (Retrieval-Augmented Generation)** | Grounds the AI in private SOP rules instead of guessing | `RecursiveCharacterTextSplitter` + `nomic-embed-text` + `InMemoryVectorStore` |
| **MCP (Model Context Protocol)** | Standardized "USB-C" bridge connecting AI to external tools and data | `mcp_server.py` built with FastMCP over `stdio` |
| **LangChain** | Connects prompts, local models, vector retrievers, and MCP adapters | `create_agent` + `langchain-mcp-adapters` |
| **LangGraph** | Multi-agent state orchestration, streaming, and Human-in-the-Loop pauses | `StateGraph` + `MemorySaver` + `interrupt_before` |
| **Streamlit** | Interactive web frontend for live streaming, CSV editing, and approvals | `streamlit_app.py` |

---

## Project Structure

```text
ai-stack-demo/
├── mcp_server.py           # Shared FastMCP tool server (CSV lookup, alerts, email, audit log)
├── inventory.csv           # Sample warehouse SKU stock dataset
├── company_policies.txt    # SOP rules indexed into the local RAG vector store
│
├── agent_app.py            # [Step 1] Single-Agent CLI (LLM + RAG + MCP)
├── multi_agent_app.py      # [Step 2] Multi-Agent CLI (LangGraph Auditor -> Dispatcher)
├── streamlit_app.py        # [Step 3] Full-Stack Web UI + Streaming + Human-in-the-Loop
│
├── requirements.txt        # Python dependencies
├── README.md               # Step-by-step learning guide
└── .gitignore              # Excludes venv, cache, and runtime logs
```

---

## Quickstart Setup

### 1. Install Ollama & Pull Free Local Models
Download [Ollama](https://ollama.com) and pull the chat model and embedding model:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 2. Clone Repository & Install Python Dependencies

```bash
git clone https://github.com<your-username>/ai-stack-demo.git
cd ai-stack-demo

python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

---

## Step-by-Step Learning Path

### Step 1: Single-Agent RAG + MCP (`agent_app.py`)
**Goal:** Understand how a single LLM connects to external data via **MCP** and retrieves company rules from a local file via **RAG**.

* **What happens:** One agent receives all tools (`search_company_policies`, `get_all_inventory`, `send_team_alert`, `send_escalation_email`), reads `company_policies.txt`, scans `inventory.csv`, and executes the matching actions.
* **Run it:**
  ```bash
  python agent_app.py
  ```
* **Key Takeaway:** Works well for simple tasks, but giving a compact 3B model both read and write tools at the same time increases the risk of wrong tool calls.

---

### Step 2: Multi-Agent Orchestration with LangGraph (`multi_agent_app.py`)
**Goal:** Learn how **LangGraph** improves reliability by splitting one overloaded agent into two specialized nodes sharing a typed state (`AuditState`).

* **What happens:**
  1. **Node 1 (`Auditor Agent`):** Only gets read-only tools (`search_company_policies`, `get_all_inventory`) and writes an `audit_report` into shared state.
  2. **Node 2 (`Dispatcher Agent`):** Only gets action tools (`send_team_alert`, `send_escalation_email`) and executes notifications based on Node 1's report.
* **Run it:**
  ```bash
  python multi_agent_app.py
  ```
* **Key Takeaway:** Role separation (Read vs. Write) prevents premature actions and makes smaller local LLMs much more dependable.

---

### Step 3: Full-Stack Web UI with Streaming & Human-in-the-Loop (`streamlit_app.py`)
**Goal:** Add production safety and observability using **LangGraph checkpoints** and a **Streamlit** browser dashboard.

* **What happens:**
  1. **Live Streaming (`stream_mode="updates"`):** Node 1 streams its audit report to the UI as soon as it completes.
  2. **Human-in-the-Loop (`interrupt_before=["dispatcher"]`):** Execution pauses before Node 2 runs, saving state in `MemorySaver`.
  3. **Operator Approval:** You review the report in the browser and click **Approve & Dispatch Alerts** (or **Reject & Cancel**) before any external alert or email fires.
* **Run it:**
  ```bash
  streamlit run streamlit_app.py
  ```
  Open `http://localhost:8501` in your browser to run audits, edit `inventory.csv` live, and inspect `audit_log.txt`.

---

## Optional: Enable Real Gmail Escalation Emails

By default, `send_escalation_email` logs simulated emails to `audit_log.txt`. To send real emails via Gmail SMTP, set these environment variables before running any step:

```powershell
# Windows PowerShell
\$env:GMAIL_USER="your_email@gmail.com"
\$env:GMAIL_APP_PASSWORD="your-16-char-app-password"
\$env:ESCALATION_RECIPIENT="recipient@gmail.com"
streamlit run streamlit_app.py
```

---

## Engineering Lessons Learned

1. **Tool-Side Status Classification:** Offloading threshold math (`0` -> `OUT OF STOCK`, `< 10` -> `LOW STOCK`) to `mcp_server.py` prevents numeric reasoning mistakes in compact 3B local models.
2. **Thread-Safe MCP File I/O:** Because LLMs often trigger parallel tool calls in a single turn, `mcp_server.py` wraps file writes in `threading.Lock()` so concurrent logs never overwrite each other.
3. **Checkpoint-Driven HITL:** Using `interrupt_before` in LangGraph ensures external side effects (emails/alerts) require explicit human sign-off.
