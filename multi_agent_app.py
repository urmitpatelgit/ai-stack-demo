"""
Multi-agent inventory audit application.

This program demonstrates an agentic workflow built with LangChain and LangGraph:

1. A local Ollama chat model reads the audit task.
2. Company policy text is split into chunks, embedded with Ollama, and stored in
   an in-memory vector store for retrieval-augmented generation (RAG).
3. An MCP client starts mcp_server.py and exposes its inventory and notification
   operations as LangChain tools.
4. An Inventory Auditor agent reads the policy and inventory CSV, then creates a
   report containing each SKU, its stock status, and the required action.
5. An Operations Dispatcher agent receives that report and executes only the
   policy-approved alert or escalation tool for low-stock and out-of-stock items.
6. LangGraph passes shared state from the auditor node to the dispatcher node,
   so the two specialized agents perform the audit in a controlled sequence.

The workflow is intentionally deterministic: the auditor is restricted to policy
retrieval and inventory tools, while the dispatcher is restricted to alert tools.
RAG context and MCP tools make the agent behavior grounded in project files and
actual operations rather than generated inventory or notification claims.
"""

import asyncio
from pathlib import Path
from typing import TypedDict

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

# Resolve project resources relative to this file instead of the terminal's
# current working directory.
BASE_DIR = Path(__file__).parent
POLICY_FILE = BASE_DIR / "company_policies.txt"

# Use a local Ollama model with zero temperature so tool selection and summaries
# remain as consistent as possible across runs.
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# RAG setup:
# 1. Read the policy document.
# 2. Split it into overlapping chunks so neighboring rules retain context.
# 3. Embed each chunk locally.
# 4. Store the embeddings in memory for the lifetime of this process.
raw_text = POLICY_FILE.read_text(encoding="utf-8")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
policy_docs = [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore.from_documents(policy_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_company_policies(query: str = "inventory stock policy rules") -> str:
    """Search company_policies.txt for stock alert and escalation rules."""
    # Retrieve several relevant chunks; joining them gives the agent enough policy
    # context without loading the entire document into every prompt.
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


class AuditState(TypedDict):
    """Data shared between the LangGraph auditor and dispatcher nodes."""

    # Original instruction passed to the auditor agent.
    task: str
    # Structured audit findings produced by the auditor node.
    audit_report: str
    # Final notification summary produced by the dispatcher node.
    final_summary: str


async def main():
    # Start the existing MCP server as a child process. Standard input/output
    # carries MCP requests and responses between the server and this application.
    mcp_client = MultiServerMCPClient(
        {
            "operations": {
                "transport": "stdio",
                "command": "python",
                "args": [str(BASE_DIR / "mcp_server.py")],
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()

    # Assign tools by responsibility. This least-privilege split prevents the
    # auditor from sending notifications and the dispatcher from rereading inventory.
    auditor_tools = [search_company_policies] + [
        t for t in mcp_tools if t.name == "get_all_inventory"
    ]
    dispatcher_tools = [
        t for t in mcp_tools if t.name in ("send_team_alert", "send_escalation_email")
    ]

    # Create a read-only auditor: it must inspect policy and inventory and return
    # an action recommendation for every SKU.
    auditor_agent = create_agent(
        model=llm,
        tools=auditor_tools,
        system_prompt=(
            "You are an Inventory Auditor. Call search_company_policies and "
            "get_all_inventory. Produce a clear list of every SKU, its stock status, "
            "and the exact action required by policy (send_team_alert, "
            "send_escalation_email, or NO ACTION)."
        ),
    )

    # Create an action-oriented dispatcher: it receives the auditor's report and
    # executes only the notifications required for low and out-of-stock items.
    dispatcher_agent = create_agent(
        model=llm,
        tools=dispatcher_tools,
        system_prompt=(
            "You are an Operations Dispatcher. Read the audit report provided. "
            "Call send_team_alert ONLY for LOW STOCK items and send_escalation_email "
            "ONLY for OUT OF STOCK items. Ignore HEALTHY STOCK items, then summarize."
        ),
    )

    async def auditor_node(state: AuditState) -> dict:
        """Run the auditor agent and return the report for the next graph node."""
        print("\n--- [Node 1: Auditor Agent Running] ---")
        result = await auditor_agent.ainvoke(
            {"messages": [("user", state["task"])]}
        )
        report = result["messages"][-1].content
        print(f"[Auditor Report]\n{report}\n")
        return {"audit_report": report}

    async def dispatcher_node(state: AuditState) -> dict:
        """Run the dispatcher agent against the auditor's persisted findings."""
        print("--- [Node 2: Dispatcher Agent Running] ---")
        prompt = f"Execute the required notifications for this audit report:\n\n{state['audit_report']}"
        result = await dispatcher_agent.ainvoke(
            {"messages": [("user", prompt)]}
        )
        summary = result["messages"][-1].content
        return {"final_summary": summary}

    # Wire a strict two-stage pipeline:
    # START -> auditor -> dispatcher -> END.
    # No branches or loops are needed because both stages have a fixed order.
    workflow = StateGraph(AuditState)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("dispatcher", dispatcher_node)

    workflow.add_edge(START, "auditor")
    workflow.add_edge("auditor", "dispatcher")
    workflow.add_edge("dispatcher", END)

    # Compilation validates the graph and returns an executable application.
    app = workflow.compile()

    # Initial state fields are explicit so TypedDict state remains easy to trace.
    # LangGraph merges each node's returned dictionary into the shared state.
    final_state = await app.ainvoke(
        {
            "task": "Audit all products in inventory.csv against company policies.",
            "audit_report": "",
            "final_summary": "",
        }
    )

    print("\n=== [Final Multi-Agent Summary] ===")
    print(final_state["final_summary"])


if __name__ == "__main__":
    asyncio.run(main())
