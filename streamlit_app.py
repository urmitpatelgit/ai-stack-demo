import asyncio
import sys
import uuid
from pathlib import Path
from typing import TypedDict

import pandas as pd
import streamlit as st
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "inventory.csv"
POLICY_FILE = BASE_DIR / "company_policies.txt"
LOG_PATH = BASE_DIR / "audit_log.txt"

st.set_page_config(page_title="Multi-Agent Inventory Auditor (HITL)", layout="wide")


class AuditState(TypedDict):
    task: str
    audit_report: str
    final_summary: str


@st.cache_resource
def get_checkpointer() -> MemorySaver:
    """Persist LangGraph state across Streamlit button clicks."""
    return MemorySaver()


@st.cache_resource
def build_rag_retriever():
    """Build and cache the local Ollama + InMemoryVectorStore RAG retriever."""
    raw_text = POLICY_FILE.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    policy_docs = [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = InMemoryVectorStore.from_documents(policy_docs, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})


retriever = build_rag_retriever()


@tool
def search_company_policies(query: str = "inventory stock policy rules") -> str:
    """Search company_policies.txt for stock alert and escalation rules."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


async def build_compiled_graph():
    """Connect to mcp_server.py and compile the HITL LangGraph workflow."""
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)

    mcp_client = MultiServerMCPClient(
        {
            "operations": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "mcp_server.py")],
            }
        }
    )
    mcp_tools = await mcp_client.get_tools()

    auditor_tools = [search_company_policies] + [
        t for t in mcp_tools if t.name == "get_all_inventory"
    ]
    dispatcher_tools = [
        t for t in mcp_tools if t.name in ("send_team_alert", "send_escalation_email")
    ]

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
        result = await auditor_agent.ainvoke({"messages": [("user", state["task"])]})
        return {"audit_report": result["messages"][-1].content}

    async def dispatcher_node(state: AuditState) -> dict:
        prompt = f"Execute the required notifications for this audit report:\n\n{state['audit_report']}"
        result = await dispatcher_agent.ainvoke({"messages": [("user", prompt)]})
        return {"final_summary": result["messages"][-1].content}

    workflow = StateGraph(AuditState)
    workflow.add_node("auditor", auditor_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_edge(START, "auditor")
    workflow.add_edge("auditor", "dispatcher")
    workflow.add_edge("dispatcher", END)

    # Pause right before 'dispatcher' so a human can approve or reject actions
    return workflow.compile(
        checkpointer=get_checkpointer(),
        interrupt_before=["dispatcher"],
    )


async def stream_until_approval(user_task: str, thread_id: str, node1_placeholder) -> str:
    """Stream Node 1 live into the UI, then pause before Node 2."""
    app = await build_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    audit_report = ""

    async for event in app.astream(
        {"task": user_task, "audit_report": "", "final_summary": ""},
        config=config,
        stream_mode="updates",
    ):
        if "auditor" in event:
            audit_report = event["auditor"]["audit_report"]
            with node1_placeholder.container():
                st.success("Node 1 Complete: Auditor Agent Report (Streamed Live)")
                st.markdown(audit_report)

    return audit_report


async def resume_after_approval(thread_id: str, node2_placeholder) -> str:
    """Resume the paused LangGraph thread and stream Node 2 live."""
    app = await build_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    final_summary = ""

    # Passing None resumes from the saved checkpoint before 'dispatcher'
    async for event in app.astream(None, config=config, stream_mode="updates"):
        if "dispatcher" in event:
            final_summary = event["dispatcher"]["final_summary"]
            with node2_placeholder.container():
                st.success("Node 2 Complete: Dispatcher Agent Summary (Streamed Live)")
                st.markdown(final_summary)

    return final_summary


# ---------------- UI LAYOUT ----------------
st.title("Multi-Agent Inventory Operations Dashboard")
st.caption("Local Stack: Ollama (qwen2.5:3b) • RAG • MCP Server • LangGraph HITL Streaming")

if "workflow_status" not in st.session_state:
    st.session_state["workflow_status"] = "idle"  # idle | awaiting_approval | approved | rejected

left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.subheader("1. Multi-Agent Workflow (Human-in-the-Loop)")
    task_input = st.text_input(
        "Audit Instruction",
        value="Audit all products in inventory.csv against company policies.",
    )

    if st.button("Start Inventory Audit (Node 1)", type="primary", use_container_width=True):
        new_thread = str(uuid.uuid4())
        st.session_state["thread_id"] = new_thread
        st.session_state["audit_report"] = ""
        st.session_state["final_summary"] = ""
        st.session_state["workflow_status"] = "running_node1"

        node1_slot = st.empty()
        with node1_slot.container():
            st.info("Streaming Node 1: Auditor Agent is reading policies & inventory.csv...")

        report = asyncio.run(stream_until_approval(task_input, new_thread, node1_slot))
        st.session_state["audit_report"] = report
        st.session_state["workflow_status"] = "awaiting_approval"
        st.rerun()

    # Render Node 1 result persistently once completed
    if st.session_state.get("audit_report"):
        with st.expander("Node 1: Auditor Agent Report (RAG + MCP Read)", expanded=True):
            st.markdown(st.session_state["audit_report"])

    # Human-in-the-Loop Approval Gate before Node 2
    if st.session_state["workflow_status"] == "awaiting_approval":
        st.warning("Human Approval Required: Review the Auditor Report above before dispatching alerts/emails.")
        btn_approve, btn_reject = st.columns(2)

        with btn_approve:
            if st.button("Approve & Dispatch Alerts (Node 2)", type="primary", use_container_width=True):
                node2_slot = st.empty()
                with node2_slot.container():
                    st.info("Streaming Node 2: Dispatcher Agent is executing approved MCP tools...")

                summary = asyncio.run(
                    resume_after_approval(st.session_state["thread_id"], node2_slot)
                )
                st.session_state["final_summary"] = summary
                st.session_state["workflow_status"] = "approved"
                st.rerun()

        with btn_reject:
            if st.button("Reject & Cancel Dispatch", use_container_width=True):
                st.session_state["workflow_status"] = "rejected"
                st.rerun()

    if st.session_state["workflow_status"] == "rejected":
        st.error("Dispatch rejected by operator. No alerts or emails were sent.")

    if st.session_state.get("final_summary"):
        with st.expander("Node 2: Dispatcher Agent Summary (MCP Actions)", expanded=True):
            st.markdown(st.session_state["final_summary"])

with right_col:
    st.subheader("2. Live Data & Audit Trail")

    tab_inv, tab_pol, tab_log = st.tabs(["inventory.csv", "company_policies.txt", "audit_log.txt"])

    with tab_inv:
        df = pd.read_csv(CSV_PATH)
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        if st.button("Save inventory.csv Changes"):
            edited_df.to_csv(CSV_PATH, index=False)
            st.success("Saved changes to inventory.csv!")

    with tab_pol:
        st.text_area(
            "Loaded into RAG Vector Store",
            value=POLICY_FILE.read_text(encoding="utf-8"),
            height=220,
            disabled=True,
        )

    with tab_log:
        if LOG_PATH.exists() and LOG_PATH.read_text(encoding="utf-8").strip():
            st.code(LOG_PATH.read_text(encoding="utf-8"), language="text")
            if st.button("Clear audit_log.txt"):
                LOG_PATH.unlink()
                st.rerun()
        else:
            st.info("No actions logged yet. Approve an audit to generate entries.")
