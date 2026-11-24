import asyncio
from pathlib import Path
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# Keep file-based resources relative to this script for consistent CLI startup.
BASE_DIR = Path(__file__).parent
POLICY_FILE = BASE_DIR / "company_policies.txt"

# Use a local, deterministic Ollama model to decide which tools to call.
llm = ChatOllama(model="qwen2.5:3b", temperature=0)

# Load and split the policy document into overlapping chunks for retrieval.
raw_text = POLICY_FILE.read_text(encoding="utf-8")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
policy_docs = [Document(page_content=chunk) for chunk in splitter.split_text(raw_text)]

# Embed policy chunks locally and retain them in memory for this application run.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = InMemoryVectorStore.from_documents(policy_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


@tool
def search_company_policies(query: str = "inventory stock policy rules") -> str:
    """Search company_policies.txt for stock alert and escalation rules."""
    # Return only the most relevant retrieved chunks, preserving their source text.
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


async def main():
    # Start the operations MCP server as a child process and exchange stdio messages.
    mcp_client = MultiServerMCPClient(
        {
            "operations": {
                "transport": "stdio",
                "command": "python",
                "args": [str(BASE_DIR / "mcp_server.py")],
            }
        }
    )

    # Combine local RAG search with tools dynamically registered by the MCP server.
    mcp_tools = await mcp_client.get_tools()
    all_tools = [search_company_policies] + mcp_tools

    # The system prompt defines the audit sequence and limits actions per SKU.
    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=(
            "You are an autonomous inventory auditor. "
            "Step 1: Call search_company_policies ONCE to read the company rules. "
            "Step 2: Call get_all_inventory ONCE to inspect every SKU in inventory.csv. "
            "Step 3: For each SKU, check its Status (HEALTHY STOCK, LOW STOCK, or OUT OF STOCK) "
            "and execute ONLY the single matching action required by the policy."
        ),
    )

    user_prompt = "Run a full inventory audit across all products in inventory.csv and summarize every SKU."

    print("Starting full-CSV Agentic AI audit...\n")
    # Stream intermediate tool calls and results so the audit remains observable.
    async for step in agent.astream(
        {"messages": [("user", user_prompt)]},
        stream_mode="updates",
    ):
        for node_name, output in step.items():
            msg = output["messages"][-1]
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"[Agent Action] Calling '{call['name']}' with {call['args']}")
            elif node_name == "tools":
                print(f"[Tool Result]\n{msg.content}\n")
            else:
                print(f"\n[Final Answer]\n{msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
