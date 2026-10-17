from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from langchain_core.tools import StructuredTool, BaseTool, tool
from pydantic import create_model, Field
from typing import Any, Optional
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from fastmcp import Client
from dotenv import load_dotenv
import aiosqlite
import requests
from langchain_groq import ChatGroq
import asyncio
import threading
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict,Dict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langchain_groq import ChatGroq
import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
os.environ["LANGCHAIN_PROJECT"] = "MultiCapability-langgraph-chatbot"
os.environ["LANGCHAIN_TRACING_V2"] = "true"


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm =  ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0
)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embedding_model)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

from langchain_core.runnables import RunnableConfig

# @tool
# def rag_tool(query: str, config: RunnableConfig) -> dict:
#     """
#     Retrieve relevant information from the uploaded PDF for this chat.
#     Args:
#         query: A natural-language search string.
#     """
#     thread_id = config.get("configurable", {}).get("thread_id")
#     retriever = _get_retriever(thread_id)
#     if retriever is None:
#         return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

#     result = retriever.invoke(query)
#     return {
#         "query": query,
#         "context": [d.page_content for d in result],
#         "metadata": [d.metadata for d in result],
#         "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
#     }

@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """Retrieve relevant information from the uploaded PDF for this chat."""

    thread_id = config.get("configurable", {}).get("thread_id")

    if not thread_id:
        return {
            "error": "Missing thread_id in RunnableConfig.",
            "query": query,
        }

    try:
        retriever = _get_retriever(thread_id)

        if retriever is None:
            return {
                "error": "No document indexed for this chat. Upload a PDF first.",
                "query": query,
            }

        result = retriever.invoke(query)

        if not result:
            return {
                "query": query,
                "context": [],
                "metadata": [],
                "source_file": _THREAD_METADATA.get(
                    str(thread_id), {}
                ).get("filename"),
                "message": "No relevant content found in the uploaded PDF.",
            }

        return {
            "query": query,
            "context": [doc.page_content for doc in result],
            "metadata": [doc.metadata for doc in result],
            "source_file": _THREAD_METADATA.get(
                str(thread_id), {}
            ).get("filename"),
        }

    except Exception as exc:
        # Log the full traceback in the backend terminal.
        import traceback
        traceback.print_exc()

        return {
            "error": f"PDF retrieval failed: {type(exc).__name__}: {exc}",
            "query": query,
        }

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=J6ENQCVCAKFA0X0M"
    r = requests.get(url)
    return r.json()


# client= MultiServerMCPClient(
#     {
#         "math": {
#         "transport": "stdio",
#         "command": "D:\AI_ML\Local_Maths_MCP_Server\.venv\Scripts\python.exe",
#         "args": ["D:\AI_ML\Local_Maths_MCP_Server\main.py"],
#     },
#         "ExpenseTracker": {
#             "transport": "streamable-http",
#             "url": "https://suitable-chocolate-caterpillar.fastmcp.app/mcp"
            
#         }
#     }
# )

# -------------------
# MCP Clients
# -------------------

MATH_CLIENT = Client(
    StdioTransport(
        command="D:\AI_ML\Local_Maths_MCP_Server\.venv\Scripts\python.exe",
        args=[
            "D:\AI_ML\Local_Maths_MCP_Server\main.py"
        ],
    )
)

EXPENSE_CLIENT = Client(
    "https://suitable-chocolate-caterpillar.fastmcp.app/mcp",
    auth="oauth",
)


def _python_type(prop: dict) -> type:
    """Convert basic JSON Schema types to Python types."""
    schema_type = prop.get("type", "string")

    if isinstance(schema_type, list):
        schema_type = next(
            (t for t in schema_type if t != "null"),
            "string",
        )

    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict[str, Any],
        "array": list[Any],
    }.get(schema_type, Any)


def _make_langchain_tool(
    mcp_client: Client,
    spec: Any,
    prefix: str,
) -> StructuredTool:
    """Adapt an MCP tool to a LangChain StructuredTool."""
    schema = getattr(spec, "inputSchema", {}) or {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}

    for field_name, prop in properties.items():
        annotation = _python_type(prop)
        description = prop.get("description", "")

        if field_name in required:
            fields[field_name] = (
                annotation,
                Field(..., description=description),
            )
        else:
            fields[field_name] = (
                Optional[annotation],
                Field(
                    default=prop.get("default"),
                    description=description,
                ),
            )

    args_schema = create_model(
        f"{prefix}_{spec.name}_Args",
        **fields,
    )

    async def invoke_tool(**kwargs):
        # Keep only arguments actually supplied by the model.
        arguments = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

        result = await mcp_client.call_tool(
            spec.name,
            arguments,
        )

        if getattr(result, "isError", False):
            raise RuntimeError(
                f"MCP tool {spec.name} failed: {result}"
            )

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured

        content = getattr(result, "content", [])
        texts = [
            item.text
            for item in content
            if hasattr(item, "text")
        ]

        return "\n".join(texts) if texts else str(result)

    return StructuredTool(
        name=f"{prefix}_{spec.name}",
        description=(
            getattr(spec, "description", None)
            or f"MCP tool: {spec.name}"
        ),
        args_schema=args_schema,
        coroutine=invoke_tool,
    )


async def _init_mcp_tools() -> list[BaseTool]:
    # Open both connections on the dedicated backend event loop.
    await MATH_CLIENT.__aenter__()
    await EXPENSE_CLIENT.__aenter__()

    math_specs = await MATH_CLIENT.list_tools()
    expense_specs = await EXPENSE_CLIENT.list_tools()

    math_tools = [
        _make_langchain_tool(MATH_CLIENT, spec, "math")
        for spec in math_specs
    ]

    expense_tools = [
        _make_langchain_tool(EXPENSE_CLIENT, spec, "expense")
        for spec in expense_specs
    ]

    print("Loaded math tools:", [t.name for t in math_tools])
    print("Loaded ExpenseTracker tools:", [t.name for t in expense_tools])

    return [*math_tools, *expense_tools]


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(_init_mcp_tools())
    except Exception as e:
        print("MCP TOOL LOADING FAILED:", repr(e))
        raise


mcp_tools = load_mcp_tools()



def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(_init_mcp_tools())
    except Exception as e:
        print("MCP TOOL LOADING FAILED:", repr(e))
        raise




mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, rag_tool, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

print("ALL TOOLS:", [tool.name for tool in tools])
print("LLM BOUND TOOLS:", [
    tool.name for tool in tools
])

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())




def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})