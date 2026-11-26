from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from typing import TypedDict, Annotated, List, Any, Dict, Optional
import sqlite3
import os
import tempfile
import requests

from dotenv import load_dotenv

load_dotenv()

# LLM + embeddings
llm = ChatOpenAI(model = "gpt-4o-mini")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")


# PDF retriever store per session thread
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Any] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


# PDF ingestion and retriever building

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
    Build a FAISS retriever for the uploaded pdf and store it in the thread.
    Returns a summary of dict that can be surface in the UI.
    """

    # This because most of the "PDF LOADER" in langchain only support loading from file path, but not from bytes. So we need to save the uploaded file to a temporary location first.

    # File validation-- check the file is uploaded or not
    if not file_bytes:
        raise ValueError("No file bytes provided")
    
    # Save the uploaded file to a temporary location and can not be deleted automatically after closed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    # --------------------------------------------------------------

    try:
        # Load the pdf and split it into chunks
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitters = RecursiveCharacterTextSplitter(
            chunk_size = 1000, chunk_overlap = 200)
        chunks = text_splitters.split_documents(docs)

        # Build a FAISS retriever
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type = "similarity", search_kwargs = {"k": 4})
        
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            'filename': filename or os.path.basename(temp_path),
            'documents': len(docs),
            'chunks': len(chunks)
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# Tools

search_tool = DuckDuckGoSearchResults()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None):
    """
    Retrieves relevant information from the uploaded pdf document.
    Use this tool when use ask factual/conceptual questions that might be answered from tored documents.
    """

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No retriever found for this thread",
            "query": query
        }
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source": _THREAD_METADATA.get(str(thread_id), {}).get('filename')
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# State

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Nodes

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)


# Checkpointer

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# Graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# Helpers

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})