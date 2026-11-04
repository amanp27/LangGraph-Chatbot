from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
from pathlib import Path

load_dotenv()

DB_PATH = Path(__file__).resolve().parent / "chatbot.db"

llm = ChatOpenAI(model="gpt-4o-mini")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        config = checkpoint.config or {}
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if thread_id is not None:
            all_threads.add(str(thread_id))

    return list(all_threads)

