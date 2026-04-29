from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults

import sqlite3
import requests
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

search_tool = DuckDuckGoSearchResults()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithemetic operation on two numbers.
    """
    if operation == 'add':
        return first_num + second_num
    elif operation == 'subtract':
        return first_num - second_num
    elif operation == 'multiply':
        return first_num * second_num
    elif operation == 'divide':
        if second_num == 0:
            raise ValueError("Cannot divide by zero")
        return first_num / second_num
    else:
        raise ValueError("Invalid operation. Supported operations are: add, subtract, multiply, divide.")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=R3PZAB0CGB0TY32I"
    r = requests.get(url)
    return r.json()


tools = [search_tool, calculator, get_stock_price]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """
    LLM node that may answer or request a tool call
    """
    messages = state['messages']

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database= 'chat.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)