
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import  SqliteSaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
import requests
from langchain_groq import ChatGroq


load_dotenv()

## let's setup our tools

search_tool= DuckDuckGoSearchRun(region= 'us-en')

model = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0,
    max_tokens=500
)

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
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}



@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=J6ENQCVCAKFA0X0M"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]

llm_with_tools= model.bind_tools(tools)




class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    


def chat_node(state: ChatState):

    #take user query from user message
    messages= state['messages']

    # send to LLM
    response= llm_with_tools.invoke(messages)

    # response stored in state
    
    return {'messages': response}


tool_node= ToolNode(tools)

conn= sqlite3.connect(database= 'chatbot.db', check_same_thread= False)
checkpointer= SqliteSaver(conn= conn)


graph= StateGraph(ChatState)

#add nodes

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot= graph.compile(checkpointer= checkpointer)


CONFIG= {'configurable':{'thread_id':'thread-1'}}

def retrieveAllThreads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)