

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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


model = ChatGroq(
    model_name="qwen/qwen3.8-27b",
    temperature=0
)

print(model.invoke('what is the capital of india'))
