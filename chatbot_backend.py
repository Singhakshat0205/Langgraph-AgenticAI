
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

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
model = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0
)

def chat_node(state: ChatState):

    #take user query from user message
    messages= state['messages']

    # send to LLM
    response= model.invoke(messages)

    # response stored in state
    
    return {'messages': response}


checkpointer= MemorySaver()


graph= StateGraph(ChatState)

#add nodes

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')

graph.add_edge('chat_node', END)

chatbot= graph.compile(checkpointer= checkpointer)

# for message_chunk, metadata in chatbot.stream(
#         {'messages': [HumanMessage(content= 'what is the recipe to make pasta')]
#         },
#         config= {'configurable': {'thread_id':'thread-1'}},
#         stream_mode= 'messages'
#     ):
#     if message_chunk.content:
#        print(message_chunk.content, end=' ', flush= True)




