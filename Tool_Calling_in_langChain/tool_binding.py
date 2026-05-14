from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool

def multiply(a:int ,b: int) -> int:
    """This gives the product of 2 numbers"""
    return a*b
llm = ChatOllama(model="mistral")
llm_with_tools = llm.bind_tools([multiply])

tool_binding=llm.bind_tools([multiply])