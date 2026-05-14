from langchain_core.tools import tool,BaseToolkit


@tool
def multiply(a:int ,b:int) -> int :
    """multiply two numbers"""
    return a*b

@tool
def add(a:int ,b:int) -> int :
    """add two numbers"""
    return a+b

class MathToolkit():
    def get_tools(self):
        return [add,multiply]
    
toolkit=MathToolkit()
tools =toolkit.get_tools()

for tool in tools:
    result = tool.invoke({"a":3,"b":6})
    print(tool.name ,"=>" ,tool.description,"result", result)