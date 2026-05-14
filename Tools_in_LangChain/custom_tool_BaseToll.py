from langchain_core.tools import BaseTool
from pydantic import BaseModel,Field
from typing import Type

class MultiplyInput(BaseModel):
    a : int =Field(description="First num of multiplication")
    b : int =Field(description="second num of multiplication")

class MultiplyTool(BaseTool):
    name:str= "multiply"
    description:str = "Multiplication of 2 numbers"
    args_schema :Type[BaseModel] = MultiplyInput

    def _run(self, a:int, b : int):
        return a * b
    
multiply_tool=MultiplyTool()

result= multiply_tool.invoke({"a":3,"b":6})

print(result)
