import os
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Type
from langchain_core.messages import HumanMessage

load_dotenv()


class MultiplyInput(BaseModel):
    a: int = Field(description="First number to multiply")
    b: int = Field(description="Second number to multiply")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers."
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int):
        return a * b

multiply_tool = MultiplyTool()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm_with_tools = llm.bind_tools([multiply_tool])

query=HumanMessage("Can you multiply 3 with 100?")

messages=[query]

result=llm_with_tools.invoke(messages)

messages.append(result)

tool_result = multiply_tool.invoke(result.tool_calls[0])

messages.append(tool_result)


print(llm_with_tools.invoke(messages).content)
