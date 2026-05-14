from langchain_core.tools import tool
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'

  response = requests.get(url, timeout=20)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 100 inr to usd')]
conversion_rate = None

response = llm_with_tools.invoke(messages)

while response.tool_calls:
  messages.append(response)

  for tool_call in response.tool_calls:
    # execute the 1st tool and get the value of conversion rate
    if tool_call['name'] == 'get_conversion_factor':
      tool_output1 = get_conversion_factor.invoke(tool_call['args'])
      # fetch this conversion rate
      conversion_rate = tool_output1['conversion_rate']
      # append this tool message to messages list
      messages.append(
        ToolMessage(
          content=json.dumps(tool_output1),
          tool_call_id=tool_call['id'],
          name=tool_call['name']
        )
      )
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name'] == 'convert':
      # fetch the current arg
      tool_call['args']['conversion_rate'] = conversion_rate
      tool_output2 = convert.invoke(tool_call['args'])
      messages.append(
        ToolMessage(
          content=str(tool_output2),
          tool_call_id=tool_call['id'],
          name=tool_call['name']
        )
      )

  response = llm_with_tools.invoke(messages)

print(response.content)


