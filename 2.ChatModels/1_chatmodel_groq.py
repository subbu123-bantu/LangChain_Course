from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model='llama-3.3-70b-versatile',temperature=1.8,max_completion_tokens=100)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)