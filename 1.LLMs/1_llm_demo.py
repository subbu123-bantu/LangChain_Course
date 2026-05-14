from langchain_groq import ChatGroq
from dotenv import load_dotenv


# Load .env
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# Invoke model
result = llm.invoke("What is the capital of Russia?")

print(result.content)