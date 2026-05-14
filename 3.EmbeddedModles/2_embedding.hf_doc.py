from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(model='BAAI/bge-small-en-v1.5')

document=[
    "Delhi is the capital of India",
    "kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result =embedding.embed_documents(document)

print(result)

