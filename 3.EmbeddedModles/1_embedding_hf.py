from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(model='BAAI/bge-small-en-v1.5')

result =embedding.embed_query("Delhi is capital of India")

print(result)

