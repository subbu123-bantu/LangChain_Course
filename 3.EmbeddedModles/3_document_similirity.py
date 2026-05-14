from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding=HuggingFaceEmbeddings(model = "BAAI/bge-small-en-v1.5")

documents=[
    "Virar Kohli is an Indian cricketer know for his aggressive batting and leadership and He is the GOAT.",
    "MS Dhoni is a former Indian captain famous for Captaincy.",
    "Sachin Tendulkar ,also known as god of cricket,holds many batting records.",
    "Rohit Sharma is also knoe for his  elite-batting and his record breaking double centuries.",
    "Jasprit Bumrah is an Indian Fast bower known for his unortodox and yorker bowling."
]

query ="who is Goat "


doc_embeddings=embedding.embed_documents(documents)
query_embedding=embedding.embed_query(query)

scores=cosine_similarity([query_embedding],doc_embeddings)[0]

index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print("query is:",query)
print(documents[index])
print("similarity score is :",score)