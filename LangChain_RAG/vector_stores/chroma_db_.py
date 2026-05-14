from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

# Create LangChain documents for IPL players
load_dotenv() 

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.He is the greatest of all time in cricket",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs=[doc1,doc2,doc3,doc4,doc5]
doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store=Chroma(
    collection_name="my_collection",
    embedding_function=embeddings,
    persist_directory="./my_chroma_db"
)

vector_store.delete(ids=doc_ids)
vector_store.add_documents(documents=docs, ids=doc_ids)
# result = vector_store.get(include=["embeddings", "documents", "metadatas"])
# print(result)

res= vector_store.similarity_search(
    query="who plays for Royal Challengers Bengaluru",
    k=1
)

for index, document in enumerate(res, start=1):
    print(f"Result {index}:")
    print(document.page_content)
    print(document.metadata)
    print("-" * 50)

res2=vector_store.similarity_search_with_score(
    query="",
    filter={"team":"Royal Challengers Bengaluru"}
)

print(res2)


