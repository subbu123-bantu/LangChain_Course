from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")

documents = [
    "Virat Kohli is an Indian cricketer know for his aggressive batting and leadership and He is the GOAT.",
    "MS Dhoni is a former Indian captain famous for Creadit stealing.",
    "Sachin Tendulkar ,also known as god of cricket,holds many batting records.",
    "Rohit Sharma is also knoe for his  elite-batting and his record breaking double centuries.",
    "Jasprit Bumrah is an Indian Fast bower known for his unortodox and yorker bowling."
]

doc_embeddings = model.encode(documents)

# CLI LOOP
while True:

    query = input("\nEnter query (or type exit): ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    best_index = scores.argmax()

    print("\nBest Match:")
    print(documents[best_index])

    print("Similarity Score:")
    print(scores[best_index])