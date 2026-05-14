from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

try:
    documents = retriever.invoke(query)

    print(f"Total documents: {len(documents)}\n")

    for index, document in enumerate(documents, start=1):
        print(f"Document {index}:")
        print(document.page_content[:500])
        print("-" * 50)
except Exception as error:
    print("Wikipedia request failed.")
    print(error)
