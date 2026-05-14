from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(r"C:\Users\bantu\Downloads\sample_health_policy.pdf")

docs = loader.load()

print(docs)