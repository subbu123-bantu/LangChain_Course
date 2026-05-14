from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path=,
    glob=,
    loadet_cls =PyPDFLoader
)

doc = loader.load()

print(doc)