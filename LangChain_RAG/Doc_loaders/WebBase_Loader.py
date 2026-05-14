from langchain_community.document_loaders import WebBaseLoader

url="https://www.youtube.com/watch?v=bL92ALSZ2Cg&list=PLKnIA16_RmvaTbihpo4MtzVm4XOQa0ER0&index=13"
loader=WebBaseLoader(url)

doc = loader.load()

print(len(doc))

print(doc[0].page_content)