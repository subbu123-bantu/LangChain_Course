from pathlib import Path

from langchain_community.document_loaders import CSVLoader

csv_path = Path(__file__).with_name("tickets.csv")
loader = CSVLoader(file_path=str(csv_path))

doc = loader.load()

print(len(doc))
print(doc[999])