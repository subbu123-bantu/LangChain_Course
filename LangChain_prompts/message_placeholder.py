from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "you are a helpful customer agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

history_file = Path(__file__).parent / "chat_history.txt"

with open(history_file, encoding="utf-8") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "what is the progress"
})

print(prompt)
