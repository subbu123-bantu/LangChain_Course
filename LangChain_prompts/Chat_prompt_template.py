from langchain_core.prompts import ChatPromptTemplate

chat_template=ChatPromptTemplate([
    ('system','you are an expert in {domain}'),
    ('human','explain about {topic} in simple terms ')
])

prompt =chat_template.invoke({ "domain":"cricket","topic":"keeping"})

print(prompt)