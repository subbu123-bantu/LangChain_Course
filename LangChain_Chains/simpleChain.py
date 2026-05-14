from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct:cerebras",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="write 5 facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = template | model | parser 


result = chain.invoke({"topic":"Cricket"})

print(result)

chain.get_graph().print_ascii()
