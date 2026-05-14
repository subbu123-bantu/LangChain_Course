from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct:cerebras",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="write a brief explaination on{topic}"
)

template2=PromptTemplate(
    template="write a 5 line summary of {text}"
)
parser =StrOutputParser()

chain= template1 | model | parser | template2 | model | parser 

result = chain.invoke({"topic":"black hole"})

print(result)