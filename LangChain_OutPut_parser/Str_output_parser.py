from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm =HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct:cerebras",
    task="text-generation"
)

model =ChatHuggingFace(llm=llm)

template1=PromptTemplate(
    template="Write a brief explaination on the {topic}"
)

template2=PromptTemplate(
    template="Write a brief 5 line summary on the {text}"
)

prompt1 = template1.invoke({ "topic":"black Hole"})
result=model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1 = model.invoke(prompt2)

print(result1.content)