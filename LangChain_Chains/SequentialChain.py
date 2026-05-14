from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
)

model =ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="Explain about the {topic} in detail",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Give 5 line Summary of the following. \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'unemployment in India '})

print(result)

chain.get_graph().print_ascii()
