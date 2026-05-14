from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct:cerebras",
    task="text-generation"
)

model= ChatHuggingFace(llm=llm)

parser= JsonOutputParser()

tempalte =PromptTemplate(
    template="give me a name age, city of a fictional person.\n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction" : parser.get_format_instructions()} 
)


chain = tempalte | model | parser

result = chain.invoke({})

print(result)