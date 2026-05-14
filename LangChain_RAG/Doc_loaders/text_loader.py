from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 120}
)

prompt = PromptTemplate(
    template="Write a summary on {poem}",
    input_variables=["poem"]
)

parser = StrOutputParser()

loader = TextLoader("LangChain_RAG/Doc_loaders/cricker.txt", encoding="utf-8")
docs = loader.load()

chain = prompt | llm | parser
print(chain.invoke({"poem": docs[0].page_content}))
