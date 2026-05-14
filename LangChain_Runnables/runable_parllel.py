from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 =PromptTemplate(
    template="Write a tweet on the {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="create a linkedin post on {topic}",
    input_variables=["topic"]
)

parallel_chain =RunnableParallel({
    "tweet":RunnableSequence(prompt1 , model , parser),
    "linkedin_post":RunnableSequence(prompt2, model , parser)
})

chain = parallel_chain.invoke({"topic":"AI"})

print(chain)