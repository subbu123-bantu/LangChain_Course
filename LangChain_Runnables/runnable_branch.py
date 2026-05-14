from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableBranch,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt =PromptTemplate(
    template="Write a detailed explaination on end to end  {topic}",
    input_variables=["topic"]
)

propmt2 = PromptTemplate(
    template="Generate the summery of the \n {text}",
    input_variables=["text"]
)

text_gen_chain= RunnableSequence(prompt,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())> 100,RunnableSequence(propmt2,model,parser)),
    RunnablePassthrough()
)

final_chian = RunnableSequence(text_gen_chain,branch_chain)

print(final_chian.invoke({"topic":"LangChain"}))