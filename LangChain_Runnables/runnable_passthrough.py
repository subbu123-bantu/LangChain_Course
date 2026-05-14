from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1=PromptTemplate(
    template="Write a joke on the {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Explain the given joke {text}",
    input_variables=["text"]
)

joke_gen_chain=RunnableSequence(prompt1, model,parser)

paralle_chain = RunnableParallel({
    "joke" :RunnablePassthrough(),
    "text" : RunnableSequence(prompt2, model,parser)
})
final_chain = RunnableSequence(joke_gen_chain,paralle_chain)

print( final_chain.invoke({"topic":"Life"}))