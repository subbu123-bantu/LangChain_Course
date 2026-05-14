from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

def word_count(text):
    return len(text.split())
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt=PromptTemplate(
    template="Write a joke on the {topic}",
    input_variables=["topic"]
)

joke_chain =RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count" :RunnableLambda(word_count)
})

final_chain =RunnableSequence(joke_chain,parallel_chain)

print(final_chain.invoke({"topic":"Cricket"}))