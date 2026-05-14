from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline


llm=HuggingFacePipeline.from_model_id(
    model_id='meta-llama/Llama-3.1-8B-Instruct:cerebras',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India")

print(result)

#huggingFace should be downloaded locally
#to dowmload in d drive
#import os 
#os.enviorn['HF_HOME']='D:/huggingface_cache