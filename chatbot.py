import langchain
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from extraction import extracted_summaries
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = FAISS.from_texts(extracted_summaries, embedding_model)
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
"/kaggle/input/phi-3/pytorch/phi-3.5-mini-instruct/2",
device_map="cuda",
torch_dtype="auto",
trust_remote_code=True,
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,  
    temperature=0.1,  
    do_sample=False  
)   
phi_llm = HuggingFacePipeline(pipeline=llm_pipeline)


qa_chain = RetrievalQA.from_chain_type(
    llm=phi_llm,
    retriever=vector_db.as_retriever(),
    return_source_documents=True  
)

def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})  
    return response["result"]  

query = "What are the main causes for appeals in cases?"
response = chat_with_bot(query)
print("Chatbot Answer:", response)