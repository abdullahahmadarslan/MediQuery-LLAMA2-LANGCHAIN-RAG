from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone 
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from sentence_transformers import SentenceTransformer
import os
# from pinecone import Pinecone, ServerlessSpec

#download embedding model
def download_hugging_face_embeddings():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

embeddings_model = download_hugging_face_embeddings()
api_key=os.environ.get("pinecone_api_key")

pinecone.init(api_key=api_key, environment="us-east-1")

# Specify your Pinecone index name
index_name = "mediquery"

# Create or connect to an existing Pinecone index
index = pinecone.Index(index_name)

pc = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings_model)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="D:\Generative Ai\MediQuery-LLAMA2-LANGCHAIN-RAG\LLM/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=pc.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

while True:
    user_input=input(f"Input Prompt:")
    result=qa({"query": user_input})
    print("Response : ", result["result"])