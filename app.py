from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
import time


app = FastAPI()

# Load embeddings from hugging face
embeddings = download_hugging_face_embeddings()
# Initializing chroma db
docsearch = Chroma(
    collection_name="mediquery",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db", 
)
print("Chroma initialized")

# Introduce a delay to simulate waiting for setup
time.sleep(2)  # Sleep for 2 seconds


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Setting up the LLM
llm=CTransformers(model="D:\Generative Ai\MediQuery-LLAMA2-LANGCHAIN-RAG\LLM/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


# Create the retriever and setup QA only after ensuring Chroma is initialized
retriever = docsearch.as_retriever(search_kwargs={'k': 2})
print("Retriever set up")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
print("QA chain set up")



# FastAPI route for rendering the initial chat page (index)
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/chat.html") as f:
        return f.read()  # Return the content of the HTML file for the chat interface

# FastAPI route to handle chat requests
@app.post("/get")
async def chat(msg: str = Form(...)):
    print(f"User message: {msg}")
    result = qa({"query": msg})
    print("Response: ", result["result"])
    return {"response": result["result"]}  # Return the response as JSON

# To run the app with Uvicorn:
# uvicorn main:app --reload --host 0.0.0.0 --port 8080
