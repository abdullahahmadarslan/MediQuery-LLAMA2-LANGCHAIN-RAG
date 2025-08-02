from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from src.helper import download_hugging_face_embeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain.llms import CTransformers
from langchain.memory import ConversationSummaryBufferMemory
from src.prompt import *
import time

# New imports for modern RAG chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

app = FastAPI()

# Load embeddings from Hugging Face
embeddings = download_hugging_face_embeddings()

# Initialize Chroma DB
docsearch = Chroma(
    collection_name="mediquery",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)
print("Chroma initialized")

# Introduce a delay to simulate waiting for setup
time.sleep(2)

# Set up the LLM
llm = CTransformers(
    model="D:/Generative Ai/MediQuery-LLAMA2-LANGCHAIN-RAG/LLM/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 250,
        'context_length': 512,
        'temperature': 0.1
    }
)

# Create retriever
retriever = docsearch.as_retriever(search_kwargs={'k': 2})
print("Retriever set up")

# Conversation memory (for summarizing history)
custom_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the following conversation briefly:"),
    ("human", "{summary}\n\n{new_lines}")
])

memory = ConversationSummaryBufferMemory(
    llm=llm,
    prompt=custom_summary_prompt,
    max_token_limit=250,
    return_messages=True,
    output_key="result"
)

# Prompt to turn follow-up questions into standalone questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, "
               "formulate a standalone question which can be understood without the chat history. "
               "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Prompt to generate the final answer using my template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_template.strip()),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create retriever that understands chat history
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q_prompt
)

# Create document answering chain
question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

# Create the complete RAG chain with chat memory
qa = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=question_answer_chain
)

print("QA chain set up")

# Route: chat page
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/chat.html") as f:
        return f.read()

# Route: chat logic
@app.post("/get")
async def chat(msg: str = Form(...)):
    print(f"User message: {msg}")

    # Invoke chain with current memory buffer
    result = qa.invoke({
        "input": msg,
        "chat_history": memory.buffer
    })

    # Extract the answer
    response_text = result["answer"] if "answer" in result else result["result"]
    print("Response: ", response_text)

    # ⬇️ Manually update the memory
    memory.chat_memory.add_user_message(msg)
    memory.chat_memory.add_ai_message(response_text)

    print("\n🔍 Memory contents:")
    print(memory.buffer)

    return {"response": response_text}

