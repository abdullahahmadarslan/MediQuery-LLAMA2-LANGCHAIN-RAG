from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma
from uuid import uuid4


extracted_data = load_pdf("D:\Generative Ai\MediQuery-LLAMA2-LANGCHAIN-RAG\data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

vector_store = Chroma(
    collection_name="mediquery",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

uuids = [str(uuid4()) for _ in range(len(text_chunks))]

vector_store.add_documents(documents=text_chunks, ids=uuids)



