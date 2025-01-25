
# MediQuery

MediQuery is a medical chatbot application that leverages cutting-edge natural language processing (NLP) technologies to provide conversational support and answers to medical queries. It combines the power of language models, semantic search, and an intuitive web-based interface to deliver an engaging and efficient user experience.

## Features
- **Conversational Medical Support**: Users can ask medical-related questions and receive informative, AI-generated responses.
- **Retrieval-Augmented Generation (RAG)**: Combines language model capabilities with a retrieval-based system for accurate and context-aware answers.
- **LangChain Integration**: Simplifies and streamlines the interaction between LLaMA 2, embeddings, and document retrieval.
- **FastAPI Backend**: Handles chatbot interactions and serves the frontend.
- **Clean and Responsive Interface**: HTML and CSS-based UI for a user-friendly experience.
- **Embeddings and Vector Search**: Uses Chroma for semantic search, ensuring relevant information is retrieved efficiently.

## Tech Stack
- **Backend**: FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Language Model**: LLaMA 2 (via `CTransformers`)
- **Embeddings Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Database**: Chroma (for document search and embeddings)
- **Integration Framework**: LangChain
- **Deployment**: Uvicorn

---

## Prerequisites
To run MediQuery locally, ensure you have the following installed:

- **Python 3.8+**
- **FastAPI**
- **CTransformers**
- **LangChain**
- **Chroma**
- **Uvicorn**
- Hugging Face embeddings library

You can install dependencies with pip:
```bash
pip install fastapi uvicorn langchain chromadb ctransformers
```

---

## Project Structure
```
MediQuery/
├── chroma_langchain_db/ # Persistent Chroma database for embeddings
├── data/                # Directory for storing data files (if applicable)
├── experiment/          # Placeholder for experiments and additional code
├── LLM/                 # Directory for the LLaMA model binaries
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin # LLaMA model file
├── MediQuery.egg-info/  # Metadata for Python packaging
├── src/                 # Source code, helper modules, and utilities
│   ├── helper.py        # Utility functions (e.g., download embeddings)
│   ├── prompt.py        # Custom prompt templates for the chatbot
│   ├── store_index.py   # Script for initializing and managing the Chroma database
│   └── template.py      # Other templates or helper logic
├── static/              # Static files (if any, e.g., CSS, JS)
├── templates/           # HTML templates
│   └── chat.html        # Main chatbot interface
├── .env                 # Environment variables
├── .gitignore           # Git ignore file
├── app.py               # Main FastAPI application
├── LICENSE              # License for the project
├── README.md            # Documentation for the project
├── requirements.txt     # Python dependencies
└── setup.py             # Setup script for packaging
```

---

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/MediQuery.git
cd MediQuery
```

### Step 2: Download the LLaMA Model
1. Place the LLaMA model file (e.g., `llama-2-7b-chat.ggmlv3.q4_0.bin`) in the `LLM/` directory.

### Step 3: Initialize the Chroma Database
Ensure that embeddings are downloaded and the Chroma database is initialized. This happens automatically when you start the app.

### Step 4: Run the Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Access the chatbot at: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

---

## How It Works
1. **Frontend**:
   - A clean HTML and CSS interface allows users to type questions and view responses.
2. **Backend**:
   - FastAPI processes user input and sends it to the QA pipeline.
3. **Retrieval-Augmented Generation (RAG)**:
   - Embeddings are downloaded via the `download_hugging_face_embeddings` function.
   - Chroma database performs semantic search to retrieve relevant documents.
   - The LLaMA model generates context-aware answers.
   - LangChain coordinates the pipeline, ensuring smooth integration of components.
4. **Response**:
   - The bot's response is displayed in the chat interface.

---

## Key Components

### 1. **Language Model (LLM)**
MediQuery uses the LLaMA 2 model (7B variant) loaded via `CTransformers` for efficient inference.

### 2. **Embeddings Model**
MediQuery employs `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings for efficient semantic search.

### 3. **Chroma**
Chroma is used as a vector database to store and retrieve documents based on their embeddings.

### 4. **LangChain**
LangChain acts as the framework to integrate the LLaMA model, embeddings, and Chroma, ensuring smooth execution of the RAG pipeline.

### 5. **Prompt Engineering**
A custom prompt template ensures high-quality responses that are contextual and relevant to the user’s medical queries.

---

## Development Workflow
1. Modify the HTML interface in `templates/chat.html` to customize the UI.
2. Update prompt engineering logic in `src/prompt.py` to refine the chatbot’s responses.
3. Extend backend functionality in `app.py` to add features like logging, analytics, or additional endpoints.

---

## Future Enhancements
- **Multi-language Support**: Extend the chatbot’s capabilities to support multiple languages.
- **Voice Interaction**: Integrate speech-to-text and text-to-speech functionalities.
- **Custom Knowledge Base**: Allow users to upload documents for personalized responses.
- **Mobile-Friendly Design**: Optimize the interface for mobile devices.

---

## License
This project is licensed under the MIT License.

---

## Acknowledgments
- Hugging Face for embedding models.
- LangChain for modular AI chains.
- Meta for the LLaMA model.
- FastAPI for a modern web framework.

