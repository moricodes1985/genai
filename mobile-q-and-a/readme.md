## 📚 FastAPI RAG Q&A Service (FAISS Edition)

A Retrieval-Augmented Generation (RAG) Q&A API built with FastAPI, LangChain, and FAISS for fast, in-memory vector search.
The service uses HuggingFace embeddings for document indexing and OpenAI Chat models for natural language responses, with per-user conversational memory.

## 🚀 Features

Retrieval-Augmented Generation with FAISS (in-memory)

HuggingFace Sentence Transformers for embedding generation

OpenAI GPT models for question answering

Per-user conversation history using LangChain's ConversationBufferMemory

Lazy loading: FAISS index is built on first request, speeding up startup time

REST API powered by FastAPI

/health endpoint for service monitoring

## 🛠️ Tech Stack

FastAPI – API framework

LangChain – LLM orchestration

FAISS – In-memory vector database

HuggingFace Transformers – Embeddings model

OpenAI API – LLMs for answering questions

## questions

📦 Installation

1️⃣ Clone the Repository

```
git clone https://github.com/yourusername/fastapi-rag-faiss.git
cd fastapi-rag-faiss

```

2️⃣ Install Dependencies

```
pip install fastapi uvicorn langchain langchain-community langchain-openai \
            langchain-text-splitters sentence-transformers faiss-cpu \
            openai tiktoken requests python-dotenv

```

3️⃣ Set Environment Variables

```
OPENAI_API_KEY=sk-your-openai-key
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
OPENAI_CHAT_MODEL=gpt-4o-mini
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RETRIEVAL_K=4
REQUEST_TIMEOUT=30
```

▶️ Running the App

```
uvicorn main:app --reload

```
OR Use Docker

```
docker compose build --no-cache
docker compose up
```

## 
📡 API Endpoints
1. Health Check
```
curl http://127.0.0.1:8000/health

```
2. Ask a Question
```
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "user_id": "u123",
           "question": "What is the company policy on leave?"
         }'

```

## 💬  Chatbot UI
Gradio UI
```
http://localhost:7860/
```

## ⚙️ How It Works

Startup

API starts eager initialization strategy on startup (FAISS index).

On Startup

Downloads the policy document from POLICY_URL

Splits text into chunks

Creates embeddings using HuggingFace model

Stores vectors in FAISS (in-memory)

On Each Question

Retrieves top-K relevant chunks from FAISS

Passes them with conversation history to OpenAI model

Returns answer + source snippets

## Quick mental model (sequence)
```
user question
   │
   ▼
[Memory] load chat_history  ───────────────┐
   │                                       │
   ▼                                       │
[Condense LLM] chat_history + question → standalone_question
   │
   ▼
[Retriever] standalone_question → top-K docs
   │
   ▼
[QA LLM] prompt(context=docs, question=standalone_question) → answer
   │
   ├─► [Memory] append (Human: question, AI: answer)
   ▼
return {answer, source_documents}

```
## Some example questions to ask from chatbot
```
What are the core principles in the Code of Conduct (integrity, respect, accountability, safety, environmental responsibility)?

Does the Recruitment Policy guarantee equal opportunity and transparent job ads?

Are employees allowed limited personal internet use and under what conditions?

When is encryption required for sending confidential data by email?

Is vaping allowed inside company buildings?

Are employees allowed to drink alcohol during work hours? Any exceptions?

What should I do if my company phone is lost or stolen?

Can the company monitor internet and email usage?
```