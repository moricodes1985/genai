"""
FastAPI RAG Q&A service with per-user conversational memory (FAISS edition)

Stack:
- LangChain
- FAISS (in-memory)
- HuggingFaceEmbeddings
- OpenAI (Chat) for LLM
- PromptTemplate
- ConversationalRetrievalChain + ConversationBufferMemory

Run:
  export OPENAI_API_KEY=your_key_here
  pip install fastapi uvicorn langchain langchain-community langchain-openai langchain-text-splitters \
              sentence-transformers faiss-cpu openai tiktoken requests
  uvicorn main:app --reload

Endpoint:
  POST /ask
  Body: {"user_id":"u123", "question":"..."}
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
import logging
from typing import Dict, Any, List

import requests
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel

# --- LangChain imports ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

# Ensure tokenizer threads don’t deadlock on Windows
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
API_PREFIX = os.getenv("API_PREFIX", "/api")
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("rag")

# Fallback import (older LC versions)
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:  # pragma: no cover
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import ChatOpenAI. Install 'langchain-openai' or use a LangChain version that provides langchain.chat_models.ChatOpenAI"
        ) from e

# ---------------------- Config ----------------------
POLICY_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt"
)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------- App ----------------------
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, api_key= OPENAI_API_KEY)
app = FastAPI(title="RAG Q&A API (FAISS)", version="1.0.0")
api = APIRouter(prefix=API_PREFIX)
app.include_router(api)
# Globals
embeddings = None  # type: ignore
vectorstore: FAISS | None = None
retriever = None  # type: ignore
_user_memories: Dict[str, ConversationBufferMemory] = {}
_mem_lock = threading.Lock()
_init_lock = threading.Lock()

# ---------------------- Models ----------------------
class AskRequest(BaseModel):
    user_id: str
    question: str

class AskResponse(BaseModel):
    user_id: str
    answer: str
    sources: List[Dict[str, Any]]

# ---------------------- Utilities ----------------------
def _download_policy_to_temp(url: str) -> str:
    """Download the policy text file to a temporary path and return the path."""
    log.info("Downloading policy from %s", url)
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        log.exception("Failed to download policy file")
        raise RuntimeError(f"Failed to download policy file: {e}")

    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    log.info("Policy file saved to %s (%d bytes)", path, len(resp.content))
    return path


def _build_vectorstore() -> FAISS:
    """Load, split, embed the policy doc into an in-memory FAISS index."""
    start_all = time.time()
    log.info("Starting vectorstore build (FAISS)...")

    t0 = time.time()
    tmp_path = _download_policy_to_temp(POLICY_URL)
    log.info("Downloaded policy in %.2fs", time.time() - t0)

    t0 = time.time()
    loader = TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    log.info("Loaded %d doc(s) in %.2fs", len(docs), time.time() - t0)

    t0 = time.time()
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separator="\n"
    )
    chunks = splitter.split_documents(docs)
    log.info("Split into %d chunks in %.2fs", len(chunks), time.time() - t0)

    global embeddings
    if embeddings is None:
        t0 = time.time()
        log.info("Initializing embeddings: %s", EMBED_MODEL_NAME)
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        log.info("Embeddings ready in %.2fs", time.time() - t0)

    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]

    t0 = time.time()
    log.info("Creating in-memory FAISS index...")
    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    log.info(
        "FAISS index built in %.2fs (total build %.2fs)",
        time.time() - t0,
        time.time() - start_all,
    )
    return vs


def _get_or_create_memory(user_id: str) -> ConversationBufferMemory:
    with _mem_lock:
        mem = _user_memories.get(user_id)
        if mem is None:
            log.info("Creating new conversation memory for user_id=%s", user_id)
            mem = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="question",
                output_key="answer",          # <-- add this
                return_messages=True,
            )
            _user_memories[user_id] = mem
        return mem



def _build_prompt() -> PromptTemplate:
    template = (
        "You are a helpful assistant for answering questions using the provided company policy.\n"
        "Follow these rules:\n"
        "- Use the retrieved context to answer.\n"
        "- If the answer is not in the context, say you don't know based on the policy.\n"
        "- Be concise and cite key policy points.\n\n"
        "Context from documents:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])



def _ensure_retriever_ready():
    """Lazy-initialize FAISS index and retriever on first request."""
    global vectorstore, retriever
    if retriever is not None:
        return
    with _init_lock:
        if retriever is not None:
            return
        log.info("Lazy init: building FAISS vectorstore/retriever ...")
        vectorstore = _build_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
        log.info("Lazy init complete: retriever ready.")


# ---------------------- Startup ----------------------
@app.on_event("startup")
def on_startup() -> None:
    log.info("Application startup")
    _ensure_retriever_ready()
    log.info("Application startup completed")


# ---------------------- Routes ----------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    log.info("Received /ask request from user_id=%s: %s", req.user_id, req.question)
    if not req.question.strip():
        log.warning("Empty question from user_id=%s", req.user_id)
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Per-user memory
    memory = _get_or_create_memory(req.user_id)

    # Prompt and chain
    prompt = _build_prompt()

    if retriever is None:
        log.error("Retriever not initialized after lazy init")
        raise HTTPException(status_code=500, detail="Retriever not initialized.")

    log.info("Creating ConversationalRetrievalChain...")
    chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,  # keeps per-user history for the condense step
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=False,
)


    # Run the chain
    try:
        t0 = time.time()
        result = chain({"question": req.question})
        log.info("Chain run OK in %.2fs", time.time() - t0)
    except Exception as e:
        log.exception("Model error while processing user_id=%s", req.user_id)
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    answer: str = result.get("answer", "")
    source_docs = result.get("source_documents", [])
    log.info("Generated answer; sources=%d", len(source_docs))

    sources: List[Dict[str, Any]] = []
    for d in source_docs:
        snippet = d.page_content[:300].strip().replace("\n", " ")
        sources.append(
            {
                "source": d.metadata.get("source", "policy.txt"),
                "snippet": snippet,
            }
        )

    return AskResponse(user_id=req.user_id, answer=answer, sources=sources)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
