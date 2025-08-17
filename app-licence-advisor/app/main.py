
"""
Backend service for RAG demo.
- Ensures index is built/loaded at startup
- Exposes /chat endpoint for UI clients (e.g., Gradio)
"""

import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from app.gaurds import check_input, wrap_output, DISCLAIMER
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
except Exception:
    chromadb = None
    ChromaVectorStore = None

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ingest"))


# ---------------- Config ----------------
BACKEND = os.getenv("RAG_BACKEND", "simple").lower()
BASE_DIR = os.getenv("BASE_DIR", "data/poc")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "licenses_poc")

app = FastAPI(title="License RAG Service")
QE = None  # Query engine


# ---------------- Helpers ----------------
def ensure_index_simple() -> VectorStoreIndex:
    persist = Path(PERSIST_DIR)
    has_index = (persist / "docstore.json").exists()
    if not has_index:
        simple_builder.build_index(BASE_DIR, PERSIST_DIR)
    storage = StorageContext.from_defaults(persist_dir=str(persist))
    return load_index_from_storage(storage)


def ensure_index_chroma() -> VectorStoreIndex:
    if chromadb is None or ChromaVectorStore is None:
        raise RuntimeError("Chroma not installed")
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Always create if missing
    try:
        coll = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        # As a fallback, build then try again
        if chroma_builder is None:
            raise RuntimeError(f"Chroma collection missing and no builder available: {e}")
        chroma_builder.build_index(BASE_DIR, COLLECTION_NAME, CHROMA_HOST, CHROMA_PORT)
        coll = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    vector_store = ChromaVectorStore(chroma_collection=coll)
    return VectorStoreIndex.from_vector_store(vector_store)



# ---------------- API Models ----------------
class ChatRequest(BaseModel):
    message: str
    strict: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------- Startup ----------------
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ...

@app.on_event("startup")
async def startup_event():
    global QE

    # --- Force query-time embedder to match index-time model ---
    embed_name = os.getenv("EMBED_MODEL_NAME", "thenlper/gte-small")
    hf_embed = HuggingFaceEmbedding(model_name=embed_name)
    Settings.embed_model = hf_embed

    # Debug: log the embed dim to ensure it's 384 (gte-small)
    try:
        dim = len(hf_embed.get_text_embedding("ping"))
        print(f"🔧 Using embedder '{embed_name}' with dimension={dim}")
    except Exception as e:
        print(f"⚠️ Failed to probe embedding dim: {e}")

    if BACKEND == "chroma":
        index = ensure_index_chroma()
    else:
        index = ensure_index_simple()

    QE = index.as_query_engine(
        similarity_top_k=12,
        node_postprocessors=[],
        response_mode="compact"
    )
    print("✅ Query engine is ready")




# ---------------- Endpoints ----------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    ok, reason = check_input(req.message)
    if not ok:
        return ChatResponse(answer=f"{DISCLAIMER}\n\n❌ {reason}", sources=[])

    # --- Query ---
    resp = QE.query(req.message)

    # Use only the plain model output
    answer = resp.response or ""

    if req.strict:
        answer += "\n\n(Strict mode: focusing on quotes.)"

    # --- Collect sources ---
    sources = list({
        sn.node.metadata.get("source") or sn.node.metadata.get("file_path", "")
        for sn in getattr(resp, "source_nodes", []) or []
    })

    # --- Final wrap (adds disclaimer + one Sources block) ---
    final = wrap_output(answer, sources)
    return ChatResponse(answer=final, sources=sources)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}
