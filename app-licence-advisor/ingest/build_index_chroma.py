
"""
build_index_chroma.py
Build a Chroma-backed vector index for the POC corpus.

Requires:
  - Chroma server running (via docker-compose at localhost:8000)
  - chromadb, llama-index-vector-stores-chroma

Run:
  python ingest/build_index_chroma.py \
      --base_dir data/poc \
      --collection licenses_poc \
      --host localhost \
      --port 8000
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

from loaders import load_poc_corpus

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

def build_index(
    base_dir: str,
    collection: str,
    host: str = "localhost",
    port: int = 8000,
) -> None:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base.resolve()}")

    # 1) Load documents
    docs = load_poc_corpus(base)
    if not docs:
        raise RuntimeError(
            f"No documents loaded from {base.resolve()}. "
            "Add files under data/poc/licenses/, data/poc/web/, data/poc/spdx/."
        )
    print(f"Loaded {len(docs)} documents from {base.resolve()}")

    # 2) Configure parser + embeddings (consistent embed model for build & query)
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="source_text",
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-small")

    # 3) Connect to Chroma (HTTP client) and get/create collection
    client = chromadb.HttpClient(host=host, port=port)
    coll = client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"}  # cosine is typical for text embeddings
    )

    # 4) Wrap Chroma collection with LlamaIndex vector store
    vector_store = ChromaVectorStore(chroma_collection=coll)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 5) Build index (vectors go straight into Chroma; no local ./storage needed)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    print(f"✅ Indexed into Chroma collection: {collection} at {host}:{port}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a Chroma-backed index for the POC corpus.")
    p.add_argument("--base_dir", type=str, default="data/poc", help="Corpus base dir")
    p.add_argument("--collection", type=str, default="licenses_poc", help="Chroma collection name")
    p.add_argument("--host", type=str, default=os.getenv("CHROMA_HOST", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_index(args.base_dir, args.collection, args.host, args.port)
