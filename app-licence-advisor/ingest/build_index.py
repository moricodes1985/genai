
"""
Build and persist the vector index for the POC corpus.

- Loads a small, mixed-format license dataset via loaders.load_poc_corpus()
- Uses a sentence-window parser (better for long, sectioned legal text)
- Embeds with a strong small model (gte-small) for fast local indexing
- Persists the index to disk so the app can load it later

Run:
    python ingest/build_index.py \
        --base_dir data/poc \
        --persist_dir ./storage
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loaders import load_poc_corpus

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_index(base_dir: str, persist_dir: str) -> None:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base.resolve()}")

    # 1) Load documents (mixed formats: txt/md/html/csv…)
    docs = load_poc_corpus(base)
    if not docs:
        raise RuntimeError(
            f"No documents loaded from {base.resolve()}. "
            "Add files under data/poc/licenses/, data/poc/web/, data/poc/spdx/."
        )

    print(f"Loaded {len(docs)} documents from {base.resolve()}")

    # 2) Configure parsing + embeddings (no LLM needed for indexing)
    # Sentence-window parser keeps a small surrounding context, which helps legal Q&A.
    Settings.node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="source_text",
    )
    # Fast, high-quality small embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-small")

    # 3) Build vector index
    index = VectorStoreIndex.from_documents(docs)

    # 4) Persist to disk
    persist = Path(persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist))

    print(f"✅ Index built and persisted to: {persist.resolve()}")
    print("You can now run:  python app/main.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and persist the POC vector index.")
    p.add_argument(
        "--base_dir",
        type=str,
        default="data/poc",
        help="Base directory for the POC corpus (default: data/poc)",
    )
    p.add_argument(
        "--persist_dir",
        type=str,
        default="./storage",
        help="Directory to persist the index (default: ./storage)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.base_dir, args.persist_dir)
