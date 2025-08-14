# main.py
import os
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------------
# 1) Config & Keys
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")  # change if you like

# -----------------------------
# 2) Minimal OpenAI chat client
# -----------------------------
from openai import OpenAI
oai = OpenAI(api_key=OPENAI_API_KEY)

from openai import APIError, RateLimitError, AuthenticationError

def chat(messages: List[Dict[str, str]], model: str = OPENAI_CHAT_MODEL, temperature: float = 0.2) -> str:
    """Single-turn chat wrapper with error handling."""
    try:
        resp = oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    except RateLimitError:
        return "[OpenAI] Quota/limit issue. Check project billing."
    except AuthenticationError:
        return "[OpenAI] Invalid API key."
    except APIError as e:
        return f"[OpenAI API error] {e}"


# -----------------------------
# 3) Tiny in-memory corpus
#    (Replace with your PDFs later)
# -----------------------------
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Company A Q2 Earnings",
        "text": """Revenue grew 12% YoY while operating income increased 10%.
Costs were stable, leading to a slight margin expansion.
Management expects continued operating leverage in H2.""",
        "source": "internal_reports/company_a_q2.txt",
    },
    {
        "id": "doc2",
        "title": "Company B Annual Report",
        "text": """Operating income growth was driven by efficiency gains.
Gross margin improved due to pricing power, despite FX headwinds.""",
        "source": "internal_reports/company_b_annual.txt",
    },
    {
        "id": "doc3",
        "title": "Sector Note: Profitability Trends",
        "text": """Profit margin changes across the sector correlate with input cost normalization and
headcount controls. Companies with high operating leverage saw faster margin recovery.""",
        "source": "notes/sector_profitability_trends.txt",
    },
]

# -----------------------------
# 4) Embeddings & Vector Store (NumPy-only)
# -----------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

def embed_texts(texts: List[str]) -> np.ndarray:
    # Returns L2-normalized vectors so dot product == cosine similarity
    X = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return X.astype("float32")

# Precompute doc matrix once (one chunk per doc for simplicity)
doc_texts = [d["text"] for d in DOCUMENTS]
DOC_MATRIX = embed_texts(doc_texts)  # shape: (N, dim)
ID2DOC = {i: DOCUMENTS[i] for i in range(len(DOCUMENTS))}

# -----------------------------
# 5) Super-simple tool: mock live finance API
#    (An "agent" decides when to call it)
# -----------------------------
def mock_live_price_api(symbol: str) -> Dict[str, Any]:
    # Pretend to hit a market data API
    now = int(time.time())
    return {
        "symbol": symbol.upper(),
        "price": round(100 + (now % 17) * 0.5, 2),  # fake moving price
        "currency": "USD",
        "as_of": now
    }

def should_use_price_tool(question: str) -> Optional[str]:
    """
    If user asks about price for a ticker, extract a naive symbol and return it.
    """
    q = question.lower()
    if "price" in q or "quote" in q:
        # toy extraction: look for common tickers
        for t in ["tsla", "aapl", "msft", "amzn"]:
            if t in q:
                return t
    return None

# -----------------------------
# 6) RAG: Retrieve → Augment → Generate (NumPy cosine)
# -----------------------------
def retrieve(query: str, k: int = 3):
    qvec = embed_texts([query])[0]          # (dim,)
    sims = DOC_MATRIX @ qvec                # cosine similarity (normalized vectors)
    topk = np.argsort(-sims)[:k]
    results = []
    for rank, i in enumerate(topk, start=1):
        doc = ID2DOC[int(i)]
        results.append({
            "rank": rank,
            "score": float(sims[i]),
            "id": doc["id"],
            "title": doc["title"],
            "source": doc["source"],
            "text": doc["text"]
        })
    return results

SYSTEM_PROMPT = """You are a precise finance assistant.
Always ground answers in the provided CONTEXT.
If needed, include a short calculation and call out uncertainties clearly.
When appropriate, include bullet points and cite sources with (title, source)."""

def build_prompt(question: str, retrieved: List[Dict[str, Any]], tool_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    context_blobs = []
    for r in retrieved:
        context_blobs.append(f"[{r['title']} | {r['source']}]\n{r['text']}")
    context_text = "\n\n---\n\n".join(context_blobs)

    tool_section = ""
    if tool_results:
        tool_section = f"\n\nTOOL_DATA (market):\n{json.dumps(tool_results, indent=2)}\n"

    user_block = f"""QUESTION:
{question}

CONTEXT:
{context_text}
{tool_section}

INSTRUCTIONS:
1) Answer using only the CONTEXT and TOOL_DATA when present.
2) Be concise and cite (title, source) after claims.
3) If the question is about profitability (e.g., margins, operating income), relate them explicitly.
4) If the user asked for price/quote, include TOOL_DATA values.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]

# -----------------------------
# 7) FastAPI app
# -----------------------------
app = FastAPI(title="AI Financial Research Assistant (Mini)")

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    retrieved: List[Dict[str, Any]]
    tool_used: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    return {"ok": True, "message": "Use POST /ask with {question: ...}"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Decide if we need the tool (the "agent" part, deliberately simple)
    maybe_symbol = should_use_price_tool(req.question)
    tool_data = mock_live_price_api(maybe_symbol) if maybe_symbol else None

    # Retrieve relevant docs
    retrieved = retrieve(req.question, k=req.top_k)

    # Build prompt and call the LLM
    messages = build_prompt(req.question, retrieved, tool_results=tool_data)
    answer = chat(messages)

    return AskResponse(answer=answer, retrieved=retrieved, tool_used=tool_data)
