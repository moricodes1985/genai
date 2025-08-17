# 📚 Open-Source License Copilot (RAG Demo)

An interactive chatbot that answers **questions about open-source software licenses** using **Retrieval-Augmented Generation (RAG)**.  
Built with [LlamaIndex](https://www.llamaindex.ai/), an LLM backend, and [Gradio](https://gradio.app/) for the UI.

⚠️ Disclaimer: This tool is **for educational purposes only** and **not legal advice**.

---

## ✨ Features
- **Multi-source document ingestion**  
  - Plaintext license files (e.g., MIT, Apache-2.0)  
  - Markdown guides (e.g., choosealicense.com excerpts)  
  - HTML pages (e.g., OSI license descriptions)  
  - Structured CSV/JSON (e.g., SPDX license metadata)  

- **Retrieval-augmented QA**  
  - Embedding-based retrieval with [HuggingFace models](https://huggingface.co/)  
  - Citations to source documents in every answer  

- **Interactive Chatbot**  
  - Gradio chat UI with conversation history  
  - Optional file upload (e.g., `package.json`) for contextual queries  
  - Toggle between *strict quoting* vs *helpful explanation* modes  

- **Extensible Architecture**  
  - Easy to add more licenses, documents, or retrievers  
  - Modular pipeline for future improvements (rerankers, graph-RAG, evaluations)  

---

## 🗂️ Project Structure

```
/app-licence-advisor
├─ app/
│ ├─ main.py # FastApi backend
│ ├─ rag.py # Query engine setup
│ ├─ ui.py # Gradio frontend that calls main.py API
│ ├─ guards.py # Disclaimer / response guards
├─ ingest/
│ ├─ build_index.py # Ingestion + index building
│ └─ loaders.py # Custom document loaders
├─ data/
│ ├─ poc/ # Sample docs (MIT, Apache-2.0, etc.)
│ └─ processed/ # Cleaned/normalized artifacts
├─ storage/ # Persisted vector index
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 🚀 Getting Started

Create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate

```
### Run with Docker Compose

Build the images and start the services:

```bash
# Build without cache
docker compose build --no-cache 

# Start the stack
docker compose up

```
### Gradio UI 
```
http://localhost:7860/
```
## Example Queries
```
“What permissions does the MIT license grant?”

“What obligations do I have under Apache-2.0?”

“Is GPL-3.0 compatible with MIT?”

“Show me SPDX metadata for GPL-3.0.”
```

## 🛠️ Tech Stack
```
Python 3.10+

LlamaIndex – ingestion, indices, RAG pipeline

OpenAI GPT-4o-mini (default LLM)

HuggingFace embeddings – semantic retrieval

Gradio – interactive chatbot UI
```

## 📈 Roadmap
```
 Add more licenses & OSI docs

 Hybrid retrieval (BM25 + embeddings)

 Reranker for better answer quality

 Evaluation harness (faithfulness, relevancy)

 Knowledge-graph index for license compatibility questions

 Dockerfile for one-click deployment
 ```

 ⚠️ Disclaimer

This project is an educational demo of Retrieval-Augmented Generation.
It is not legal advice. For real licensing questions, consult a qualified attorney.