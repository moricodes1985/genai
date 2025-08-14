# AI Financial Research Assistant  
*A RAG-powered assistant for financial document search and live market data queries*  

---

## 📌 Overview  
The **AI Financial Research Assistant** combines **retrieval-augmented generation (RAG)** with **live market data integration** to help users quickly find, understand, and trust financial information.  

It can:  
- Search through **internal reports, financial statements, and sector notes**.  
- Pull **real-time stock prices** via external APIs.  
- Generate **concise, source-cited answers** in seconds.  

This tool is ideal for analysts, researchers, and decision-makers who need **fast, accurate, and explainable** financial insights.

---

## 🚀 Features  

### 🔍 Intelligent Search (RAG)  
- Retrieves relevant chunks from internal documents using **Sentence Transformers embeddings**.  
- Always cites document title and source file for transparency.  

### 📈 Live Market Data Integration  
- Detects when a question requires **live price data**.  
- Fetches from a mock or real API (e.g., Yahoo Finance, Alpha Vantage).  

### 🗣️ Conversational Interface  
- Accepts natural language questions (“What was Company A’s Q2 operating income growth?”).  
- Handles follow-up queries seamlessly.  

### ✅ Reliable & Transparent  
- Grounds answers in retrieved context or API data.  
- Displays sources and timestamps for verification.  
- Flags uncertainty when data is missing or incomplete.  

---

## 🛠 Architecture  

```plaintext
User Question
     ↓
Intent Detection → Tool Calls (e.g., Live Price API)
     ↓
Vector Store Retrieval (Semantic Search)
     ↓
Prompt Construction
     ↓
LLM Response with Citations
```
---

## Core Technologies:
- FastAPI – REST API backend.

- Sentence Transformers – Embedding-based semantic search.

- NumPy / Vector Store – Efficient similarity search.

- OpenAI GPT Models – Natural language reasoning and answer generation.

---
## 📦 Installation & Usage
```
# 1. Clone the repository
git clone https://github.com/<your-username>/ai-financial-assistant.git
cd ai-financial-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
# Create a .env file:
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# 4. Run the API
uvicorn main:app --reload

```
### example query
```
curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{"question": "What was Company A’s Q2 operating income growth?"}'
```
### example response
```
{
  "answer": "Company A's operating income increased by 10% YoY (Company A Q2 Earnings | internal_reports/company_a_q2.txt).",
  "retrieved": [...],
  "tool_used": null
}
```
## 📅 Roadmap
- Replace mock price API with a real finance data source.

- Support document upload via API.

- Add UI chat interface (React + FastAPI backend).

- Implement multilingual support.

- Deploy on Render/Fly.io for public demo.




