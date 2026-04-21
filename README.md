# AI FAQ Assistant

An intelligent FAQ assistant built with **LangChain + Azure OpenAI**, deployed as part of a statewide citizen portal at Deloitte (Georgia State). Deflects **20% of inbound support tickets** via RAG-powered self-service.

## Architecture
```
User Query → Embedding Model → Vector Search (FAISS/Pinecone)
           → Retrieved Context → GPT-4 (Azure OpenAI)
           → Answer + Sources → User
```

## Features
- 🤖 RAG pipeline: retrieves relevant FAQ chunks before answering
- 🔍 Semantic search over FAQ knowledge base using embeddings
- 💬 Conversational memory (multi-turn support)
- 📊 Confidence scoring — escalates to human agent when unsure
- 🚀 FastAPI REST endpoint for integration with citizen portals
- 📈 Built-in usage analytics and deflection tracking

## Tech Stack
`Python 3.11` `LangChain` `Azure OpenAI` `FAISS` `FastAPI` `Docker`

## Quick Start
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Azure OpenAI credentials
python main.py
```

## API
```
POST /ask        { "question": "How do I renew my benefits?" }
GET  /health     Health check
GET  /metrics    Deflection rate, avg confidence, query volume
```

## Results
| Metric | Value |
|---|---|
| Ticket deflection rate | 20% |
| Avg response latency | <1.2s |
| Answer confidence threshold | 0.75 |
