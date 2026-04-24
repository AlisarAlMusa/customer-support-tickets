# Customer Support Tickets

Customer-support AI application with a React frontend and a FastAPI backend.

It compares:
- RAG-based answering vs plain LLM answering
- ML priority prediction vs LLM priority prediction

## Repo Structure

```text
customer_support_frontend/
  frontend/                 # React + Vite app

customer_support_tickets/
  app/                      # FastAPI app
  rag/                      # loader, chunker, embedder, store
  models/                   # saved ML pipeline
```

## What It Does

- Ingests support-ticket data into ChromaDB
- Retrieves similar historical issues
- Generates:
  - a RAG answer
  - a plain LLM answer
- Predicts priority using:
  - a trained ML model
  - an LLM zero-shot classifier
- Returns evaluation signals designed for comparison in the UI

## Backend Setup

From the repo root:

```bash
cd customer_support_tickets
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in `customer_support_tickets/` and set at least:

```env
OPENAI_API_KEY=your_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Backend default URL:

```text
http://127.0.0.1:8000
```

## Frontend Setup

From the repo root:

```bash
cd customer_support_frontend/frontend
npm install
npm run dev
```

Frontend default URL:

```text
http://localhost:5173
```

## Main API Routes

- `POST /ingest`
- `POST /answer`
- `POST /predict`
- `GET /validate/chunker`
- `GET /validate/embedder`
- `POST /validate/retrieve`

## Logging

The backend writes logs under:

```text
customer_support_tickets/logs/
```

Files:
- `app.log`: general application logs
- `query_events.log`: structured logs for queries, retrievals, predictions, and generated outputs

To inspect query activity while the backend is running:

```bash
cd customer_support_tickets
tail -f logs/query_events.log
```

## Notes

- Chroma persistence is stored locally under `customer_support_tickets/rag/chroma_db/`
- Raw CSV data and local secrets are ignored by git
- The backend is already configured with CORS for the local frontend dev server

## Status

Current focus:
- reliable ingestion
- retrieval validation
- answer comparison
- prediction comparison

This repo is structured for experimentation first, with the core pipeline ready to evolve into a fuller production app.
