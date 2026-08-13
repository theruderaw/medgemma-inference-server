# MedGemma Inference Server

A FastAPI backend for analyzing medical images (chest X‑rays) with vision/text LLMs and chatting about the results via retrieval‑augmented generation (RAG). Inference runs on models served locally through [Ollama](https://ollama.com); long‑running work is offloaded to a background worker over a Redis queue.

## Features

- **Document upload** — clients upload image documents for analysis.
- **Vision analysis** — a vision model (`ANALYSIS_MODEL`) turns an image into a textual findings report.
- **Structured extraction** — a text model (`TEXT_MODEL`) derives a concise summary and a list of entities from the raw report.
- **Embedding & retrieval** — each analysis summary is embedded (`EMBED_MODEL`) and stored in Postgres via `pgvector` for semantic similarity search.
- **Chat with RAG** — chat sessions can have documents attached; queries are answered using a context bundle built from the current document, similar chunks, and prior document summaries.
- **Async pipeline** — analysis and RAG generation run in a separate worker process that consumes tasks from Redis, so the API stays responsive and can be restarted independently of in‑flight work.

## Architecture

```mermaid
flowchart LR
    A[Client] --> B[FastAPI API]
    B -->|enqueue task| R[(Redis Queue)]
    R -->|consume| W[Worker Process]

    subgraph Server [Inference Server]
        B
        W
        C[(PostgreSQL + pgvector)]
    end

    W -->|vision / text / embeddings| O[Ollama]
    B --> C
    W --> C
```

The API validates requests, reads/writes Postgres, and enqueues work — it never calls the inference pipelines directly. The worker consumes tasks from Redis, runs the image analysis / RAG services, and writes results back to Postgres. Redis is purely the handoff mechanism between the two processes; Postgres remains the source of truth.

**Layout**

```
app/
├── main.py             # FastAPI app, CORS, routers, log endpoints
├── worker.py            # standalone worker process (consumes the Redis queue)
├── core/                 # settings, database session, queue client, audit
├── features/
│   ├── documents/        # upload, list, delete documents
│   ├── analysis/          # trigger + poll vision analysis
│   ├── chats/              # chat session + document attachment
│   └── chat_messages/       # messages, RAG trigger
├── inference/              # Ollama client, prompts, RAG orchestration
├── models/                  # SQLModel ORM models + enums
├── schemas/                  # Pydantic request/response models
└── queue/                     # task envelope, Redis queue, dispatcher
```

## Prerequisites

- Python 3.11+
- PostgreSQL 14+ with the `pgvector` extension
- Redis (task queue between the API and worker)
- [Ollama](https://ollama.com) running locally, or reachable, with the models below pulled
- Docker & Docker Compose (optional, for containerized setup)

## Configuration

Create a `.env` file in the project root:

```bash
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
OLLAMA_URL=http://localhost:11434
ANALYSIS_MODEL=medgemma1.5:4b
TEXT_MODEL=qwen2.5:3b
EMBED_MODEL=nomic-embed-text:latest
EMBEDDING_DIM=768
REDIS_URL=redis://localhost:6379/0
```

| Variable | Description |
| --- | --- |
| `DATABASE_URL` | asyncpg connection string for Postgres |
| `OLLAMA_URL` | base URL of the Ollama server |
| `ANALYSIS_MODEL` | vision model used to produce the findings report from an image |
| `TEXT_MODEL` | text model used for summarization, entity extraction, and RAG answer generation |
| `EMBED_MODEL` | embedding model used for chunk and query vectors |
| `EMBEDDING_DIM` | dimensionality of the embedding vectors, must match `EMBED_MODEL`'s output |
| `REDIS_URL` | Redis connection string for the task queue |

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/theruderaw/medgemma-inference-server.git
   cd medgemma-inference-server
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Pull the Ollama models**

   ```bash
   ollama pull medgemma1.5:4b
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

4. **Run database migrations**

   ```bash
   alembic upgrade head
   ```

5. **Start Redis** (if not already running)

   ```bash
   redis-server
   ```

6. **Start the API**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API listens on `http://localhost:8000` by default (`http://localhost:14323` in the Docker setup — see below). Interactive docs are at `/docs`; a simple built‑in test UI is served at `/`.

7. **Start the worker** (separate terminal — analysis and RAG inference run here, not in the API process)

   ```bash
   python -m app.worker
   ```

### Docker

`Dockerfile` and `compose.yml` define three services — `inference` (API, port `14323`), `worker`, and `redis`:

```bash
docker compose build
docker compose up -d
```

The API and worker containers are independent: `docker compose restart inference` does not interrupt in‑flight worker tasks, and vice versa. Postgres/Ollama are expected to run outside this compose file (`host.docker.internal` is mapped for reaching services on the host).

## API Overview

All endpoints are grouped under `/documents`, `/analysis`, and `/chats`. Full request/response schemas are available at `/docs` once the server is running.

**Documents & analysis**

| Method | Path | Description |
| --- | --- | --- |
| POST | `/documents/upload` | Upload an image document |
| GET | `/documents` | List documents (paginated) |
| GET | `/documents/{document_id}` | Get document metadata |
| DELETE | `/documents/{document_id}` | Delete a document |
| POST | `/documents/{document_id}/analysis` | Start analysis (enqueued, async) |
| GET | `/documents/{document_id}/analyses` | List analyses for a document |
| GET | `/analysis/{analysis_id}` | Get a full analysis, including chunks |
| GET | `/analysis/{analysis_id}/status` | Poll analysis status |
| DELETE | `/analysis/{analysis_id}` | Delete an analysis |

**Chats & messages**

| Method | Path | Description |
| --- | --- | --- |
| POST | `/chats` | Create a chat session |
| GET | `/chats` | List chats (paginated) |
| GET/PATCH/DELETE | `/chats/{chat_id}` | Get, rename, or delete a chat |
| POST/DELETE | `/chats/{chat_id}/document/{document_id}` | Attach / detach a document |
| GET | `/chats/{chat_id}/documents` | List documents attached to a chat |
| POST | `/chats/{chat_id}/query` | Submit a user query (triggers RAG, enqueued) |
| GET | `/chats/{chat_id}/messages` | Get chat history |
| GET/PATCH/DELETE | `/chats/{chat_id}/messages/{message_id}` | Get, update, or delete a message |

Analysis and RAG queries are asynchronous: the endpoint that triggers them returns immediately, and clients poll (`/analysis/{id}/status`, `/chats/{id}/messages`) until the result appears.

## How it works

1. A document is uploaded and an analysis is triggered.
2. The API enqueues a task and the worker picks it up: the vision model (`ANALYSIS_MODEL`) generates a findings report from the image, the text model (`TEXT_MODEL`) extracts a summary and entities, and the summary is embedded (`EMBED_MODEL`) and stored as a chunk with a `pgvector` embedding.
3. A chat session can have one or more documents attached.
4. When a user sends a query, it's embedded and used to retrieve similar chunks (across the current and prior documents) via `pgvector` similarity search.
5. The retrieved context, current document findings, and the query are assembled into a prompt and sent to `TEXT_MODEL`, which generates the assistant's reply — saved as a new chat message for the client to poll for.

## Developer Tools

- **`analyze.py`** — CLI to run analysis against a document outside the API.
- **`evaluate_benchmark.py`** — batch benchmarking of the analysis/RAG pipeline.
- **`index.html`** — a standalone debugging UI (served at `/`) for uploading, analyzing, and chatting through the REST API.
- **`/logs`** and **`/logs/ui`** — structured JSON application logs and a small viewer, backed by `logs/app.jsonl`.

Typical local development runs three processes side by side: Postgres + Redis + Ollama, `uvicorn app.main:app --reload`, and `python -m app.worker`. Because the two application processes are decoupled by the Redis queue, queued work survives an API restart, and each can be debugged independently (API logs vs. worker logs vs. `Analysis.status` in Postgres).

## Tests

```bash
pytest
```

## Glossary

| Term | Meaning |
| --- | --- |
| Analysis | Result of processing a document: raw output, summary, entities, status |
| Document | An uploaded image |
| Chunk | A piece of analyzed text with an embedding, used for retrieval |
| ChatSession / ChatMessage / ChatDocument | A conversation, its messages, and its attached documents |
| RAG | Retrieval‑augmented generation — similarity search used to enrich a prompt before generation |
| TaskEnvelope | The unit of work placed on the Redis queue (`task_id`, `task_type`, payload) |
| Worker | The standalone process (`app/worker.py`) that consumes queued tasks and runs inference |

---

⚠️ This is a research/engineering project for processing medical images with LLMs — it is not a certified medical device and should not be used for clinical decision‑making.
