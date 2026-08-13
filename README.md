# MedGemma Inference Server

A FastAPI backend for analyzing medical images and PDFs (chest X‑rays, radiology reports, medical text) with vision/text LLMs, and for chatting about the results via hybrid retrieval‑augmented generation (RAG). Inference runs on models served locally through [Ollama](https://ollama.com); analysis and RAG generation are executed by a separate worker process connected to the API only through a Redis queue.

> ⚠️ This is a research/engineering project for processing medical documents with LLMs. It is **not** a certified medical device and must not be used for clinical decision‑making. Prompts explicitly instruct the models to describe findings, not diagnose.

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Getting started](#getting-started)
- [Docker](#docker)
- [Data model](#data-model)
- [Document processing pipeline](#document-processing-pipeline)
- [Task queue and worker](#task-queue-and-worker)
- [RAG and chat system](#rag-and-chat-system)
- [API reference](#api-reference)
- [Audit logging](#audit-logging)
- [Logging and observability](#logging-and-observability)
- [Database migrations](#database-migrations)
- [Developer tools](#developer-tools)
- [Testing](#testing)
- [Glossary](#glossary)

---

## Features

- **Document upload** — image (`image/jpeg`, `image/png`) and PDF (`application/pdf`) uploads, up to 50 MB, content-validated (Pillow for images, PyMuPDF for PDFs) and checksummed (SHA‑256).
- **Vision analysis** — a vision model (`ANALYSIS_MODEL`) produces a structured, section‑by‑section descriptive findings report from an image; PDFs are rendered page‑by‑page and OCR'd as a fallback for scanned pages.
- **Structured extraction** — a text model (`TEXT_MODEL`) turns the raw report into a concise summary and a controlled vocabulary of entities (regex extraction is used as a fallback if the model doesn't return valid JSON).
- **Hybrid retrieval** — each chunk is embedded (`EMBED_MODEL`, stored via `pgvector`) *and* indexed for full‑text search (Postgres `tsvector`/`websearch_to_tsquery`); query time merges semantic and lexical results with Reciprocal Rank Fusion (RRF).
- **Chat with RAG** — chat sessions can have documents attached; queries are answered using a context bundle built from the current document, retrieved chunks, and prior document summaries.
- **Asynchronous pipeline** — analysis and RAG generation run in a separate worker process that consumes tasks from Redis, so the API stays responsive and either process can be restarted independently.
- **Audit trail** — uploads, analyses, and deletions are recorded as `AuditEvent` rows.
- **Structured logging** — JSON logs (`structlog`) with a request‑scoped `request_id`, queryable via a built‑in `/logs` endpoint and viewer.

## Architecture

```mermaid
flowchart LR
    Client --> API[FastAPI API]
    API -->|enqueue task| Redis[(Redis queue)]
    Redis -->|BRPOP, blocking| Worker[Worker process]

    subgraph Server [Inference Server]
        API
        Worker
        PG[(PostgreSQL + pgvector)]
    end

    Worker -->|chat / generate / embed| Ollama[Ollama]
    API --> PG
    Worker --> PG
```

The API validates requests, reads/writes Postgres, and enqueues work — it never calls the inference pipelines directly. The worker consumes tasks from Redis, runs `ImageAnalysisService.run` / `PDFAnalysisService.run` / `RAGService.run`, and writes results back to Postgres. Redis is purely the handoff mechanism between the two processes; **Postgres remains the single source of truth** for application state (`Analysis.status`, chat messages, etc.) — the Redis-side status tracked by the queue (`queued`/`processing`/`completed`/`failed`) only describes the queue entry itself.

This design replaced an earlier implementation based on FastAPI's `BackgroundTasks`, which lost all in‑flight work on a process restart and had no failure visibility beyond logs.

## Repository layout

```
app/
├── main.py                  # FastAPI app: CORS, routers, /logs endpoints, static UI
├── worker.py                 # standalone worker process entry point (python -m app.worker)
├── logger.py, logging_config.py   # structlog setup
├── middleware.py              # per-request logging context (request_id, path, method)
├── core/
│   ├── config.py               # pydantic-settings Settings (env vars)
│   ├── database.py              # async engine/session factory, get_db()
│   ├── queue.py                  # singleton RedisQueue + get_queue() dependency
│   └── audit.py                   # audit(event_type, ...) helper -> AuditEvent
├── features/
│   ├── documents/                  # upload, list, get, delete documents; trigger/list/delete analyses
│   ├── analysis/                    # get/poll/delete/validate a single analysis
│   ├── chats/                        # chat session CRUD, document attach/detach
│   └── chat_messages/                 # message CRUD, RAG trigger (/query)
├── inference/
│   ├── llm.py                          # Ollama client (chat / generate / embed)
│   ├── prompts.py                       # system prompts + entity vocabulary/JSON schema
│   ├── types.py                          # ChestXrayEntity enum, ImageInput type
│   ├── image_analysis.py                  # ImageAnalysisService (analyse -> extract -> embed)
│   ├── pdf_analysis.py                     # PDFAnalysisService (per-page analyse -> extract -> embed)
│   ├── context.py                           # ContextEngine: hybrid retrieval + RRF merge
│   └── rag.py                                # RAGService: embed query -> context -> generate
├── models/                    # SQLModel ORM: Document, Analysis, Chunk, ChatSession,
│                               # ChatMessage, ChatDocument, AuditEvent, enums
├── schemas/                    # Pydantic request/response models
├── queue/
│   ├── tasks.py                 # TaskEnvelope, TaskType, AnalysisTaskPayload, RAGTaskPayload
│   ├── redis_queue.py             # RedisQueue: add / get / check / remove / mark_*
│   └── dispatcher.py                # maps TaskType -> the right service's run()
├── utils/
│   ├── pdf_ocr.py                     # PyMuPDF text extraction, page rendering, pytesseract OCR fallback
│   └── uploads.py
└── static/                     # index.html (debug UI) and logs.html (log viewer), served at "/" and "/logs/ui"

alembic/                 # migrations (19 revisions as of this writing)
tests/                    # pytest suite (documents, analysis, chats, messages)
analyze.py                 # CLI: run analysis against a document outside the API
evaluate_benchmark.py        # batch benchmarking of the analysis/RAG pipeline
index.html                    # duplicate of app/static/index.html at repo root
```

## Prerequisites

- Python 3.11+ (Docker image uses 3.13)
- PostgreSQL 14+ with the `pgvector` extension enabled
- Redis (task queue between the API and worker)
- [Ollama](https://ollama.com), reachable at `OLLAMA_URL`, with the vision/text/embedding models pulled
- Docker & Docker Compose (optional, for the containerized setup)
- `tesseract-ocr` on the host/image if you need OCR fallback for scanned PDF pages (used via `pytesseract`)

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

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | yes | asyncpg connection string for Postgres |
| `OLLAMA_URL` | yes | base URL of the Ollama server |
| `ANALYSIS_MODEL` | yes | vision model used to produce the findings report from an image/PDF page. Its `name:version` is split and stored as `Analysis.model_name` / `Analysis.model_version` for lineage. |
| `TEXT_MODEL` | yes | text model used for summarization, entity extraction, and RAG answer generation |
| `EMBED_MODEL` | yes | embedding model used for chunk and query vectors |
| `EMBEDDING_DIM` | yes | dimensionality of the embedding vectors; must match `EMBED_MODEL`'s output — it's baked into the `pgvector` column type at migration time |
| `REDIS_URL` | yes* | Redis connection string for the task queue (*the `Settings` model requires it; a `compose.yml`-provided default of `redis://localhost:6379/0` is common in practice) |

`Settings` (`app/core/config.py`) is `pydantic_settings.BaseSettings` — the app fails fast at startup if any required variable is missing.

## Getting started

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

4. **Enable `pgvector` and run migrations**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

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

   By default this listens on `http://localhost:8000` (the Docker image instead binds `0.0.0.0:14323` — see [Docker](#docker)). Swagger UI is at `/docs`; a built‑in debugging UI is served at `/`; JSON logs are queryable at `/logs` and viewable at `/logs/ui`.

7. **Start the worker** (separate terminal — analysis and RAG inference run here, not in the API process)

   ```bash
   python -m app.worker
   ```

   The worker connects to Redis, blocks on `BRPOP` for the next task, and processes one task at a time by design (the pipeline shares local Ollama model resources). It handles `SIGTERM`/`SIGINT` gracefully — it finishes the current task before exiting.

## Docker

`Dockerfile` (Python 3.13‑slim) and `compose.yml` define three services:

| Service | What it runs | Notes |
| --- | --- | --- |
| `inference` | `uvicorn app.main:app --host 0.0.0.0 --port 14323` | exposes `14323:14323`; mounts `./uploads` |
| `worker` | `python -m app.worker` | same image, overridden command; mounts `./uploads` |
| `redis` | `redis:7-alpine` | exposes `6379`; persists to a `redis-data` volume |

```bash
docker compose build
docker compose up -d
```

Both application containers read `.env` via `env_file`, and both map `host.docker.internal` to the host gateway so they can reach a Postgres/Ollama instance running outside the compose file. The `inference` and `worker` containers are independent: `docker compose restart inference` does not interrupt in‑flight worker tasks, and vice versa — because they only communicate through Redis and Postgres, not directly.

## Data model

All tables are defined as SQLModel ORM classes (`app/models/`) and versioned via Alembic. Timestamps use `datetime.now` server-side defaults unless noted.

### `Document` (`documents`)

| Field | Type | Notes |
| --- | --- | --- |
| `document_id` | UUID, PK | |
| `original_filename` | str | as uploaded |
| `stored_filename` | str, unique | `{document_id}{ext}` on disk |
| `file_path` | str | under `./uploads/` |
| `content_type` | str | `image/jpeg`, `image/png`, or `application/pdf` |
| `checksum` | str? (64 chars, indexed) | SHA‑256 of the uploaded bytes; not yet enforced unique |
| `file_size` | int | bytes |
| `created_at` / `updated_at` | datetime | |

A `Document` has no status field — analysis progress lives entirely on `Analysis`.

### `Analysis` (`analyses`)

| Field | Type | Notes |
| --- | --- | --- |
| `analysis_id` | UUID, PK | |
| `document_id` | UUID, FK → `documents`, `ON DELETE CASCADE` | |
| `model_name` / `model_version` | str | parsed from `ANALYSIS_MODEL` at creation time, for lineage |
| `raw_output` | text? | full report text from the vision model |
| `summary` | text? | extracted concise summary |
| `entities` | *(not stored here)* | derived at query time as the union of all associated `Chunk.entities` |
| `status` | `AnalysisStatus` enum | `READY → ANALYZING → CHUNKING → EMBEDDING → COMPLETE`, or `FAILED` / `DELETED` |
| `error_message` | text? | populated on failure paths (retry/DLQ support) |
| `retry_count` | int, default 0 | |
| `validated` | bool, default false | set via `POST /analysis/{id}/validate` |
| `started_at` / `completed_at` | datetime? | latency tracking |
| `prompt_template` / `extract_prompt_template` | str / text? | the exact prompts used, stored for reproducibility |
| `created_at` / `updated_at` | datetime | |

Deleting a `Document` cascades to its analyses at the database level. `DELETE /analysis/{id}` is a **soft delete** — it sets `status = DELETED` rather than removing the row.

### `Chunk` (`chunks`)

| Field | Type | Notes |
| --- | --- | --- |
| `chunk_id` | UUID, PK | |
| `document_id`, `analysis_id` | UUID, FK | |
| `page_number` | int (≥1) | `0` for single‑page image analyses; the actual PDF page number otherwise |
| `chunk_index` | int (≥0) | |
| `chunk_type` | `ChunkType` enum | `TEXT`, `TABLE`, `IMAGE`, `CAPTION`, `METADATA` |
| `chunk_content` | text | the summary text used for embedding and retrieval |
| `search_vector` | generated column | `to_tsvector('english', chunk_content)`, computed and persisted by Postgres for full‑text search |
| `entities` | JSONB list[str] | controlled‑vocabulary findings |
| `notes` | JSONB list[str] | e.g. projection/technical notes, also entity‑typed on read |
| `source_locations` | JSONB dict | reserved for page/coordinate metadata |
| `embedding` | `pgvector` (`EMBEDDING_DIM`), nullable | set during the embed step |
| `embedding_model` | str | |
| `created_at` / `updated_at` | datetime | |

### Chat models (`chat.py`, `chat_document.py`)

| Model | Table | Key fields |
| --- | --- | --- |
| `ChatSession` | `chat_sessions` | `chat_id`, `title?`; deleting cascades to messages and attachments |
| `ChatMessage` | `chat_messages` | `message_id`, `chat_id`, `role` (`MessageRole`: `SYSTEM`/`USER`/`ASSISTANT`), `content`, `message_metadata` (JSON) |
| `ChatDocument` | `chat_documents` | composite PK (`chat_id`, `document_id`), `message_id?` (the system message created on attach), `attached_at` — junction table |

### `AuditEvent` (`audit_events`)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID, PK | |
| `document_id` / `analysis_id` | UUID?, FK, indexed | either may be null |
| `event_type` | str (indexed) | e.g. `document:upload`, `document:delete`, `document:analysis` |
| `status` | str | e.g. `success` |
| `audit_metadata` | JSONB | free‑form event details |
| `created_at` | datetime (tz‑aware, indexed) | |

### Enums (`app/models/enums.py`)

```text
AnalysisStatus  READY, ANALYZING, CHUNKING, EMBEDDING, COMPLETE, FAILED, DELETED
ChunkType       TEXT, TABLE, IMAGE, CAPTION, METADATA
MessageRole     SYSTEM, USER, ASSISTANT
```

## Document processing pipeline

The pipeline is entirely driven by `Analysis.status`:

```mermaid
flowchart LR
    A[Upload] --> B[Document created]
    B --> C["POST /documents/{id}/analysis creates Analysis, status=ANALYZING"]
    C --> Q[Task enqueued on Redis]
    Q --> D["Worker: analyse() - vision model"]
    D --> E["extract() -> CHUNKING -> summary + entities, create Chunk(s)"]
    E --> F["embed() -> EMBEDDING -> embed each chunk"]
    F --> G[COMPLETE]
    D -.error.-> H[FAILED]
    E -.error.-> H
    F -.error.-> H
```

Two implementations exist, dispatched by `Document.content_type`:

- **`ImageAnalysisService`** (`app/inference/image_analysis.py`) — for `image/jpeg` / `image/png`. Produces a single `Chunk` (`page_number=0`, `chunk_index=1`).
- **`PDFAnalysisService`** (`app/inference/pdf_analysis.py`) — for `application/pdf`. Extracts text per page (PyMuPDF, falling back to `pytesseract` OCR for scanned/image‑only pages), analyses each page, and produces one `Chunk` per page (`page_number` = the actual page).

Both services share the same three‑step shape:

1. **`analyse()`** — sends the image (or rendered PDF page) to the vision model (`ANALYSIS_MODEL`) with `IMG_PROCESS_PROMPT` (images) or a page‑analysis prompt (PDFs). The prompt asks for a strict, section‑by‑section **descriptive findings report** — technical quality, cardiomediastinal, lungs, pleura, diaphragm, bones/soft tissue, devices — using a fixed vocabulary and explicitly forbidding diagnostic language (`suggests`, `likely`, `may represent`, etc.). The raw response is cleaned of an internal `<unused95>` model marker and stored in `raw_output`; the prompt itself is stored on `Analysis.prompt_template` for reproducibility.
2. **`extract()`** (status → `CHUNKING`) — sends `raw_output` to the text model (`TEXT_MODEL`) with `IMG_EXTRACT_PROMPT` / `PDF_EXTRACT_PROMPT`, which demand a strict JSON object (`{summary, entities, notes/technical_notes, comparison}`) built from a **closed entity vocabulary** (`ChestXrayEntity`: `No Finding`, `Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, `Pneumothorax`, `Consolidation`, `Edema`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`). If the model doesn't return parseable JSON, the service falls back to a regex‑based summary/entity extraction over `raw_output`. A `Chunk` is created per page/image with the extracted `summary`, `entities`, and `notes`.
3. **`embed()`** (status → `EMBEDDING`) — embeds each chunk's `chunk_content` via `EMBED_MODEL` and stores the vector; status becomes `COMPLETE`. Any exception at any step sets `status = FAILED` (and, in `ImageAnalysisService.run`, clears `summary`/`raw_output`).

`POST /documents/{id}/analysis` creates the `Analysis` row (`status=ANALYZING` immediately, not `READY` — the worker task is enqueued in the same call) and rejects a second concurrent request for the same document with `409` while a prior analysis is `ANALYZING`/`CHUNKING`/`EMBEDDING`.

## Task queue and worker

Long‑running inference is decoupled from the API process via a Redis‑backed queue consumed by a separate worker.

```mermaid
flowchart LR
    API[FastAPI API] -->|add| Q[("Redis: queue:pending")]
    Q -->|BRPOP, blocking| W[Worker Process]
    W --> S["dispatch() -> ImageAnalysisService.run / PDFAnalysisService.run / RAGService.run"]
    S --> DB[(PostgreSQL)]
```

- **`TaskEnvelope`** (`app/queue/tasks.py`) — `{task_id, task_type, payload, enqueued_at}`, JSON‑serialized for Redis. `task_type` is one of `analysis:image`, `analysis:pdf`, `rag` (`TaskType` enum). Convenience constructors: `TaskEnvelope.for_analysis(analysis_id, document_type)` (picks `analysis:pdf` vs `analysis:image` from the content type) and `TaskEnvelope.for_rag(chat_id, query, current_document_id=None)`.
- **`RedisQueue`** (`app/queue/redis_queue.py`) — a minimal deque abstraction, *not* a priority queue or full task manager. Redis key layout: `{prefix}:pending` (list of task IDs), `{prefix}:payload:{task_id}`, `{prefix}:status:{task_id}`.
  - `add(task)` — pushes the task ID onto `pending`, stores the payload, sets status `queued`.
  - `get(timeout=5)` — blocks via `BRPOP` up to `timeout` seconds; returns `None` on timeout so the worker loop can check for a shutdown request; marks the popped task `processing`.
  - `check(task_id)` — returns `queued`/`processing`/`completed`/`failed`/`None`. This is the **queue entry's** status only — `Analysis.status` in Postgres remains authoritative for application state.
  - `remove(task_id)` — cancels a task still sitting in `pending` (no-op if already picked up).
  - `mark_completed(task_id)` / `mark_failed(task_id)` — called by the worker after `dispatch()` returns or raises; terminal statuses expire from Redis after 1 hour (`_TERMINAL_STATUS_TTL_SECONDS`), and the payload is deleted immediately.
- **`dispatch(task)`** (`app/queue/dispatcher.py`) — maps `task_type` to the corresponding service's `run()`. Contains no pipeline logic of its own.
- **`app/worker.py`** — run with `python -m app.worker`. A single `Worker` instance loops on `queue.get()`, calls `dispatch()`, and marks completion/failure; registers `SIGTERM`/`SIGINT` handlers (via `asyncio`'s `add_signal_handler`, with a fallback for platforms like Windows where that's unavailable) that request a graceful stop — the current task finishes before the loop exits.
- **`app/core/queue.py`** — a module‑level singleton `RedisQueue` plus a `get_queue()` FastAPI dependency, mirroring `get_db()`.

**Concurrency**: one worker process at a time is the default — inference shares local Ollama model resources. `BRPOP` is atomic per item, so multiple worker processes can safely consume the same queue if throughput later requires it, but that should be measured before introducing.

**Duplicate submissions**: the queue does no deduplication itself; `DocumentService.analyze_document` already rejects a concurrent request (`409`) at the database level while an analysis is in progress.

## RAG and chat system

### Attaching documents

A chat session can have zero or more documents attached (`POST /chats/{id}/document/{document_id}`). Attaching or detaching inserts a `SYSTEM`-role `ChatMessage` (`"ADDED DOCUMENT {filename}"` / `"REMOVED DOCUMENT {filename}"`) and, on attach, links it via `ChatDocument.message_id`.

### Query flow

1. `POST /chats/{id}/query` saves the user's `ChatMessage`, enqueues a `rag` task (`TaskEnvelope.for_rag`), and returns the saved message immediately (`201`) — it does **not** wait for the answer.
2. The worker's `RAGService.run(chat_id, query, current_document_id=None)`:
   a. **Embeds the query** via `EMBED_MODEL`.
   b. **Builds a `ContextBundle`** (`ContextEngine.build_context`) from three sources:
      - **Similar chunks** — hybrid retrieval, described below.
      - **Previous documents** — every document attached to the chat, each with its *latest* analysis summary and the union of its chunks' `notes` (parsed back into `ChestXrayEntity`, invalid values silently skipped).
      - **Current document** — the raw findings (`Analysis.raw_output`) of `current_document_id`, if supplied.
   c. **Augments** the query into a prompt (`QUERY_PROMPT`) that explicitly restricts the model to the retrieved context and forbids outside medical knowledge.
   d. **Generates** the answer via `TEXT_MODEL` with a `GENERATE_PROMPT` system message, and saves it as a new `ChatMessage` (`role=ASSISTANT`).
3. Clients poll `GET /chats/{id}/messages` to see the assistant's reply appear.

### Hybrid retrieval (semantic + lexical, RRF merge)

`ContextEngine._get_similar_chunks` and `_get_lexical_chunks` run **in parallel query paths**, then `_merge_chunks` combines them with Reciprocal Rank Fusion:

- **Semantic** — cosine distance between the query embedding and `chunks.embedding` via `pgvector`'s `<->` operator (`Chunk.embedding.cosine_distance(...)`), ordered ascending, top‑k (default 5). Similarity reported as `1 - distance`.
- **Lexical** — Postgres full‑text search: `websearch_to_tsquery('english', query)` matched against the generated `Chunk.search_vector` column, ranked by `ts_rank`, top‑k.
- **Merge (RRF)** — for each list, each chunk's score is incremented by `1 / (60 + rank)` (rank starting at 1); scores from both lists are summed per `chunk_id`, and the top‑k highest‑scoring chunks (by combined score) are returned. This means a chunk that ranks decently on *both* signals will usually outrank one that ranks #1 on only one signal.

## API reference

All routes are mounted under `/api/v1` and grouped `/documents`, `/analysis`, and `/chats`. Interactive, always‑current schemas are at `/docs` (Swagger UI) once the server is running — the tables below mirror those routes as implemented in `app/features/*/router.py`.

Analysis and RAG queries are asynchronous: the endpoint that triggers them returns immediately, and clients poll (`GET /api/v1/analysis/{id}/status`, `GET /api/v1/chats/{id}/messages`) until the result appears.

### Documents & analysis

| Method | Path | Description | Status codes |
| --- | --- | --- | --- |
| POST | `/api/v1/documents/upload` | Upload a document (`multipart/form-data`, `image/jpeg`\|`image/png`\|`application/pdf`, ≤ 50 MB) | 201, 400, 415, 500 |
| GET | `/api/v1/documents` | List documents — `?page_no=1&page_size=10` | 200, 500 |
| GET | `/api/v1/documents/{document_id}` | Get document metadata | 200, 404, 500 |
| DELETE | `/api/v1/documents/{document_id}` | Delete a document (DB row + file on disk) | 204, 404, 500 |
| POST | `/api/v1/documents/{document_id}/analysis` | Start analysis (creates `Analysis`, enqueues a worker task) | 202, 404, 409, 500 |
| GET | `/api/v1/documents/{document_id}/analyses` | List analyses for a document — `?page_no=1&page_size=10` | 200, 500 |
| DELETE | `/api/v1/documents/{document_id}/analyses` | Delete all analyses for a document | 204, 404, 500 |
| GET | `/api/v1/analysis/{analysis_id}` | Get a full analysis, entities merged from chunks | 200, 404, 500 |
| GET | `/api/v1/analysis/{analysis_id}/status` | Poll analysis status | 200, 404, 500 |
| DELETE | `/api/v1/analysis/{analysis_id}` | Soft-delete an analysis (`status → DELETED`) | 204, 404, 500 |
| POST | `/api/v1/analysis/{analysis_id}/validate` | Mark an analysis as human‑validated (`validated → true`) | 200, 404, 500 |

`409` on `POST /documents/{id}/analysis` means an analysis is already `ANALYZING`/`CHUNKING`/`EMBEDDING` for that document.

**Response schemas**

```text
DocumentRead           document_id, original_filename, content_type, file_size, created_at
AnalysisCreateResponse  analysis_id, document_id, status, created_at
AnalysisListItem         analysis_id, document_id, summary, status, created_at
AnalysisStatusResponse    analysis_id, status, updated_at
AnalysisRead               analysis_id, document_id, raw_output, summary, status,
                            entities (deduplicated union of chunk entities),
                            analysis_metadata { model_name, model_version, created_at }
ErrorResponse                error (machine-readable code), message, details?
```

### Chats & messages

| Method | Path | Description | Status codes |
| --- | --- | --- | --- |
| POST | `/api/v1/chats` | Create a chat — `{"title": "..."}` | 201, 400, 500 |
| GET | `/api/v1/chats` | List chats — `?page_no=1&page_size=10` | 200, 500 |
| GET | `/api/v1/chats/{chat_id}` | Get a chat | 200, 404, 500 |
| PATCH | `/api/v1/chats/{chat_id}` | Update chat title | 200, 400, 404, 500 |
| DELETE | `/api/v1/chats/{chat_id}` | Delete a chat (cascades to messages/attachments) | 204, 404, 500 |
| POST | `/api/v1/chats/{chat_id}/document/{document_id}` | Attach a document | 202, 404, 409, 500 |
| GET | `/api/v1/chats/{chat_id}/documents` | List attached documents (with latest summary) | 200, 404, 500 |
| DELETE | `/api/v1/chats/{chat_id}/document/{document_id}` | Detach a document | 204, 404, 500 |
| POST | `/api/v1/chats/{chat_id}/query` | Submit a user message; enqueues a RAG task | 201, 400, 404, 500 |
| GET | `/api/v1/chats/{chat_id}/messages` | Get chat history, chronological | 200, 404, 500 |
| GET | `/api/v1/chats/{chat_id}/messages/{message_id}` | Get a specific message | 200, 404, 500 |
| PATCH | `/api/v1/chats/{chat_id}/messages/{message_id}` | Update a message's content/metadata | 200, 400, 404, 500 |
| DELETE | `/api/v1/chats/{chat_id}/messages/{message_id}` | Delete a message | 204, 404, 500 |

**Response schemas**

```text
ChatCreate / ChatUpdate     title?
ChatRead                     chat_id, title?
ChatDocumentRead               chat_id, title?, document_id, file_path, file_size,
                                content_type, summary?, attached_at
ChatDocumentSummary              chat_id, document_id, analysis_id, summary?, status,
                                  attached_at, chunks (list[UUID]), notes (list[entity])
ChatMessageCreate                  role (SYSTEM | USER | ASSISTANT), content, message_metadata?
ChatMessageUpdate                    content?, message_metadata?
ChatMessageRead                        message_id, chat_id, role, content, message_metadata,
                                        created_at, updated_at
```

## Audit logging

`app.core.audit.audit(db, event_type, *, document_id=None, analysis_id=None, status="success", audit_metadata=None)` appends an `AuditEvent` row (not yet committed by the helper itself — it's added to the session and committed alongside the calling operation). Currently emitted from `DocumentService`:

- `document:upload` — filename, content type, file size
- `document:delete` — filename, content type, file size
- `document:analysis` — filename, content type, file size (on triggering an analysis)

These are queryable directly against the `audit_events` table; there is no dedicated API endpoint for them yet.

## Logging and observability

- **Structured logging** — `structlog`, configured in `app/logging_config.py` (`configure_structlog(truncate_at=1000)`), writes JSON lines. `LoggingMiddleware` binds `request_id`, `path`, and `method` as context vars for the duration of each HTTP request (skipped for `/logs*` paths to avoid polluting the dashboard's own requests).
- **`GET /logs`** *(hidden from the OpenAPI schema)* — reads the JSON log file (`logs/app.jsonl`) and returns the most recent entries, newest first. Query params: `level`, `event` (substring match), `limit` (1–5000, default 500).
- **`GET /logs/ui`** — serves `app/static/logs.html`, a small viewer for the above.
- **`GET /`** — serves `app/static/index.html`, a standalone debugging UI for uploading, analyzing, and chatting through the REST API directly from the browser.

## Database migrations

Alembic manages schema changes (`alembic/versions/`, 19 revisions at the time of writing), including: initial schema + `vector` extension, making `chunks.embedding` nullable, adding `analyses.status`, adding the `documents.checksum` column, adding `analyses.validated`, adding `analyses.error_message`, adding `audit_events`, adding BM25/full‑text search support, and index/lookup optimizations for the latest‑analysis‑per‑document query.

```bash
alembic upgrade head                              # apply all pending migrations
alembic revision --autogenerate -m "description"  # generate a new migration
```

Always review autogenerated migration scripts before applying them.

## Developer tools

- **`analyze.py`** — CLI to run the analysis pipeline against a document outside the API/worker, useful for local debugging of prompts and extraction.
- **`evaluate_benchmark.py`** — batch benchmarking of the analysis/RAG pipeline (throughput, latency).
- **`index.html`** (served at `/`, also duplicated at the repo root) — a self‑contained debugging UI for uploading documents, triggering analysis, and chatting via the REST API.
- **`/logs`** and **`/logs/ui`** — see [Logging and observability](#logging-and-observability).

Typical local development runs three processes side by side: Postgres + Redis + Ollama, `uvicorn app.main:app --reload`, and `python -m app.worker`. Because the API and worker are decoupled by the Redis queue, queued work survives an API restart, and each layer can be debugged independently:

```text
API problem?         → check the API process/logs
Inference problem?    → check the worker process/logs
A task not running?    → check Redis (`RedisQueue.check(task_id)` / `queue:pending` length)
Analysis stuck?          → check PostgreSQL (`Analysis.status`, `error_message`)
```

## Testing

```bash
pytest
```

The suite (`tests/`) covers documents, analysis, chats, and messages, using `tests/conftest.py` fixtures and `tests/pytest.ini` configuration.

## Glossary

| Term | Meaning |
| --- | --- |
| **Analysis** | The result of processing a document (or one page of a PDF's worth of results, aggregated): raw output, summary, entities, status, lineage (model/prompt used) |
| **Document** | An uploaded image or PDF; itself has no status |
| **Chunk** | A unit of analyzed, embeddable, full‑text‑searchable text (one per image analysis, one per PDF page) |
| **ChatSession / ChatMessage / ChatDocument** | A conversation, its messages, and the documents attached to it |
| **RAG** | Retrieval‑augmented generation — here, hybrid semantic + lexical retrieval merged via RRF, used to ground the answer in retrieved chunks only |
| **RRF** | Reciprocal Rank Fusion — the `1/(60+rank)` scoring scheme used to merge semantic and lexical retrieval rankings |
| **TaskEnvelope** | The unit of work placed on the Redis queue (`task_id`, `task_type`, typed payload) |
| **Worker** | The standalone process (`app/worker.py`) that consumes queued tasks and executes the inference services |
| **ChestXrayEntity** | The closed vocabulary of radiological findings the models are constrained to use (`Cardiomegaly`, `Effusion`, `Nodule`, `No Finding`, etc.) |
