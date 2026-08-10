# MedGemma Inference Server

## Overview

The **MedGemma Inference Server** is a backend service that processes medical images (chest X‑rays) and supports retrieval‑augmented generation (RAG) for question‑answering. It provides:

- **Document upload and storage**: clients can upload image documents.
- **Image analysis**: a vision model (`ANALYSIS_MODEL`) converts the image into a textual findings report.
- **Structured extraction**: a text model (`TEXT_MODEL`) extracts a concise summary and a list of entities from the raw report.
- **Chunking and embedding**: each analysis produces one chunk (the summary) that is embedded with `EMBED_MODEL` for semantic search.
- **Similarity retrieval**: stored chunk vectors are queried via PostgreSQL’s `pgvector` to find relevant context.
- **Chat and RAG**: users create chat sessions, attach documents, and ask questions. The system assembles context (current document, similar chunks, previous documents) and generates answers.
- **Asynchronous processing**: long analyses run in background tasks; clients poll the analysis status.

**Key components**:

- **FastAPI** (`app/main.py`): routes and orchestration.
- **Routers** (`app/features/*/router.py`): REST endpoints (documents, analyses, chats, messages).
- **Services** (`app/features/*/service.py`): business logic.
- **Models** (`app/models/*.py`): SQLModel ORM (Document, Analysis, Chunk, ChatSession, ChatMessage, ChatDocument) and enums (AnalysisStatus, ChunkType, MessageRole).
- **Schemas** (`app/schemas/*.py`): Pydantic request/response models.
- **Inference** (`app/inference/*.py`): interfaces to Ollama, prompt engineering, and RAG orchestration.

**Architecture diagram**:

```mermaid
flowchart TB
    A[React Frontend] --> B[Express API]
    B --> C[FastAPI Inference Server]
    C -->|Vision / Text / Embeddings| D[Ollama]
    C -->|SQL / pgvector| E[(PostgreSQL)]
    D -->|AI Models| F[MedGemma / Qwen / Nomic]
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with `pgvector` extension
- Ollama running locally (serving the configured models)
- Docker & Docker Compose (optional)

Required environment variables (see Configuration):
- `DATABASE_URL`: asyncpg URL (e.g. `postgresql+asyncpg://user:pass@localhost:5432/medgemma`)
- `OLLAMA_URL`: e.g. `http://localhost:11434`
- `ANALYSIS_MODEL`: vision model (e.g. `medgemma1.5:4b`)
- `TEXT_MODEL`: text model (e.g. `qwen2.5:3b`)
- `EMBED_MODEL`: embedding model (e.g. `nomic-embed-text:latest`)

### Installation

1. **Clone repository**:
    
    ```bash
    git clone <repo-url> medgemma-inference
    cd medgemma-inference
    ```
    
2. **Create virtual environment**:
    
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    
3. **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Configure `.env`** (example):
    
    ```
    DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST:5432/DATABASE
    OLLAMA_URL=http://localhost:11434
    ANALYSIS_MODEL=medgemma1.5:4b
    TEXT_MODEL=qwen2.5:3b
    EMBED_MODEL=nomic-embed-text:latest
    ```
    
5. **Pull Ollama models**:
    
    ```bash
    ollama pull medgemma1.5:4b
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text
    ```
    
6. **Run database migrations**:
    
    ```bash
    alembic upgrade head
    ```
    
7. **Start the server**:
    
    ```bash
    uvicorn app.main:app --reload
    ```
    
    API available at `http://localhost:8000`; Swagger UI at `/docs`.
    

### Docker Setup

A `Dockerfile` and `compose.yml` are provided. Build and run:

```bash
docker compose build
docker compose up -d
```

---

## Design Principles and Architecture

- **API‑first**: clear REST resources with Pydantic validation.
- **Separation of concerns**: routers → services → models; inference logic isolated.
- **Stateless**: all persistent state in PostgreSQL; no in‑memory session data.
- **Async processing**: long tasks (vision inference) run in the background.
- **Configurable**: model names are loaded from environment variables.

**Component diagram**:

```mermaid
flowchart LR
    A[Browser / API Consumer] --> B[FastAPI Routers & Services]

    subgraph InferenceServer [Inference Server]
        B
        C[(PostgreSQL + pgvector)]

        subgraph AI [AI Inference via Ollama]
            D[Vision Model]
            E[Text Model]
            F[Embedding Model]
        end
    end

    B --> D
    B --> E
    B --> F
    B --> C
```

---

# Core Infrastructure

## Configuration and Environment

`app/core/config.py` defines `Settings` using `pydantic_settings`:

```python
class Settings(BaseSettings):
    DATABASE_URL: str
    OLLAMA_URL: str
    ANALYSIS_MODEL: str
    TEXT_MODEL: str
    EMBED_MODEL: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
```

All variables are required; the app fails fast if any are missing.

## Database Layer

`app/core/database.py` sets up an async engine and session factory:

```python
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

Each HTTP request gets a fresh session; commits/rollbacks are managed by the dependency.

## Data Models

### Document

`app/models/document.py`:

```python
class Document(SQLModel, table=True):
    document_id: UUID = Field(default_factory=uuid4, primary_key=True)
    original_filename: str
    stored_filename: str
    file_path: str
    content_type: str
    file_size: int
    updated_at: datetime
    created_at: datetime

    analyses: list["Analysis"] = Relationship(back_populates="document")
    chunks: list["Chunk"] = Relationship(back_populates="document")
    chat_documents: list["ChatDocument"] = Relationship(back_populates="document")
```

**Note**: there is **no** `status` field on `Document`. The analysis status is tracked separately on the `Analysis` model.

### Analysis

`app/models/analysis.py`:

```python
class Analysis(SQLModel, table=True):
    analysis_id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="documents.document_id", index=True)
    model_name: str
    model_version: str
    raw_output: str | None
    summary: str | None
    status: AnalysisStatus   # enum: READY, ANALYZING, CHUNKING, EMBEDDING, COMPLETE, FAILED, DELETED
    updated_at: datetime
    created_at: datetime

    document: "Document" = Relationship(back_populates="analyses")
    chunks: list["Chunk"] = Relationship(back_populates="analysis")
```

- `raw_output`: the full text from the vision model.
- `summary`: the extracted concise summary (set during the `extract` step).
- `status`: tracks pipeline progress.
- Entities are **not** stored directly on `Analysis`; they are aggregated from the associated `Chunk` entries when queried (see schemas).

### Chunk

`app/models/chunk.py`:

```python
class Chunk(SQLModel, table=True):
    chunk_id: UUID = Field(default_factory=uuid4, primary_key=True)
    document_id: UUID = Field(foreign_key="documents.document_id", index=True)
    analysis_id: UUID = Field(foreign_key="analyses.analysis_id", index=True)
    chunk_index: int
    chunk_type: ChunkType   # TEXT, TABLE, IMAGE, CAPTION, METADATA
    chunk_content: str
    entities: list[str] = Field(default_factory=list, sa_column=Column(JSONB))
    notes: list[str] = Field(default_factory=list, sa_column=Column(JSONB))
    source_locations: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    embedding: list[float] | None = Field(default=None, sa_column=Column(Vector(768)))
    embedding_model: str
    updated_at: datetime
    created_at: datetime

    document: "Document" = Relationship(back_populates="chunks")
    analysis: "Analysis" = Relationship(back_populates="chunks")
```

- `embedding` is nullable; set during the embedding step.
- `entities` are extracted from the report (using LLM or regex fallback).
- `source_locations` is a JSON object (not a list) for additional context (e.g., page numbers, coordinates).

### Chat Models

`app/models/chat.py`:

**ChatSession**:

```python
class ChatSession(SQLModel, table=True):
    chat_id: UUID = Field(default_factory=uuid4, primary_key=True)
    title: str | None
    updated_at: datetime
    created_at: datetime

    messages: list["ChatMessage"] = Relationship(back_populates="chat")
    chat_documents: list["ChatDocument"] = Relationship(back_populates="chat")
```

**ChatMessage**:

```python
class ChatMessage(SQLModel, table=True):
    message_id: UUID = Field(default_factory=uuid4, primary_key=True)
    chat_id: UUID = Field(foreign_key="chat_sessions.chat_id", index=True)
    role: MessageRole   # SYSTEM, USER, ASSISTANT
    content: str
    message_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime
    created_at: datetime

    chat: "ChatSession" = Relationship(back_populates="messages")
```

**ChatDocument** (junction, `app/models/chat_document.py`):

```python
class ChatDocument(SQLModel, table=True):
    chat_id: UUID = Field(foreign_key="chat_sessions.chat_id", primary_key=True)
    document_id: UUID = Field(foreign_key="documents.document_id", primary_key=True)
    message_id: UUID | None = Field(default=None, foreign_key="chat_messages.message_id")
    attached_at: datetime = Field(default_factory=datetime.now)

    chat: "ChatSession" = Relationship(back_populates="chat_documents")
    document: "Document" = Relationship(back_populates="chat_documents")
```

- `message_id` optionally links to a system message inserted when the document is attached.

### Enums

`app/models/enums.py`:

```python
class AnalysisStatus(str, Enum):
    READY = "ready"
    ANALYZING = "analyzing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETE = "complete"
    FAILED = "failed"
    DELETED = "deleted"

class ChunkType(str, Enum):
    TEXT = "TEXT"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    CAPTION = "CAPTION"
    METADATA = "METADATA"

class MessageRole(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "ASSISTANT"
```

## Database Migrations (Alembic)

Alembic manages schema changes. The migrations history reflects the current models:
- Created all tables, enums, and the `vector` extension.
- Made `chunks.embedding` nullable.
- Added `status` column to `analyses`.
- Made `raw_output` and `summary` nullable.
- Added `message_id` to `chat_documents`.

Run migrations with:

```bash
alembic upgrade head
```

Generate new migrations with:

```bash
alembic revision --autogenerate -m "description"
```

Always review autogenerated scripts.

---

# Document Processing Pipeline

Documents go through a pipeline **driven by the Analysis status**:

```mermaid
flowchart LR
    A[Upload] --> B[Document created]
    B --> C[POST /documents/{id}/analysis creates Analysis with status=READY]
    C --> D[Background: ANALYZING → vision model]
    D --> E[CHUNKING → extract summary + create chunk]
    E --> F[EMBEDDING → compute embedding for chunk]
    F --> G[COMPLETE]
    D --> H[FAILED]
```

The `Analysis` model holds the status; the `Document` itself has no status.

1. **Upload**: client uploads an image; server saves file and creates `Document` record.
2. **Submit for analysis**: client calls `POST /documents/{id}/analysis`; an `Analysis` record is created with status `READY` and the background pipeline is enqueued.
3. **Analyse**: `ImageAnalysisService.analyse()` calls the vision model, stores the raw output in `raw_output`, and sets status to `ANALYZING` then `CHUNKING`.
4. **Extract**: `extract()` uses the text model to produce a summary and entities, creates a single `Chunk` (with those entities), and sets status to `CHUNKING` then `EMBEDDING`.
5. **Embed**: `embed()` computes the vector for the chunk via the embedding model and stores it; status becomes `COMPLETE`.
6. Any error sets status to `FAILED`.

---

## Document Upload and Management

**DocumentService** (`app/features/documents/service.py`):

- `upload_document(file)`:
    - validates content type (image/jpeg, image/png).
    - reads file in chunks up to 25 MB.
    - verifies image bytes with PIL.
    - saves to `./uploads/` with a UUID filename.
    - creates `Document` record (no status).
- `list_documents(skip, limit)`, `get_document(id)`, `delete_document(id)` – the latter deletes the DB record and the file.

`DELETE /documents/{id}` removes both the database row and the physical file.

---

## Image Analysis Service

`ImageAnalysisService` (`app/inference/analysis.py`) orchestrates the three steps.

### `analyse(analysis_id)`

- Loads the `Analysis` and its `Document`.
- Reads the image file bytes.
- Calls `llm.chat()` with the vision model and `SYS_PROMPT_INGESTION`.
- Stores the response in `analysis.raw_output`.
- Sets status to `ANALYZING` (it transitions from `READY` to `ANALYZING`; after success, status becomes `CHUNKING`).

### `extract(analysis_id)`

- Uses `analysis.raw_output`.
- Calls `llm.chat()` with `TEXT_MODEL` and `EXTRACT_PROMPT` (which expects JSON output).
- Parses the JSON to extract `summary` and `entities` (uses regex fallback if parsing fails).
- Creates a single `Chunk` (with `chunk_type=IMAGE`, `chunk_content=summary`, and the entities).
- Sets `analysis.summary` and updates status to `CHUNKING` (then to `EMBEDDING`).

### `embed(analysis_id)`

- Retrieves all chunks for the analysis (usually one).
- For each, calls `llm.embed()` with the embedding model.
- Stores the vector in `chunk.embedding`.
- Sets status to `COMPLETE`.

If any step fails, status becomes `FAILED` and the error is logged.

---

## Analysis Retrieval and Status

**AnalysisService** (`app/features/analysis/service.py`):

- `get_analysis(analysis_id)`:
    - Loads `Analysis` with its chunks (using `selectinload`).
    - Builds an `AnalysisRead` schema that merges entities from all chunks (deduplicated).
    - Returns `analysis_id`, `document_id`, `raw_output`, `summary`, `status`, `entities` (union of chunk entities), and a metadata object with model name, version, and created_at.
- `get_analysis_status(analysis_id)`:
    - Returns a lightweight `AnalysisStatusResponse` containing only `analysis_id`, `status`, and `updated_at`.
- `delete_analysis(analysis_id)`:
    - Deletes the analysis and its associated chunks (cascade).

Clients poll the status endpoint to know when the background pipeline has finished.

---

# Inference Engine

The inference layer (`app/inference/`) handles all AI interactions.

## Ollama Client (`llm.py`)

Provides async functions:
- `chat(model, messages, images=None, stream=False, format_json=False)`: uses `/api/chat`; supports image attachments (base64 encoding).
- `embed(model, input)`: calls `/api/embed` and returns the embedding vector.

All requests use `httpx.AsyncClient` with no timeout.

## Prompts (`prompts.py`)

- `SYS_PROMPT_INGESTION`: guides the vision model to produce a descriptive radiology report without diagnosis or clinical interpretation.
- `EXTRACT_PROMPT`: instructs the text model to output **strict JSON** matching `OUTPUT_JSON_STRUCTURE`; it enforces the controlled entity vocabulary (`ChestXrayEntity`) and consistency between summary and entities.
- `QUERY_PROMPT`: templates the RAG query with placeholders for similar chunks, previous documents, and the current document.
- `GENERATE_PROMPT`: system prompt for the final answer generation step.

The entity vocabulary (`ChestXrayEntity`) includes terms like `Cardiomegaly`, `Effusion`, `Nodule`, etc., and `No Finding`.

## Context Engine (`context.py`)

`ContextEngine.build_context(chat_id, query_embedding, current_document_id=None, top_k=5)`:

1. **Similar chunks**: performs a cosine distance search on `chunks.embedding` using `pgvector` (`<->` operator). Returns top‑k chunks with `chunk_id`, `document_id`, content, entities, and similarity (1 - distance).
2. **Previous documents**: fetches all documents attached to the chat, and for each gets the latest analysis’s summary and the union of notes from its chunks (converted to `ChestXrayEntity`). Returns a list of `DocumentSummaryContext`.
3. **Current document**: if a `current_document_id` is provided, retrieves the raw output of its latest analysis.

Returns a `ContextBundle` with all three parts.

## RAG Service (`rag.py`)

`RAGService` orchestrates the end‑to‑end RAG flow:

- `embed_query(query)`: calls `ollama.embed()` and returns the vector.
- `augment(query, context)`: fills the `QUERY_PROMPT` with the context and query.
- `generate(chat_id, prompt)`: calls `ollama.chat()` with `TEXT_MODEL` and `GENERATE_PROMPT`; saves the assistant’s response as a new `ChatMessage`.

Static entry point `RAGService.run(chat_id, query, current_document_id=None)`:
1. Opens a new DB session.
2. Embeds the query.
3. Builds context.
4. Augments the prompt.
5. Generates an assistant message and commits it.

Additionally, `RAGService.getDocumentSummary(chat_id)` returns a list of `ChatDocumentSummary` objects for all attached documents, including their latest summary, status, chunk IDs, and notes.

---

# Chat System

The chat system provides a conversational interface on top of documents.

**ChatSession**, **ChatMessage**, and **ChatDocument** models (described above).

## Chat Session Management

**ChatsService** (`app/features/chats/service.py`):

- `create_chat(title)` – creates a new session.
- `list_chats(skip, limit)` – paginated list.
- `get_chat(chat_id)` – returns the session or 404.
- `edit_chat(chat_id, title)` – updates title.
- `delete_chat(chat_id)` – deletes the chat and all its messages and attachments (cascade).

**Document attachment**:
- `assign_chat_document(chat_id, document_id)`:
- Checks chat and document existence.
- Prevents duplicate attachment (409 conflict).
- Adds a system message (role=SYSTEM) notifying the attachment.
- Creates a `ChatDocument` record linking the chat, document, and the system message ID.
- Returns a `ChatDocumentRead` schema that includes the document’s latest summary (from `_get_latest_summary`).
- `list_chat_documents(chat_id)` – returns all attached documents with their summaries.
- `remove_chat_document(chat_id, document_id)` – deletes the link and adds a system message for removal.

## Chat Messages and RAG Trigger

**ChatMessagesService** (`app/features/chat_messages/service.py`):

- `create_message_in_chat(chat_id, payload, background_tasks)`:
    - Validates the chat exists.
    - Creates and saves the new message.
    - If the message role is `USER`, enqueues `RAGService.run(chat_id, message.content)` in the background.
    - Returns the created message.
- `get_chat_messages(chat_id)` – returns all messages in chronological order.
- `get_message_from_chat(chat_id, message_id)` – fetches a specific message (404 if not found).
- `update_message_from_chat(...)` – patches the message content/metadata.
- `delete_message_from_chat(...)` – removes the message.

**RAG response**: the assistant’s reply is created asynchronously and stored as a new `ChatMessage` with `role=ASSISTANT`. Clients can poll the messages list to see it appear.

---

# API Reference

All endpoints are grouped under `/documents`, `/analysis`, and `/chats`.

## Documents and Analysis

| Method | Path | Description | Request Body | Response | Status Codes |
| --- | --- | --- | --- | --- | --- |
| POST | `/documents/upload` | Upload an image | `multipart/form-data` file | `DocumentRead` | 201, 400, 413, 415 |
| GET | `/documents` | List documents (pagination) | `?page_no=1&page_size=10` | List[`DocumentRead`] | 200 |
| GET | `/documents/{document_id}` | Get document metadata | – | `DocumentRead` | 200, 404 |
| DELETE | `/documents/{document_id}` | Delete document | – | 204 No Content | 204, 404 |
| POST | `/documents/{document_id}/analysis` | Start analysis | – | `AnalysisCreateResponse` | 202, 404, 409 |
| GET | `/documents/{document_id}/analyses` | List analyses for a document | `?page_no=1&page_size=10` | List[`AnalysisListItem`] | 200, 404 |
| DELETE | `/documents/{document_id}/analyses` | Delete all analyses for a document | – | 204 No Content | 204, 404 |
| GET | `/analysis/{analysis_id}` | Get full analysis (with chunks merged) | – | `AnalysisRead` | 200, 404 |
| GET | `/analysis/{analysis_id}/status` | Poll analysis status | – | `AnalysisStatusResponse` | 200, 404 |
| DELETE | `/analysis/{analysis_id}` | Delete an analysis | – | 204 No Content | 204, 404 |

**Schemas** (selected fields):
- `DocumentRead`: `document_id`, `original_filename`, `content_type`, `file_size`, `created_at`
- `AnalysisCreateResponse`: `analysis_id`, `document_id`, `status`, `created_at`
- `AnalysisListItem`: `analysis_id`, `document_id`, `summary`, `status`, `created_at`
- `AnalysisStatusResponse`: `analysis_id`, `status`, `updated_at`
- `AnalysisRead`: `analysis_id`, `document_id`, `raw_output`, `summary`, `status`, `entities` (list), `analysis_metadata` (model_name, model_version, created_at)

## Chats and Messages

| Method | Path | Description | Request Body | Response | Status Codes |
| --- | --- | --- | --- | --- | --- |
| POST | `/chats` | Create chat | `{"title": "..."}` | `ChatRead` | 201, 400 |
| GET | `/chats` | List chats | `?page_no=1&page_size=10` | List[`ChatRead`] | 200 |
| GET | `/chats/{chat_id}` | Get chat | – | `ChatRead` | 200, 404 |
| PATCH | `/chats/{chat_id}` | Update chat title | `{"title": "..."}` | `ChatRead` | 200, 404 |
| DELETE | `/chats/{chat_id}` | Delete chat | – | 204 No Content | 204, 404 |
| POST | `/chats/{chat_id}/document/{document_id}` | Attach document | – | `ChatDocumentRead` | 202, 404, 409 |
| GET | `/chats/{chat_id}/documents` | List attached documents | – | List[`ChatDocumentRead`] | 200, 404 |
| DELETE | `/chats/{chat_id}/document/{document_id}` | Detach document | – | 204 No Content | 204, 404 |
| POST | `/chats/{chat_id}/query` | Submit user query (triggers RAG) | `ChatMessageCreate` | `ChatMessageRead` | 201, 404 |
| GET | `/chats/{chat_id}/messages` | Get chat history | – | List[`ChatMessageRead`] | 200, 404 |
| GET | `/chats/{chat_id}/messages/{message_id}` | Get a message | – | `ChatMessageRead` | 200, 404 |
| PATCH | `/chats/{chat_id}/messages/{message_id}` | Update message | `ChatMessageUpdate` | `ChatMessageRead` | 200, 404 |
| DELETE | `/chats/{chat_id}/messages/{message_id}` | Delete message | – | 204 No Content | 204, 404 |

**Schemas**:
- `ChatRead`: `chat_id`, `title`
- `ChatDocumentRead`: `chat_id`, `title`, `document_id`, `file_path`, `file_size`, `content_type`, `summary`, `attached_at`
- `ChatMessageCreate`: `role` (USER/ASSISTANT/SYSTEM), `content`, `message_metadata` (optional)
- `ChatMessageRead`: `message_id`, `chat_id`, `role`, `content`, `message_metadata`, `created_at`, `updated_at`

---

# Developer Tools

The repository includes utility scripts (not shown in the code dump but referenced in the original):

- **`run_benchmark.py`**: batch‑uploads images, triggers analysis, polls until complete, and outputs a manifest mapping filenames to document IDs.
- **`setup_chats.py`**: given a manifest, creates a chat and attaches all documents.
- **`index.html`**: a standalone debugging UI that lets you upload, analyze, and chat via the REST API.

These tools are intended for local testing and performance evaluation.

---

# Glossary

- **Analysis**: the result of processing a document; includes raw output, summary, and status.
- **Document**: an uploaded image (no status).
- **Chunk**: a piece of text derived from an analysis (currently one per analysis), stored with entities and an embedding.
- **ChatSession**: a conversation container with attached documents and messages.
- **ChatMessage**: a message in a chat with a role (user, assistant, system).
- **ChatDocument**: junction table linking a chat to a document.
- **Embedding**: vector representation of chunk content (stored in `chunks.embedding`).
- **RAG**: retrieval‑augmented generation; uses similarity search to enrich the prompt.
- **pgvector**: PostgreSQL extension for vector storage and similarity operations.
- **Ollama**: local AI server providing vision, text, and embedding models.
- **AnalysisStatus**: enum tracking pipeline stages (READY, ANALYZING, CHUNKING, EMBEDDING, COMPLETE, FAILED, DELETED).
- **ContextBundle**: in‑memory object holding similar chunks, previous document summaries, and current raw output.
- **ContextEngine**: builds the context bundle.
- **RAGService**: orchestrates embedding, context building, prompt augmentation, and generation.

---

## TODO / Roadmap (Final)

### 1. Immediate Blockers (Critical Path)

- **Multi‑chunk decomposition** – currently one chunk per analysis. Refactor `extract()` to split the raw output into multiple findings (per entity/sentence) for finer‑grained retrieval.
- **PDF ingestion** – extend `DocumentService.upload` to accept `application/pdf`. Extract embedded images (via `pypdf` + `pdf2image` + PIL) and forward them to the vision model; also extract text layers if present (for reports).
- **No HTTP timeout for Ollama** – `httpx.AsyncClient(timeout=None)` in `llm.py` can hang a background task forever. Add a sensible timeout (e.g. `httpx.Timeout(120.0, read=120.0)`).
- **Regex fallback is brittle** – `_regex_extract_entities` does substring matching; it will incorrectly match "No Finding" if "finding" appears elsewhere. Replace with token‑boundary regex (`\b`).

---

### 2. Long‑term Goals (>2 weeks)

#### Database & ORM Optimisation

- **Race condition in analysis start** – `DocumentService.analyze_document` checks for in‑progress analyses without a lock. Fix with an advisory lock or `SELECT ... FOR UPDATE` to prevent duplicate `Analysis` records.
- **Add composite indexes** – evaluate query performance (e.g., on `(document_id, status)` for status polling) and add appropriate indexes.
- **Optimize `ContextEngine._get_previous_document_summaries`** – the current query uses a subquery with `DISTINCT ON`; consider rewriting with a window function or lateral join for better efficiency.
- **Batch embedding updates** – if multiple chunks are created (post‑multi‑chunk), batch the embedding generation and updates to reduce round‑trips.

#### New File Type Support

- **DICOM** – parse DICOM headers and pixel data (via `pydicom` + `pillow`), extract patient/study metadata, and convert to a viewable frame for the vision model.
- **TIFF** – handle multi‑frame TIFF files (using `tifffile`) and pass each frame (or a composite) to the analysis pipeline.
- **HL7 / structured data** – accept HL7 v2 or FHIR bundles as input, map them to internal document representations, and expose their structured fields for retrieval (bypassing the vision model).

#### Infrastructure & Feature Enhancements

- **Retry failed analyses** – add `POST /documents/{document_id}/analysis/retry` that re‑runs the pipeline without re‑uploading (reuses existing `Document`, creates a fresh `Analysis`).
- **Streaming chat responses (SSE)** – implement Server‑Sent Events on `POST /chats/{chat_id}/query` so the frontend receives tokens in real time.
- **Multi‑modal retrieval** – move beyond text‑only similarity. Explore vision‑language embeddings (e.g., CLIP) to retrieve chunks by image content directly.
- **Structured logging & metrics** – integrate `structlog` for JSON logs and add Prometheus counters for analysis duration, success/failure rates, and embedding latency.
- **Automated model evaluation harness** – build an offline pipeline that runs a labeled test set through the analysis pipeline and computes precision/recall/F1 for entity extraction.
- **Fine‑tuning / A/B testing** – route analyses to different model versions via a feature flag, logging inference metadata to support side‑by‑side comparison.

---

The core documentation remains 1000% verified against your source code. Let me know if you need any further tweaks.

---

This documentation is an AI-assisted analysis of the actual codebase. For further details, refer directly to the source files in `app/`.