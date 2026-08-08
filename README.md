# Document Analysis & RAG Chat Service

A FastAPI backend for uploading medical images (chest X-rays), running VLM-based
analysis on them, extracting structured findings, and answering questions about
them through a retrieval-augmented chat interface backed by pgvector.

---

## How it works

1. **Upload** — a JPEG/PNG is uploaded via `POST /documents/upload`, validated
   (content-type header **and** decoded bytes via Pillow), and stored on disk
   under `./uploads/{document_id}{ext}`.
2. **Analyze** — `POST /documents/{document_id}/analysis` creates an `Analysis`
   record and schedules a background pipeline (`ImageAnalysisService.run`):
   - `analyse()` — sends the image to a vision-language model via Ollama
     (`ANALYSIS_MODEL`) and stores the raw report text.
   - `extract()` — asks a text model (`TEXT_MODEL`) to pull a structured
     summary + entity list out of the raw report as JSON. If the LLM call
     fails or returns unparseable/empty JSON, falls back to regex-based
     extraction against a fixed entity vocabulary (`ChestXrayEntity`).
   - `embed()` — embeds the extracted summary (`EMBED_MODEL`) and stores it on
     a `Chunk` row for retrieval.
   - Any exception during this pipeline marks the analysis `FAILED` and clears
     `summary`/`raw_output` — by design, `FAILED` means "nothing usable here,
     retry from scratch" rather than exposing partial state.
3. **Chat** — messages posted to `POST /chats/{chat_id}/query` are saved, and a
   background RAG cycle (`RAGService.run`) runs: embed the query, retrieve the
   nearest `Chunk`s by cosine distance, build a context-augmented prompt, and
   generate an assistant reply that's saved back onto the same chat.
4. **Context** — documents/images/prompts can also be attached directly to a
   chat as context via `POST /chats/{chat_id}/context`, independent of the
   analysis pipeline.

---

## Tech stack


- **FastAPI 0.141.1** on **Starlette 1.4.1** / **Uvicorn 0.52.1** (with
  `uvloop` + `httptools` for the async event loop and HTTP parsing)
- **SQLAlchemy 2.0.51 (async) + SQLModel 0.0.39** — ORM / data models
- **Alembic 1.19.0** — DB migrations (present, though no migration files were
  reviewed here)
- **PostgreSQL with `pgvector` 0.5.0** — confirms the `pgvector` column type
  backing `Chunk.embedding.cosine_distance(...)`
- **Both `asyncpg` 0.31.0 and `psycopg`/`psycopg-binary` 3.3.4 are present** —
  worth confirming which one `database.py` actually uses for the async
  engine; having both installed is unusual and may be leftover from a driver
  switch
- **Pillow 12.3.0** — image validation on upload
- **opencv-python 5.0.0.93** — installed but not seen used in the reviewed
  service code; may be used elsewhere in image preprocessing
- **torch 2.13.0 / torchvision 0.28.0 / triton 3.7.1 / `cuda-*` packages** —
  a full local GPU-capable ML stack is installed, which is notable since the
  analysis/extraction/embedding calls reviewed here all go through Ollama
  over HTTP rather than loading models directly. Suggests either local model
  loading happens elsewhere in the codebase (not reviewed) or this is
  leftover from an earlier architecture — worth confirming which
- **aiofiles 25.1.0** — async file I/O for storing uploads
- **pydantic 2.13.4 / pydantic-settings 2.14.2** — validation and
  environment-based configuration
- **python-multipart 0.0.32** — required by FastAPI for the multipart upload
  endpoints (`/documents/upload`, `/chats/{chat_id}/context`)
- **Ollama** — local LLM/VLM serving (not in `requirements.txt` since it runs
  as a separate service, reached via `OLLAMA_URL`)

---

## Project layout

```
app/
  core/
    config.py          # Settings (env vars)
    database.py         # DB engine/session (not reviewed)
  features/
    documents/           # upload, list, get, delete documents; trigger analysis
    analysis/             # read analysis by id, poll status, delete
    chats/                  # create/list/get/patch/delete chat sessions
    chat_messages/            # post/read/edit/delete messages in a chat
    context/                   # attach/list/delete chat-level context (doc/prompt/image)
  inference/
    llm.py               # chat() / embed() wrappers around Ollama
    analysis.py            # ImageAnalysisService (analyse -> extract -> embed)
    rag.py                   # RAGService (retrieve -> augment -> generate)
    prompts.py                # SYS_PROMPT_INGESTION, USER_PROMPT_INGESTION,
                               # EXTRACT_PROMPT, QUERY_PROMPT, GENERATE_PROMPT
  models/
    document.py, analysis.py, chunk.py, chat.py, enums.py
  schemas/
    message.py, ... (Pydantic request/response schemas)
main.py                 # FastAPI app, CORS, router registration
```

---

## API surface

All endpoints return a consistent `ErrorResponse` shape
(`{ error, message, details }`) for 4xx/5xx, plus standard FastAPI 422
validation errors.

### Documents
| Method | Path | Description |
|---|---|---|
| POST | `/documents/upload` | Upload a JPEG/PNG (max 25 MB) |
| GET | `/documents` | Paginated list of documents |
| GET | `/documents/{document_id}` | Get one document |
| DELETE | `/documents/{document_id}` | Permanently delete a document + file |
| POST | `/documents/{document_id}/analysis` | Kick off analysis (background) |
| GET | `/documents/{document_id}/analyses` | Paginated list of analyses for a doc |
| DELETE | `/documents/{document_id}/analyses` | Delete all analyses for a doc |

### Analysis
| Method | Path | Description |
|---|---|---|
| GET | `/analysis/{analysis_id}` | Full analysis record (raw output, summary, entities) |
| DELETE | `/analysis/{analysis_id}` | Delete one analysis |
| GET | `/analysis/{analysis_id}/status` | Lightweight status poll |

### Chats
| Method | Path | Description |
|---|---|---|
| POST | `/chats` | Create a chat session |
| GET | `/chats` | Paginated list of chats |
| GET | `/chats/{chat_id}` | Get one chat |
| PATCH | `/chats/{chat_id}` | Update chat (e.g. title) |
| DELETE | `/chats/{chat_id}` | Delete a chat |

### Messages
| Method | Path | Description |
|---|---|---|
| POST | `/chats/{chat_id}/query` | Post a message; triggers background RAG reply |
| GET | `/chats/{chat_id}/messages` | Full message history |
| GET | `/chats/{chat_id}/messages/{message_id}` | Get one message |
| PATCH | `/chats/{chat_id}/messages/{message_id}` | Edit a message |
| DELETE | `/chats/{chat_id}/messages/{message_id}` | Delete a message |

### Context
| Method | Path | Description |
|---|---|---|
| GET | `/chats/{chat_id}/context` | List documents/context attached to a chat |
| POST | `/chats/{chat_id}/context` | Attach a document, prompt, and/or image to a chat |
| DELETE | `/chats/{chat_id}/context/{context_id}` | Remove one piece of context |

---

## Environment variables

Set in `.env` (loaded via `pydantic-settings`):

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string *(pgvector extension required)* |
| `OLLAMA_URL` | Base URL of the Ollama server |
| `ANALYSIS_MODEL` | Vision-language model used for initial image analysis, e.g. `qwen2.5vl:7b` — parsed as `name:version` in `analyze_document` |
| `TEXT_MODEL` | Text model used for structured extraction and RAG answer generation |
| `EMBED_MODEL` | Embedding model used for chunk and query embeddings |

All five are required (no defaults) — the app will fail to start if any is
missing from the environment or `.env` file.

---

## Prompts

Prompt text itself lives in `app/inference/prompts.py` and isn't reproduced
here; at a high level:

- **`SYS_PROMPT_INGESTION`** — instructs the VLM how to read and report on the
  uploaded chest X-ray image.
- **`USER_PROMPT_INGESTION`** — (present in `prompts.py`; not currently wired
  into `analyse()`, which only sends `SYS_PROMPT_INGESTION` as a user-role
  message — worth double-checking if a user-turn prompt was intended there).
- **`EXTRACT_PROMPT`** — instructs the text model to convert a raw report into
  structured JSON (`summary`, `entities`, `notes`).
- **`QUERY_PROMPT`** — assembles retrieved chunks + the user's question into a
  single prompt for the generation step.
- **`GENERATE_PROMPT`** — system prompt for the final RAG answer generation call.

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env   # fill in DATABASE_URL, OLLAMA_URL, *_MODEL vars

# 3. Ensure Postgres has the pgvector extension enabled
#    CREATE EXTENSION IF NOT EXISTS vector;

# 4. Run Ollama and pull the configured models
ollama pull <ANALYSIS_MODEL>
ollama pull <TEXT_MODEL>
ollama pull <EMBED_MODEL>

# 5. Start the API
uvicorn app.main:app --reload
```

CORS is currently configured to allow `localhost`/`127.0.0.1` on ports
`3000`, `5000`, and `5173` — update `origins` in `main.py` for other
frontends or production domains.

---

## Known gaps / things to double-check before production

- **No role gate on RAG triggering.** `create_message_in_chat` schedules a
  background RAG run for every message regardless of `role` — fine while only
  user-authored messages hit this endpoint, but worth gating explicitly if
  that assumption ever changes.
- **`USER_PROMPT_INGESTION` appears unused** in the current `analyse()` flow
  (see Prompts section above).

## TODO (after backend contract configuration)

These are deferred until the backend API contract is finalized, since each
touches request/response shape or status-code behavior that's easier to
settle once:

- **Concurrency guards.** Add checks so `POST /documents/{document_id}/analysis`
  and `POST /chats/{chat_id}/context` actually return the documented
  `409 Resource Conflict` when an analysis/context operation is already
  in progress for that document/chat, instead of silently creating a
  duplicate.
- **Upload size checks.** Reorder `upload_document` so the size limit is
  enforced before the file is fully read into memory (e.g. via `file.size`
  or a chunked/streaming read with an early cutoff), rather than checking
  `len(contents)` after `await file.read()` has already buffered it.
- **PDF read support.** Extend document upload/analysis to accept PDFs in
  addition to JPEG/PNG — content-type allowlist, byte validation, and how
  a multi-page PDF maps onto the existing single-image analysis pipeline
  (e.g. per-page analysis vs. a single combined pass) all need to be
  decided as part of the contract.
