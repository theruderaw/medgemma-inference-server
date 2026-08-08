# Architecture

## Overview

This project is a Retrieval-Augmented Generation (RAG) platform for multimodal
medical documents. The system accepts uploaded documents, processes them using
specialized AI models, stores structured outputs and vector embeddings, and
provides conversational querying over the processed knowledge.

The architecture follows a service-oriented design where each layer has a single
responsibility.

---

# High-Level Architecture

```racket
                React Frontend
                       │
                       ▼
              Express REST API
                       │
      ┌────────────────┴────────────────┐
      │                                 │
      ▼                                 ▼
 PostgreSQL + pgvector           FastAPI Inference
                                        │
┌───────────────────────────┴───────────────────────────┐
│                         │                             │
▼                         ▼                             ▼
Vision Models             Chat Models                  Embedding Models
```

---

# Design Principles

- API-first development
- Stateless services
- Separation of concerns
- Single responsibility
- Database as source of truth
- Idempotent processing
- Async long-running operations
- Extensible pipeline
- Model-agnostic inference

---

# Components

## Frontend

Responsibilities

- Upload files
- Monitor processing
- Display results
- Chat interface
- Session management

Never

- Perform AI inference
- Store business logic
- Handle embeddings

---

## Express API

Responsibilities

- Request validation
- Authentication
- Route definitions
- Response formatting
- Job orchestration

Never

- Run LLMs
- Generate embeddings
- Parse medical documents

---

## FastAPI Inference Service

Responsibilities

- Vision inference
- Text extraction
- Embedding generation
- Retrieval
- Prompt management
- Model interaction

Never

- User authentication
- HTTP session management
- Frontend concerns

---

## Database

Stores

- Documents
- Analyses
- Chunks
- Embeddings
- Chat history
- Metadata

Database never performs inference.

---

# Processing Pipeline

Upload
↓

Store Document
↓

Submit

↓

Analysis

↓

Structured Extraction

↓

Chunk Generation

↓

Embedding Generation

↓

Ready for Retrieval

↓

Chat Queries

---

# Chat Pipeline

Question

↓

Embed Query

↓

Vector Search

↓

Context Selection

↓

Prompt Construction

↓

LLM

↓

Response

---

# Directory Responsibilities

frontend/
React application

backend/
Express REST API

inference/
FastAPI AI services

database/
schema
migrations

docs/
architecture
api
workflow

---

# Service Layer

Each feature should expose a service.

Example

DocumentService
AnalysisService
EmbeddingService
RetrievalService
ChatService

Controllers should never contain business logic.

---

# Repository Layer

Repositories only interact with the database.

DocumentRepository

AnalysisRepository

ChunkRepository

ChatRepository

Repositories never call AI models.

---

# AI Layer

VisionModel

ChatModel

EmbeddingModel

Every model implements a common interface.

This allows replacing Ollama with OpenAI, Gemini, vLLM,
or Hugging Face without changing business logic.

---

# State Machine

Uploaded

↓

Submitted

↓

Processing

↓

Analyzed

↓

Embedded

↓

Ready

↓

Deleted

Invalid state transitions are rejected.

---

# Error Handling

Recoverable

- Retry inference
- Retry embeddings

Fatal

- Unsupported file
- Corrupt document
- Missing database record

---

# Logging

Every request receives

- Request ID
- Document ID
- Chat ID
- Processing time

All services log structured JSON.

---

# Configuration

No hardcoded values.

Everything comes from

- environment variables
- configuration files

---

# Extensibility

Supported today

- PNG
- JPG
-string input

Future

- PDF
- DICOM
- DOCX
- PPTX
- XLSX
- CSV
- Audio
- Video

No existing interfaces should change when adding formats.

---

# Non-goals

The API is not responsible for

- model training
- dataset generation
- GPU scheduling
- user management
- deployment infrastructure

---

# Future Improvements

- Background workers
- Message queues
- Distributed inference
- Result caching
- Hybrid retrieval
- Reranking
- OCR pipeline
- Streaming responses
- Multi-document retrieval
- Agent workflows

---

# Guiding Rule

Every piece of code should have exactly one responsibility.

If changing one feature requires modifying multiple unrelated modules,
the architecture should be reconsidered before adding more code.

[](https://app.notion.com/p/3b4766dace718081b76ac4e5e2f4da67?pvs=21)