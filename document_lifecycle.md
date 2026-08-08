# Document Status Lifecycle

Every document progresses through a fixed set of states represented by the `document_status` enum. This state machine allows the frontend to accurately display processing progress, enables workers to resume interrupted jobs, and provides clear visibility into failures.

## PostgreSQL Enum

```sql
CREATE TYPE document_status AS ENUM (
    'UPLOADED',
    'SUBMITTED',
    'ANALYZING',
    'CHUNKING',
    'EMBEDDING',
    'READY',
    'FAILED',
    'DELETED'
);
```

---

## State Definitions

| State         | Description                                                                                                          |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| **UPLOADED**  | The file has been uploaded and stored successfully. No processing has started.                                       |
| **SUBMITTED** | The document has been accepted for processing and a worker has been assigned or queued.                              |
| **ANALYZING** | The inference service is generating the initial analysis from the uploaded document.                                 |
| **CHUNKING**  | The analysis output is being split into semantic chunks for retrieval. Each chunk contains at most 768 model tokens. |
| **EMBEDDING** | Vector embeddings are being generated for every chunk. Each embedding is a 768-dimensional vector.                   |
| **READY**     | Processing has completed successfully. The document is fully searchable and available for chat retrieval.            |
| **FAILED**    | Processing terminated due to an unrecoverable error. Error details should be logged separately.                      |
| **DELETED**   | The document has been permanently removed from the system.                                                           |

---

## State Transition Diagram

```
UPLOADED
    │
    ▼
SUBMITTED
    │
    ▼
ANALYZING
    │
    ▼
CHUNKING
    │
    ▼
EMBEDDING
    │
    ▼
READY

Any State
    │
    ▼
FAILED

READY
    │
    ▼
DELETED
```

---

## Design Principles

* Documents may only move forward through the processing pipeline.
* Workers are responsible for updating document status after completing each stage.
* The frontend should treat `READY` as the only state in which a document can be queried through the RAG pipeline.
* `FAILED` is a terminal state unless a retry mechanism explicitly resets the document to `SUBMITTED`.
* Each status transition should be persisted atomically with the corresponding processing step to ensure consistency.
