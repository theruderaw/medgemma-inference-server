"""
Task definitions for the Redis-backed queue.

A task represents *work to be performed*, not the result of that work.
The queue stores a small envelope (id, type, enqueued_at) around a
type-specific payload. Workers use `task_type` to decide which existing
service entry point (`ImageAnalysisService.run`, `PDFAnalysisService.run`,
`RAGService.run`) to invoke.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    ANALYSIS_IMAGE = "analysis:image"
    ANALYSIS_PDF = "analysis:pdf"
    RAG = "rag"


class AnalysisTaskPayload(BaseModel):
    """Enough information for the worker to invoke the analysis pipeline."""

    analysis_id: UUID
    document_type: str  # Document.content_type, e.g. "application/pdf"


class RAGTaskPayload(BaseModel):
    """Enough information for the worker to invoke RAGService.run()."""

    chat_id: UUID
    query: str
    current_document_id: UUID | None = None


class TaskEnvelope(BaseModel):
    """
    Wraps a payload with queue metadata.

    `task_id` doubles as the Redis key suffix for the payload/status
    entries and as the value pushed onto the pending list, so a queued
    task can be looked up or removed by id without scanning the list.
    """

    task_id: UUID = Field(default_factory=uuid4)
    task_type: TaskType
    # Kept as a plain dict (rather than a typed union) so JSON round-tripping
    # is unambiguous. Use `.analysis_payload()` / `.rag_payload()` to get a
    # validated, typed payload based on `task_type`.
    payload: dict[str, Any]
    enqueued_at: datetime = Field(default_factory=datetime.utcnow)

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str | bytes) -> "TaskEnvelope":
        return cls.model_validate_json(data)

    def analysis_payload(self) -> AnalysisTaskPayload:
        return AnalysisTaskPayload.model_validate(self.payload)

    def rag_payload(self) -> RAGTaskPayload:
        return RAGTaskPayload.model_validate(self.payload)

    @classmethod
    def for_analysis(cls, analysis_id: UUID, document_type: str) -> "TaskEnvelope":
        task_type = (
            TaskType.ANALYSIS_PDF
            if document_type == "application/pdf"
            else TaskType.ANALYSIS_IMAGE
        )
        return cls(
            task_type=task_type,
            payload=AnalysisTaskPayload(
                analysis_id=analysis_id,
                document_type=document_type,
            ).model_dump(mode="json"),
        )

    @classmethod
    def for_rag(
        cls,
        chat_id: UUID,
        query: str,
        current_document_id: UUID | None = None,
    ) -> "TaskEnvelope":
        return cls(
            task_type=TaskType.RAG,
            payload=RAGTaskPayload(
                chat_id=chat_id,
                query=query,
                current_document_id=current_document_id,
            ).model_dump(mode="json"),
        )