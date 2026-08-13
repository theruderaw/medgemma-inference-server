from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Column, Enum, ForeignKey, Text  # <-- Import Text and ForeignKey
from sqlmodel import Field, Relationship, SQLModel

from app.models.enums import AnalysisStatus

if TYPE_CHECKING:
    from .chunk import Chunk
    from .document import Document


class Analysis(SQLModel, table=True):
    __tablename__ = "analyses"

    analysis_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
    )

    # CRITICAL: Add ondelete CASCADE for DB-003
    document_id: UUID = Field(
        sa_column=Column(
            ForeignKey("documents.document_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )

    # Required for DATA-002 (lineage)
    model_name: str = Field(nullable=False)
    model_version: str = Field(nullable=False)

    # Use Text to avoid truncation (LLM outputs can be huge)
    raw_output: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    summary: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    # NEW: For API-001 retries and QUEUE-002 DLQ handling
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    retry_count: int = Field(
        default=0,
        nullable=False,
        ge=0,
    )
    
    validated : bool = False

    status: AnalysisStatus = Field(
        default=AnalysisStatus.READY,
        sa_column=Column(
            Enum(
                AnalysisStatus,
                name="analysisstatus",
                create_type=True,
            ),
            nullable=False,
            # DO NOT add index=True here (defer DB-002 to P1)
        ),
    )

    # NEW: For OBS-003 latency tracking
    started_at: datetime | None = Field(
        default=None,
        nullable=True,
    )
    completed_at: datetime | None = Field(
        default=None,
        nullable=True,
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
    )
    
    #NEW: For DATA-002
    prompt_template: str = Field(
        default="",
        nullable=False
    )
    
# NEW: For DATA-002 (extraction prompt lineage)
    extract_prompt_template: str | None = Field(
        default=None,
        sa_column=Column(Text,nullable=True),
    )

    # Relationships with CASCADE (Matches Chunk behavior)
    document: "Document" = Relationship(
        back_populates="analyses",
        sa_relationship_kwargs={"cascade": "delete"},  # <-- ADD THIS
    )

    chunks: list["Chunk"] = Relationship(
        back_populates="analysis",
        sa_relationship_kwargs={"cascade": "delete"},  # <-- ADD THIS
    )