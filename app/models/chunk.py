from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Enum, ForeignKey  # Import ForeignKey explicitly
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, Relationship, SQLModel

from app.core.config import settings  # Import your settings
from .enums import ChunkType

if TYPE_CHECKING:
    from .analysis import Analysis
    from .document import Document


class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    chunk_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
    )

    document_id: UUID = Field(
        foreign_key="documents.document_id",
        nullable=False,
        index=True,
    )

    analysis_id: UUID = Field(
        foreign_key="analyses.analysis_id",
        nullable=False,
        index=True,
    )

    page_number: int = Field(
        nullable=False,
        ge=1,
    )

    chunk_index: int = Field(
        nullable=False,
        ge=0,
    )

    chunk_type: ChunkType = Field(
        default=ChunkType.TEXT,
        sa_column=Column(
            Enum(
                ChunkType,
                name="chunktype",
                create_type=True,
            ),
            nullable=False,
        ),
    )

    chunk_content: str = Field(
        nullable=False,
    )

    entities: list[str] = Field(
        default_factory=list,
        sa_column=Column(
            JSONB,
            nullable=False,
        ),
    )

    notes: list[str] = Field(
        default_factory=list,
        sa_column=Column(
            JSONB,
            nullable=False,
        ),
    )

    source_locations: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(
            JSONB,
            nullable=False,
        ),
    )

    embedding: list[float] | None = Field(
        default=None,
        sa_column=Column(
            Vector(settings.EMBEDDING_DIM),  
            nullable=True,
        ),
    )

    embedding_model: str = Field(
        nullable=False,
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
    )

    document: "Document" = Relationship(
        back_populates="chunks",
        sa_relationship_kwargs={"cascade": "delete"}  
    )

    analysis: "Analysis" = Relationship(
        back_populates="chunks",
        sa_relationship_kwargs={"cascade": "delete"}  
    )
    
    