from datetime import datetime
from uuid import uuid4, UUID
from typing import TYPE_CHECKING

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, Enum


if TYPE_CHECKING:
    from .analysis import Analysis
    from .chunk import Chunk
    from .chat_document import ChatDocument


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    document_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )

    original_filename: str = Field(
        nullable=False
    )

    stored_filename: str = Field(
        unique=True,
        index=True,
        nullable=False
    )

    file_path: str = Field(
        nullable=False
    )

    content_type: str = Field(
        nullable=False
    )

    file_size: int = Field(
        nullable=False
    )



    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    analyses: list["Analysis"] = Relationship(
        back_populates="document"
    )

    chunks: list["Chunk"] = Relationship(
        back_populates="document"
    )

    chat_documents: list["ChatDocument"] = Relationship(
        back_populates="document"
    )