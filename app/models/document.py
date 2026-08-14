from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlmodel import Field, Relationship, SQLModel

if TYPE_CHECKING:
    from .analysis import Analysis
    from .chat_document import ChatDocument
    from .chunk import Chunk


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    document_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True,
    )

    original_filename: str = Field(
        nullable=False,
    )

    stored_filename: str = Field(
        unique=True,
        index=True,
        nullable=False,
    )

    file_path: str = Field(
        nullable=False,
    )

    content_type: str = Field(
        nullable=False,
    )

    checksum: str | None = Field(
        default=None,
        max_length=64,
        index=True,
        # TODO: Make checksum unique once duplicate-document handling is implemented.
    )


    file_size: int = Field(
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

    analyses: list["Analysis"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
        },
    )

    chunks: list["Chunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade":"all,delete-orphan"
        }
    )

    chat_documents: list["ChatDocument"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade":"all,delete-orphan"
        }
    )