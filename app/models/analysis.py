from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import Column, Enum
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

    document_id: UUID = Field(
        foreign_key="documents.document_id",
        nullable=False,
        index=True,
    )

    model_name: str
    model_version: str

    raw_output: str | None = None
    summary: str | None = None

    status: AnalysisStatus = Field(
        default=AnalysisStatus.READY,
        sa_column=Column(
            Enum(
                AnalysisStatus,
                name="analysisstatus",
                create_type=True,
            ),
            nullable=False,
        ),
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
    )

    document: "Document" = Relationship(
        back_populates="analyses",
    )

    chunks: list["Chunk"] = Relationship(
        back_populates="analysis",
    )