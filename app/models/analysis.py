from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4,UUID
from sqlalchemy import Enum

from sqlmodel import SQLModel,Field,Relationship,Column
from app.models.enums import AnalysisStatus

if TYPE_CHECKING:
    from .document import Document
    from .chunk import Chunk
    
class Analysis(SQLModel,table=True):
    __tablename__ = "analyses"
    
    analysis_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )
    
    document_id: UUID = Field(
        foreign_key="documents.document_id",
        nullable=False,
        index=True
    )
    
    model_name: str
    model_version: str
    
    raw_output: str | None
    summary: str | None
    
    status: AnalysisStatus = Field(
        default=AnalysisStatus.READY,
        sa_column=Column(
            Enum(
                AnalysisStatus,
                name="analysisstatus",
                create_type=True
            ),
            nullable=False
        )
    )
    
    updated_at: datetime = Field(
            default_factory=datetime.now
    )
    created_at: datetime = Field(
        default_factory=datetime.now
    )
    
    document: "Document" = Relationship(
        back_populates="analyses"
    )
    
    chunks: list["Chunk"] = Relationship(
        back_populates="analysis"
    )