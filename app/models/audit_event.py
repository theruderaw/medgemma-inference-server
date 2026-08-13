from datetime import datetime,timezone
from typing import Any
from uuid import uuid4,UUID

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AuditEvent(SQLModel,table=True):
    __tablename__ = "audit_events"
    
    id: UUID = Field(default_factory=uuid4,primary_key=True)
    
    document_id : UUID | None = Field(
        default = None,
        foreign_key = "documents.document_id",
        index = True
    )
    
    analysis_id : UUID | None = Field(
        default = None,
        foreign_key = "analyses.analysis_id",
        index = True
    )
    
    event_type : str = Field(
        max_length = 100,
        index = True
    )
    
    status: str = Field(
        max_length=30
    )
    
    audit_metadata: dict[str,Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False)
    )
    
    created_at: datetime = Field(
        default_factory = lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            index=True
        ) 
    )