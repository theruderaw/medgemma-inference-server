from datetime import datetime
from uuid import UUID

from sqlmodel import SQLModel, Field, Relationship

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .chat import ChatSession
    from .document import Document
    
class ChatDocument(SQLModel,table=True):
    __tablename__ = "chat_documents"
    
    chat_id: UUID = Field(
        foreign_key="chat_sessions.chat_id",
        primary_key=True
    )
    
    document_id: UUID = Field(
        foreign_key="documents.document_id",
        primary_key=True
    )
    
    message_id: UUID = Field(
        foreign_key="chat_messages.message_id",
        default=None
    )
    
    attached_at: datetime = Field(
        default_factory=datetime.now
    )
    
    chat : "ChatSession" = Relationship(
        back_populates="chat_documents"
    )
    
    document: "Document" = Relationship(
        back_populates="chat_documents"
    )