from datetime import datetime
from uuid import UUID, uuid4
from typing import TYPE_CHECKING

from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON, Enum

from .enums import MessageRole

if TYPE_CHECKING:
    from .chat_document import ChatDocument


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    chat_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )

    title: str | None = Field(
        default=None,
        nullable=True
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    messages: list["ChatMessage"] = Relationship(
        back_populates="chat"
    )

    chat_documents: list["ChatDocument"] = Relationship(
        back_populates="chat"
    )


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    message_id: UUID = Field(
        default_factory=uuid4,
        primary_key=True
    )

    chat_id: UUID = Field(
        foreign_key="chat_sessions.chat_id",
        nullable=False,
        index=True
    )

    role: MessageRole = Field(
        sa_column=Column(
            Enum(
                MessageRole,
                name="messagerole",
                create_type=True
            ),
            nullable=False
        )
    )

    content: str = Field(
        nullable=False
    )

    message_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(
            JSON,
            nullable=False
        )
    )

    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False
    )

    chat: "ChatSession" = Relationship(
        back_populates="messages"
    )