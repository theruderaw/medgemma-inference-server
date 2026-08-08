from app.schemas.common import ORMBase
from app.models.enums import MessageRole
from datetime import datetime
from pydantic import Field
from typing import Any
from uuid import UUID

class ChatMessageBase(ORMBase):    
    role: MessageRole
    content: str
    message_metadata: dict[str,Any] = Field(default_factory = dict)
    
class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageUpdate(ORMBase):
    content: str | None = None
    message_metadata: dict[str,Any] = Field(default_factory = dict)
    
class ChatMessageRead(ChatMessageBase):
    message_id: UUID
    chat_id: UUID
    created_at: datetime
    updated_at: datetime