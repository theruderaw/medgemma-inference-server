from .common import ORMBase
from uuid import UUID

class ChatCreate(ORMBase):
    title: str | None = None

class ChatUpdate(ORMBase):
    title: str | None = None

class ChatRead(ORMBase):
    chat_id: UUID
    title: str | None = None