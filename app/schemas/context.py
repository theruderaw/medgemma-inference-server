from app.schemas.common import ORMBase
from datetime import datetime
from uuid import UUID

class ChatDocumentRead(ORMBase):
    chat_id: UUID
    title: str
    document_id: UUID
    file_path: str
    file_size: int
    content_type: str
    attached_at: datetime
