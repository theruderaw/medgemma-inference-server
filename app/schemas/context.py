from app.inference.types import ChestXrayEntity
from app.models.enums import AnalysisStatus
from app.schemas.common import ORMBase
from datetime import datetime
from uuid import UUID

class ChatDocumentRead(ORMBase):
    chat_id: UUID
    title: str | None
    document_id: UUID
    file_path: str
    file_size: int
    content_type: str
    summary: str | None
    attached_at: datetime
    
class ChatDocumentSummary(ORMBase):
    chat_id: UUID
    document_id: UUID
    analysis_id: UUID
    summary: str | None
    status: AnalysisStatus
    attached_at: datetime
    chunks: list[UUID]
    notes: list[ChestXrayEntity]
