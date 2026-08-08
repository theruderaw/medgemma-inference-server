from datetime import datetime
from uuid import UUID

from .common import ORMBase

class DocumentRead(ORMBase):
    document_id: UUID
    original_filename: str
    content_type: str
    file_size: int
    created_at: datetime
