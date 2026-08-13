from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_event import AuditEvent

async def audit(
    db: AsyncSession,
    event_type: str,
    *,
    document_id: UUID | None = None,
    analysis_id: UUID | None = None,
    status: str = "success",
    audit_metadata: dict[str,Any] | None = None
) -> None:
    event = AuditEvent(
        document_id=document_id,
        analysis_id=analysis_id,
        event_type=event_type,
        status=status,
        audit_metadata=audit_metadata
    )
    
    db.add(event)