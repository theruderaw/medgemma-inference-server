from datetime import datetime
import fastapi
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.enums import AnalysisStatus

class AnalysisListItem(BaseModel):
    analysis_id: UUID
    document_id: UUID
    summary: str | None
    status: AnalysisStatus
    created_at: datetime

class AnalysisCreateResponse(BaseModel):
    analysis_id: UUID
    document_id: UUID
    status: AnalysisStatus
    created_at: datetime


class AnalysisStatusResponse(BaseModel):
    analysis_id: UUID
    status: AnalysisStatus
    updated_at: datetime


class AnalysisMetadata(BaseModel):
    model_name: str
    model_version: str
    created_at: datetime


class AnalysisRead(BaseModel):
    analysis_id: UUID
    document_id: UUID
    raw_output: str | None
    summary: str | None
    status: AnalysisStatus
    entities : list[str]
    analysis_metadata: AnalysisMetadata