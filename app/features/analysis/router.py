from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, Response, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.analysis.service import AnalysisService
from app.schemas.analysis import (
    AnalysisRead,
    AnalysisStatusResponse,
)
from app.schemas.errors import ERROR_404, ERROR_500

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisRead,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_latest_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = AnalysisService(db)
    return await service.get_analysis(analysis_id)


@router.get(
    "/{analysis_id}/status",
    response_model=AnalysisStatusResponse,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_analysis_status(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = AnalysisService(db)
    return await service.get_analysis_status(analysis_id)


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def delete_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    service = AnalysisService(db)
    await service.delete_analysis(analysis_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)