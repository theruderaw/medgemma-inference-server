from uuid import UUID
import time

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Response, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.analysis.service import AnalysisService
from app.schemas.analysis import (
    AnalysisRead,
    AnalysisStatusResponse,
)


from app.schemas.errors import ERROR_404, ERROR_500
from app.logger import logger

router = APIRouter(
    prefix="/api/v1/analysis",
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
    start = time.perf_counter()
    logger.info("Request received", path="/api/v1/analysis/{analysis_id}", method="GET", analysis_id=str(analysis_id))
    try:
        service = AnalysisService(db)
        result = await service.get_analysis(analysis_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


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
    start = time.perf_counter()
    logger.info("Request received", path="/api/v1/analysis/{analysis_id}/status", method="GET", analysis_id=str(analysis_id))
    try:
        service = AnalysisService(db)
        result = await service.get_analysis_status(analysis_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


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
    start = time.perf_counter()
    logger.info("Request received", path="/api/v1/analysis/{analysis_id}", method="DELETE", analysis_id=str(analysis_id))
    try:
        service = AnalysisService(db)
        await service.delete_analysis(analysis_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=204, duration_ms=round(duration, 2))
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise

@router.post(
    "/{analysis_id}/validate",
    response_model=AnalysisRead,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def validate_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()

    logger.info(
        "Request received",
        path="/api/v1/analysis/{analysis_id}/validate",
        method="POST",
        analysis_id=str(analysis_id),
    )

    try:
        service = AnalysisService(db)
        result = await service.validate_analysis(analysis_id)

        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "Request completed",
            status_code=200,
            duration_ms=round(duration, 2),
        )

        return result

    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000

        logger.warning(
            "Request failed",
            status_code=e.status_code,
            detail=e.detail,
            duration_ms=round(duration, 2),
        )

        raise

    except Exception:
        duration = (time.perf_counter() - start) * 1000

        logger.exception(
            "Unhandled error",
            duration_ms=round(duration, 2),
        )

        raise