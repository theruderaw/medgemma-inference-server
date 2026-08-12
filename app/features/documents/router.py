import time
from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Path, Query, UploadFile, status
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.documents.service import DocumentService
from app.schemas.analysis import AnalysisCreateResponse, AnalysisListItem
from app.schemas.document import DocumentRead
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_409,
    ERROR_415,
    ERROR_500,
)
from app.logger import logger

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
)


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: ERROR_400,
        415: ERROR_415,
        500: ERROR_500,
    },
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/upload",
        method="POST",
        filename=file.filename,
        content_type=file.content_type,
    )
    try:
        service = DocumentService(db)
        result = await service.upload_document(file)
        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "Request completed",
            status_code=201,
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "",
    response_model=list[DocumentRead],
    responses={
        500: ERROR_500,
    },
)
async def list_document(
    db: AsyncSession = Depends(get_db),
    page_no: int = Query(default=1),
    page_size: int = Query(default=10),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents",
        method="GET",
        page_no=page_no,
        page_size=page_size,
    )
    try:
        service = DocumentService(db)
        result = await service.list_documents((page_no - 1) * page_size, page_size)
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_document(
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/{document_id}",
        method="GET",
        document_id=str(document_id),
    )
    try:
        service = DocumentService(db)
        result = await service.get_document(document_id)
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def delete_document_permanently(
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/{document_id}",
        method="DELETE",
        document_id=str(document_id),
    )
    try:
        service = DocumentService(db)
        await service.delete_document(document_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "Request completed",
            status_code=204,
            duration_ms=round(duration, 2),
        )
        return
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.post(
    "/{document_id}/analysis",
    response_model=AnalysisCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: ERROR_404,
        409: ERROR_409,
        500: ERROR_500,
    },
)
async def analyze_document(
    background_tasks: BackgroundTasks,
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/{document_id}/analysis",
        method="POST",
        document_id=str(document_id),
    )
    try:
        service = DocumentService(db)
        result = await service.analyze_document(document_id, background_tasks)
        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "Request completed",
            status_code=202,
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "/{document_id}/analyses",
    response_model=list[AnalysisListItem],
    responses={
        500: ERROR_500,
    },
)
async def list_analyses(
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
    page_no: int = Query(default=1),
    page_size: int = Query(default=10),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/{document_id}/analyses",
        method="GET",
        document_id=str(document_id),
        page_no=page_no,
        page_size=page_size,
    )
    try:
        service = DocumentService(db)
        result = await service.list_analyses(
            document_id=document_id,
            skip=(page_no - 1) * page_size,
            limit=page_size,
        )
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.delete(
    "/{document_id}/analyses",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    }
)
async def delete_analyses(
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/documents/{document_id}/analyses",
        method="DELETE",
        document_id=str(document_id),
    )
    try:
        service = DocumentService(db)
        await service.delete_document_analyses(document_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info(
            "Request completed",
            status_code=204,
            duration_ms=round(duration, 2),
        )
        return
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
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise