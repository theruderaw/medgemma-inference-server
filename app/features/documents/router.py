from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, BackgroundTasks, Depends, File, Path, Query, UploadFile, status
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

router = APIRouter(
    prefix="/documents",
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
    service = DocumentService(db)
    return await service.upload_document(file)


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
    service = DocumentService(db)
    return await service.list_documents(
        (page_no - 1) * page_size,
        page_size,
    )


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
    service = DocumentService(db)
    return await service.get_document(document_id)


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
    service = DocumentService(db)
    return await service.delete_document(document_id)



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
    background_tasks:BackgroundTasks,
    document_id: UUID = Path(...),
    db: AsyncSession = Depends(get_db),
):
    service = DocumentService(db)
    return await service.analyze_document(
        document_id,
        background_tasks
    )


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
    service = DocumentService(db)
    return await service.list_analyses(
        document_id=document_id,
        skip=(page_no - 1) * page_size,
        limit=page_size,
    )

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
    service = DocumentService(db)
    return await service.delete_document_analyses(document_id)
