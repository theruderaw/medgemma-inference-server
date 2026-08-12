import time
from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import APIRouter, status, Query, Depends, HTTPException
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.chats.service import ChatsService
from app.schemas.chat import ChatCreate, ChatRead, ChatUpdate
from app.schemas.context import ChatDocumentRead
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_409,
    ERROR_500,
)
from app.logger import logger

router = APIRouter(
    prefix="/api/v1/chats",
    tags=["Chats"],
)


@router.post(
    "",
    response_model=ChatRead,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: ERROR_400,
        500: ERROR_500,
    },
)
async def create_chat_session(
    payload: ChatCreate,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info("Request received", path="/api/v1/chats", method="POST")
    try:
        service = ChatsService(db)
        result = await service.create_chat(title=payload.title)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=201, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "",
    response_model=list[ChatRead],
    responses={
        500: ERROR_500,
    },
)
async def list_all_chats(
    db: AsyncSession = Depends(get_db),
    page_no: int = Query(default=1),
    page_size: int = Query(default=10)
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats",
        method="GET",
        page_no=page_no,
        page_size=page_size,
    )
    try:
        service = ChatsService(db)
        result = await service.list_chats((page_no - 1) * page_size, page_size)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "/{chat_id}",
    response_model=ChatRead,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_chat_details(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}",
        method="GET",
        chat_id=str(chat_id),
    )
    try:
        service = ChatsService(db)
        result = await service.get_chat(chat_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.patch(
    "/{chat_id}",
    response_model=ChatRead,
    responses={
        400: ERROR_400,
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def patch_chat_session(
    chat_id: UUID,
    payload: ChatUpdate,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}",
        method="PATCH",
        chat_id=str(chat_id),
    )
    try:
        service = ChatsService(db)
        result = await service.edit_chat(chat_id=chat_id, title=payload.title)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.delete(
    "/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def delete_chat_session(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}",
        method="DELETE",
        chat_id=str(chat_id),
    )
    try:
        service = ChatsService(db)
        await service.delete_chat(chat_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=204, duration_ms=round(duration, 2))
        return
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


# Removed duplicate POST /{chat_id}/document/{document_id} – only one version kept


@router.post(
    "/{chat_id}/document/{document_id}",
    response_model=ChatDocumentRead,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        404: ERROR_404,
        409: ERROR_409,
        500: ERROR_500,
    },
)
async def assign_chat_document(
    chat_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/document/{document_id}",
        method="POST",
        chat_id=str(chat_id),
        document_id=str(document_id),
    )
    try:
        service = ChatsService(db)
        result = await service.assign_chat_document(
            chat_id=chat_id,
            document_id=document_id,
        )
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=202, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.get(
    "/{chat_id}/documents",
    summary="List Chat Documents",
    status_code=status.HTTP_200_OK,
    response_model=list[ChatDocumentRead],
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def list_chat_documents(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/documents",
        method="GET",
        chat_id=str(chat_id),
    )
    try:
        service = ChatsService(db)
        result = await service.list_chat_documents(chat_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=200, duration_ms=round(duration, 2))
        return result
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise


@router.delete(
    "/{chat_id}/document/{document_id}",
    summary="Remove Chat Document",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def remove_chat_document(
    chat_id: UUID,
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/document/{document_id}",
        method="DELETE",
        chat_id=str(chat_id),
        document_id=str(document_id),
    )
    try:
        service = ChatsService(db)
        await service.remove_chat_document(chat_id, document_id)
        duration = (time.perf_counter() - start) * 1000
        logger.info("Request completed", status_code=204, duration_ms=round(duration, 2))
        return
    except HTTPException as e:
        duration = (time.perf_counter() - start) * 1000
        logger.warning("Request failed", status_code=e.status_code, detail=e.detail, duration_ms=round(duration, 2))
        raise
    except Exception:
        duration = (time.perf_counter() - start) * 1000
        logger.exception("Unhandled error", duration_ms=round(duration, 2))
        raise