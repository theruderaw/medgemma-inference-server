import time
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.chat_messages.service import ChatMessagesService
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_500,
)
from app.schemas.message import ChatMessageCreate, ChatMessageRead, ChatMessageUpdate
from app.logger import logger

router = APIRouter(
    prefix="/api/v1/chats",
    tags=["Messages"],
)


@router.post(
    "/{chat_id}/query",
    status_code=status.HTTP_201_CREATED,
    response_model=ChatMessageRead,
    responses={
        400: ERROR_400,
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def add_message_to_chat(
    chat_id: UUID,
    payload: ChatMessageCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/query",
        method="POST",
        chat_id=str(chat_id),
    )
    try:
        service = ChatMessagesService(db)
        result = await service.create_message_in_chat(
            chat_id,
            payload,
            background_tasks
        )
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
    "/{chat_id}/messages",
    response_model=list[ChatMessageRead],
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_chat_history(
    chat_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/messages",
        method="GET",
        chat_id=str(chat_id),
    )
    try:
        service = ChatMessagesService(db)
        result = await service.get_chat_messages(chat_id)
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
    "/{chat_id}/messages/{message_id}",
    response_model=ChatMessageRead,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_message_from_chat(
    chat_id: UUID,
    message_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/messages/{message_id}",
        method="GET",
        chat_id=str(chat_id),
        message_id=str(message_id),
    )
    try:
        service = ChatMessagesService(db)
        result = await service.get_message_from_chat(chat_id, message_id)
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


@router.patch(
    "/{chat_id}/messages/{message_id}",
    response_model=ChatMessageRead,
    responses={
        400: ERROR_400,
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def edit_message_by_id(
    chat_id: UUID,
    message_id: UUID,
    payload: ChatMessageUpdate,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/messages/{message_id}",
        method="PATCH",
        chat_id=str(chat_id),
        message_id=str(message_id),
    )
    try:
        service = ChatMessagesService(db)
        result = await service.update_message_from_chat(chat_id, message_id, payload)
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
    "/{chat_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def delete_message_from_chat(
    chat_id: UUID,
    message_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    start = time.perf_counter()
    logger.info(
        "Request received",
        path="/api/v1/chats/{chat_id}/messages/{message_id}",
        method="DELETE",
        chat_id=str(chat_id),
        message_id=str(message_id),
    )
    try:
        service = ChatMessagesService(db)
        await service.delete_message_from_chat(chat_id, message_id)
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