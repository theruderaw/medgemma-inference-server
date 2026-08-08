from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.core.database import get_db
from app.features.chat_messages.service import ChatMessagesService
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_500,
)
from app.schemas.message import ChatMessageCreate, ChatMessageRead, ChatMessageUpdate
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(
    prefix="/chats",
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
    service = ChatMessagesService(db)
    return await service.create_message_in_chat(
        chat_id, 
        payload,
        background_tasks
    )


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
    service = ChatMessagesService(db)
    return await service.get_chat_messages(chat_id)


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
    service = ChatMessagesService(db)
    return await service.get_message_from_chat(chat_id, message_id)


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
    service = ChatMessagesService(db)
    return await service.update_message_from_chat(chat_id, message_id, payload)


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
    service = ChatMessagesService(db)
    await service.delete_message_from_chat(chat_id, message_id)