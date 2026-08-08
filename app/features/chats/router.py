from app.core.database import get_db
# pyrefly: ignore [missing-import]
from fastapi import APIRouter,status,Path,Query,Depends
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.schemas.chat import ChatCreate, ChatRead, ChatUpdate
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_500,
)
from app.features.chats.service import ChatsService

router = APIRouter(
    prefix="/chats",
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
    service = ChatsService(db)
    return await service.create_chat(title=payload.title)


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
    service = ChatsService(db)
    return await service.list_chats(
        (page_no-1)*page_size,
        page_size
    )


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
    service = ChatsService(db)
    return await service.get_chat(chat_id)


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
    service = ChatsService(db)
    return await service.edit_chat(
        chat_id=chat_id,
        title=payload.title
    )


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
    service = ChatsService(db)
    await service.delete_chat(chat_id)
    