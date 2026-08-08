from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.context.service import ChatContextService
from app.schemas.context import ChatDocumentRead
from app.schemas.errors import (
    ERROR_400,
    ERROR_404,
    ERROR_409,
    ERROR_500,
)

router = APIRouter(
    prefix="/chats",
    tags=["Context"],
)


@router.get(
    "/{chat_id}/context",
    response_model=list[ChatDocumentRead],
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def get_chat_context(
    chat_id: UUID,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    return await ChatContextService(db).get_chat_context(chat_id, skip, limit)


@router.post(
    "/{chat_id}/context",
    status_code=status.HTTP_201_CREATED,
    response_model=ChatDocumentRead | None,
    responses={
        400: ERROR_400,
        404: ERROR_404,
        409: ERROR_409,
        500: ERROR_500,
    },
)
async def add_chat_context(
    chat_id: UUID,
    document_id: UUID | None = Form(None),
    prompt: str | None = Form(None),
    image: UploadFile | None = File(None),
    db: AsyncSession = Depends(get_db),
):
    return await ChatContextService(db).add_chat_context(
        chat_id, document_id, prompt, image
    )


@router.delete(
    "/{chat_id}/context/{context_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: ERROR_404,
        500: ERROR_500,
    },
)
async def delete_chat_context(
    chat_id: UUID,
    context_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    await ChatContextService(db).delete_chat_context(chat_id, context_id)