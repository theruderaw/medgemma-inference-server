from datetime import datetime
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.features.chats.service import ChatsService
from app.inference.rag import RAGService
from app.models.chat import ChatMessage
from app.schemas.message import ChatMessageCreate, ChatMessageUpdate


class ChatMessagesService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_message_in_chat(
        self,
        chat_id: UUID,
        payload: ChatMessageCreate,
        background_tasks: BackgroundTasks,
    ):
        chat = await ChatsService(self.db).get_chat(chat_id)

        message = ChatMessage(
            chat_id=chat_id,
            role=payload.role,
            content=payload.content,
            message_metadata=payload.message_metadata,
        )

        self.db.add(message)

        # Bump parent chat's updated_at since a new message changes it.
        chat.updated_at = datetime.now()

        await self.db.commit()
        await self.db.refresh(message)

        background_tasks.add_task(
            RAGService.run,
            chat_id,
            message.content,
        )

        return message

    async def get_chat_messages(self, chat_id: UUID):
        await ChatsService(self.db).get_chat(chat_id)

        result = await self.db.exec(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
        )

        return result.all()

    async def get_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
    ):
        await ChatsService(self.db).get_chat(chat_id)

        result = await self.db.exec(
            select(ChatMessage).where(
                ChatMessage.message_id == message_id,
                ChatMessage.chat_id == chat_id,
            )
        )

        message = result.first()

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message deleted or didn't exist",
            )

        return message

    async def update_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
        payload: ChatMessageUpdate,
    ):
        message = await self.get_message_from_chat(
            chat_id,
            message_id,
        )

        # Only update fields explicitly supplied by the client.
        update_data = payload.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(message, field, value)

        await self.db.commit()
        await self.db.refresh(message)

        return message

    async def delete_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
    ):
        message = await self.get_message_from_chat(
            chat_id,
            message_id,
        )
        if not message:
            raise HTTPException(404,"Message not Found")

        self.db.delete(message)
        await self.db.commit()