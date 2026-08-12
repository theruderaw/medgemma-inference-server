from datetime import datetime
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, status
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.features.chats.service import ChatsService
from app.inference.rag import RAGService
from app.models.chat import ChatMessage
from app.schemas.message import ChatMessageCreate, ChatMessageUpdate
from app.logger import logger


class ChatMessagesService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_message_in_chat(
        self,
        chat_id: UUID,
        payload: ChatMessageCreate,
        background_tasks: BackgroundTasks,
    ):
        logger.info("Creating message in chat", chat_id=str(chat_id), role=payload.role)
        chat = await ChatsService(self.db).get_chat(chat_id)

        message = ChatMessage(
            chat_id=chat_id,
            role=payload.role,
            content=payload.content,
            message_metadata=payload.message_metadata,
        )

        self.db.add(message)
        chat.updated_at = datetime.now()

        await self.db.commit()
        await self.db.refresh(message)

        background_tasks.add_task(
            RAGService.run,
            chat_id,
            message.content,
        )
        logger.info(
            "Message created and RAG task queued",
            chat_id=str(chat_id),
            message_id=str(message.message_id),
        )
        return message

    async def get_chat_messages(self, chat_id: UUID):
        logger.info("Fetching all messages for chat", chat_id=str(chat_id))
        await ChatsService(self.db).get_chat(chat_id)

        result = await self.db.exec(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat_id)
            .order_by(ChatMessage.created_at)
        )

        messages = result.all()
        logger.info(
            "Retrieved chat messages",
            chat_id=str(chat_id),
            count=len(messages),
        )
        return messages

    async def get_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
    ):
        logger.info(
            "Fetching single message from chat",
            chat_id=str(chat_id),
            message_id=str(message_id),
        )
        await ChatsService(self.db).get_chat(chat_id)

        result = await self.db.exec(
            select(ChatMessage).where(
                ChatMessage.message_id == message_id,
                ChatMessage.chat_id == chat_id,
            )
        )

        message = result.first()

        if not message:
            logger.warning(
                "Message not found in chat",
                chat_id=str(chat_id),
                message_id=str(message_id),
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message deleted or didn't exist",
            )

        logger.info(
            "Message retrieved",
            chat_id=str(chat_id),
            message_id=str(message_id),
        )
        return message

    async def update_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
        payload: ChatMessageUpdate,
    ):
        logger.info(
            "Updating message in chat",
            chat_id=str(chat_id),
            message_id=str(message_id),
        )
        message = await self.get_message_from_chat(
            chat_id,
            message_id,
        )

        update_data = payload.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(message, field, value)

        await self.db.commit()
        await self.db.refresh(message)

        logger.info(
            "Message updated",
            chat_id=str(chat_id),
            message_id=str(message_id),
            updated_fields=list(update_data.keys()),
        )
        return message

    async def delete_message_from_chat(
        self,
        chat_id: UUID,
        message_id: UUID,
    ):
        logger.info(
            "Deleting message from chat",
            chat_id=str(chat_id),
            message_id=str(message_id),
        )
        # get_message_from_chat already raises 404 if not found
        message = await self.get_message_from_chat(
            chat_id,
            message_id,
        )

        await self.db.delete(message)
        await self.db.commit()
        logger.info(
            "Message deleted",
            chat_id=str(chat_id),
            message_id=str(message_id),
        )