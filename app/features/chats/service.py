from uuid import UUID, uuid4

# pyrefly: ignore [missing-import]
from fastapi import HTTPException, status
# pyrefly: ignore [missing-import]
from sqlalchemy import select
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatSession


class ChatsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_chats(self, skip: int, limit: int):
        result = await self.db.execute(
            select(ChatSession)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def create_chat(self, title: str):
        chat = ChatSession(
            chat_id=uuid4(),
            title=title,
        )

        self.db.add(chat)
        await self.db.commit()
        await self.db.refresh(chat)

        return chat

    async def get_chat(self, chat_id: UUID):
        result = await self.db.execute(
            select(ChatSession).where(ChatSession.chat_id == chat_id)
        )

        chat = result.scalars().first()

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        return chat

    async def edit_chat(self, chat_id: UUID, title: str):
        chat = await self.get_chat(chat_id)

        chat.title = title

        await self.db.commit()
        await self.db.refresh(chat)

        return chat

    async def delete_chat(self, chat_id: UUID):
        chat = await self.get_chat(chat_id)

        await self.db.delete(chat)
        await self.db.commit()