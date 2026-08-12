from uuid import UUID, uuid4

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.analysis import Analysis
from app.models.chat import ChatMessage, ChatSession
from app.models.chat_document import ChatDocument
from app.models.document import Document
from app.models.enums import MessageRole
from app.schemas.context import ChatDocumentRead


class ChatsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_chats(self, skip: int, limit: int):
        result = await self.db.exec(
            select(ChatSession)
            .order_by(ChatSession.created_at.desc())
            .offset(skip)
            .limit(limit)
        )

        return result.all()

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
        result = await self.db.exec(
            select(ChatSession)
            .where(ChatSession.chat_id == chat_id)
        )

        chat = result.first()

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
        if not chat:
            raise HTTPException(404, "Chat not found")
        
        await self.db.delete(chat)

        await self.db.commit()

    async def _get_latest_summary(self, document_id: UUID) -> str | None:
        """Fetches the most recent analysis's summary for a document, or
        None if no analysis exists yet (e.g. document just uploaded/attached
        before analysis has run or completed)."""
        result = await self.db.exec(
            select(Analysis.summary)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        return result.first()

    async def assign_chat_document(
        self,
        chat_id: UUID,
        document_id: UUID,
    ):
        chat = await self.db.get(ChatSession, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        document = await self.db.get(Document, document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        existing = await self.db.get(
            ChatDocument,
            (chat_id, document_id),
        )

        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already attached to this chat",
            )

        message = ChatMessage(
            chat_id=chat_id,
            role=MessageRole.SYSTEM,
            content=f"ADDED DOCUMENT {document.original_filename}",
        )

        self.db.add(message)

        await self.db.flush()

        chat_document = ChatDocument(
            chat_id=chat_id,
            document_id=document_id,
            message_id=message.message_id,
        )

        self.db.add(chat_document)

        try:
            await self.db.commit()

        except IntegrityError:
            await self.db.rollback()

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already attached to this chat",
            )

        await self.db.refresh(chat_document)

        summary = await self._get_latest_summary(document_id)

        return ChatDocumentRead(
            chat_id=chat.chat_id,
            title=chat.title,
            document_id=document.document_id,
            file_path=document.file_path,
            file_size=document.file_size,
            content_type=document.content_type,
            summary=summary,
            attached_at=chat_document.attached_at,
        )

    async def list_chat_documents(
        self,
        chat_id: UUID,
    ) -> list[ChatDocumentRead]:
        chat = await self.db.get(ChatSession, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        result = await self.db.exec(
            select(ChatDocument, Document)
            .join(
                Document,
                ChatDocument.document_id == Document.document_id,
            )
            .where(ChatDocument.chat_id == chat_id)
            .order_by(ChatDocument.attached_at.desc())
        )

        rows = result.all()

        summaries_out = []
        for chat_document, document in rows:
            summary = await self._get_latest_summary(document.document_id)
            summaries_out.append(
                ChatDocumentRead(
                    chat_id=chat.chat_id,
                    title=chat.title,
                    document_id=document.document_id,
                    file_path=document.file_path,
                    file_size=document.file_size,
                    content_type=document.content_type,
                    summary=summary,
                    attached_at=chat_document.attached_at,
                )
            )

        return summaries_out

    async def remove_chat_document(
        self,
        chat_id: UUID,
        document_id: UUID,
    ) -> None:
        chat = await self.db.get(ChatSession, chat_id)

        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        chat_document = await self.db.get(
            ChatDocument,
            (chat_id, document_id),
        )

        if chat_document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document is not attached to this chat",
            )

        document = await self.db.get(
            Document,
            document_id,
        )

        await self.db.delete(chat_document)

        message = ChatMessage(
            chat_id=chat_id,
            role=MessageRole.SYSTEM,
            content=(
                f"REMOVED DOCUMENT "
                f"{document.original_filename if document else document_id}"
            ),
        )

        self.db.add(message)

        await self.db.commit()