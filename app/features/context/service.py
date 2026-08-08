from datetime import datetime
from uuid import UUID, uuid4

from fastapi import HTTPException, Form, UploadFile, File
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.chats.service import ChatsService
from app.features.documents.service import DocumentService
from app.models.chat import ChatMessage
from app.models.chat_document import ChatDocument
from app.models.enums import MessageRole
from app.schemas.context import ChatDocumentRead


class ChatContextService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_chat_context(
        self, chat_id: UUID, skip: int, limit: int
    ) -> list[ChatDocumentRead]:
        chat = await ChatsService(self.db).get_chat(chat_id)
        if not chat:
            raise HTTPException(400, "Chat not found")
        try:
            result = await self.db.execute(
                select(ChatDocument)
                .where(ChatDocument.chat_id == chat_id)
                .options(
                    selectinload(ChatDocument.chat),
                    selectinload(ChatDocument.document),
                )
                .offset(skip)
                .limit(limit)
            )
            chat_docs = result.scalars().all()
            return [
                ChatDocumentRead(
                    chat_id=cd.chat_id,
                    title=cd.chat.title,
                    document_id=cd.document_id,
                    file_path=cd.document.file_path,
                    file_size=cd.document.file_size,
                    content_type=cd.document.content_type,
                    attached_at=cd.attached_at,
                )
                for cd in chat_docs
            ]
        except Exception as e:
            raise HTTPException(500, f"{e}")

    async def add_chat_context(
        self,
        chat_id: UUID,
        document_id: UUID | None = Form(None),
        prompt: str | None = Form(None),
        image: UploadFile | None = File(None),
    ) -> ChatDocumentRead | None:
        chat = await ChatsService(self.db).get_chat(chat_id)
        if not chat:
            raise HTTPException(400, "Chat not found")

        if image and document_id:
            raise HTTPException(409, "Can't analyse two documents simultaneously")
        if not image and not document_id and not prompt:
            raise HTTPException(400, "Provide at least a prompt, an image, or a document_id")

        doc_service = DocumentService(self.db)
        document = None
        if image:
            document = await doc_service.upload_document(image)
            analysis = await doc_service.analyze_document(document.document_id)
        elif document_id:
            document = await doc_service.get_document(document_id)
            if not document:
                raise HTTPException(404, "Document not found")

        message = ChatMessage(
            message_id=uuid4(),
            chat_id=chat_id,
            role=MessageRole.USER,
            content=prompt if prompt else f"[SYSTEM]:Context inserted {document.document_id}",
        )
        self.db.add(message)

        chat_document = None
        if document:
            chat_document = ChatDocument(
                chat_id=chat_id,
                document_id=document.document_id,
                message_id=message.message_id,
            )
            self.db.add(chat_document)

        try:
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(500, f"{e}")

        if not chat_document:
            return None

        await self.db.refresh(chat_document)

        return ChatDocumentRead(
            chat_id=chat_document.chat_id,
            title=chat.title,
            document_id=document.document_id,
            file_path=document.file_path,
            file_size=document.file_size,
            content_type=document.content_type,
            attached_at=chat_document.attached_at,
        )

    async def delete_chat_context(self, chat_id: UUID, document_id: UUID) -> None:
        chat = await ChatsService(self.db).get_chat(chat_id)
        if not chat:
            raise HTTPException(400, "Chat not found")

        chat_document = await self.db.get(ChatDocument, (chat_id, document_id))
        if not chat_document:
            raise HTTPException(404, "Document not attached to this chat")

        try:
            await self.db.delete(chat_document)
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(500, f"{e}")