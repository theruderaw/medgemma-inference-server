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
from app.logger import logger


class ChatsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_chats(self, skip: int, limit: int):
        logger.info("Listing chats", skip=skip, limit=limit)
        result = await self.db.exec(
            select(ChatSession)
            .order_by(ChatSession.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        chats = result.all()
        logger.info("Retrieved chat list", count=len(chats))
        return chats

    async def create_chat(self, title: str):
        logger.info("Creating chat", title=title)
        chat = ChatSession(
            chat_id=uuid4(),
            title=title,
        )
        self.db.add(chat)
        await self.db.commit()
        await self.db.refresh(chat)
        logger.info("Chat created", chat_id=str(chat.chat_id), title=title)
        return chat

    async def get_chat(self, chat_id: UUID):
        logger.info("Fetching chat", chat_id=str(chat_id))
        result = await self.db.exec(
            select(ChatSession)
            .where(ChatSession.chat_id == chat_id)
        )
        chat = result.first()
        if not chat:
            logger.warning("Chat not found", chat_id=str(chat_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )
        logger.info("Chat retrieved", chat_id=str(chat_id), title=chat.title)
        return chat

    async def edit_chat(self, chat_id: UUID, title: str):
        logger.info("Editing chat", chat_id=str(chat_id), title=title)
        chat = await self.get_chat(chat_id)
        chat.title = title
        await self.db.commit()
        await self.db.refresh(chat)
        logger.info("Chat edited", chat_id=str(chat_id), title=title)
        return chat

    async def delete_chat(self, chat_id: UUID):
        logger.info("Deleting chat", chat_id=str(chat_id))
        # get_chat raises 404 if not found, so no redundant check needed
        chat = await self.get_chat(chat_id)
        await self.db.delete(chat)
        await self.db.commit()
        logger.info("Chat deleted", chat_id=str(chat_id))

    async def _get_latest_summary(self, document_id: UUID) -> str | None:
        """Fetches the most recent analysis's summary for a document."""
        result = await self.db.exec(
            select(Analysis.summary)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        summary = result.first()
        if summary:
            logger.debug("Latest summary retrieved", document_id=str(document_id))
        return summary

    async def assign_chat_document(
        self,
        chat_id: UUID,
        document_id: UUID,
    ):
        logger.info(
            "Assigning document to chat",
            chat_id=str(chat_id),
            document_id=str(document_id),
        )
        chat = await self.db.get(ChatSession, chat_id)
        if not chat:
            logger.warning("Chat not found for document assignment", chat_id=str(chat_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        document = await self.db.get(Document, document_id)
        if document is None:
            logger.warning("Document not found for assignment", document_id=str(document_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found",
            )

        existing = await self.db.get(
            ChatDocument,
            (chat_id, document_id),
        )
        if existing is not None:
            logger.warning("Document already attached", chat_id=str(chat_id), document_id=str(document_id))
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
            logger.error(
                "Integrity error assigning document (duplicate)",
                chat_id=str(chat_id),
                document_id=str(document_id),
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already attached to this chat",
            )

        await self.db.refresh(chat_document)
        summary = await self._get_latest_summary(document_id)

        logger.info(
            "Document assigned to chat",
            chat_id=str(chat_id),
            document_id=str(document_id),
        )
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
        logger.info("Listing chat documents", chat_id=str(chat_id))
        chat = await self.db.get(ChatSession, chat_id)
        if not chat:
            logger.warning("Chat not found for document listing", chat_id=str(chat_id))
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

        logger.info(
            "Chat documents listed",
            chat_id=str(chat_id),
            count=len(summaries_out),
        )
        return summaries_out

    async def remove_chat_document(
        self,
        chat_id: UUID,
        document_id: UUID,
    ) -> None:
        logger.info(
            "Removing document from chat",
            chat_id=str(chat_id),
            document_id=str(document_id),
        )
        chat = await self.db.get(ChatSession, chat_id)
        if not chat:
            logger.warning("Chat not found for document removal", chat_id=str(chat_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found",
            )

        chat_document = await self.db.get(
            ChatDocument,
            (chat_id, document_id),
        )
        if chat_document is None:
            logger.warning("Document not attached to chat", chat_id=str(chat_id), document_id=str(document_id))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document is not attached to this chat",
            )

        document = await self.db.get(Document, document_id)
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

        logger.info(
            "Document removed from chat",
            chat_id=str(chat_id),
            document_id=str(document_id),
        )