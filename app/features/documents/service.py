import hashlib
import os
from uuid import UUID, uuid4

import aiofiles
# pyrefly: ignore [missing-import]
from fastapi import HTTPException, UploadFile
# pyrefly: ignore [missing-import]
from sqlalchemy import select
# pyrefly: ignore [missing-import]
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import audit
from app.core.config import settings
from app.models.document import Document
from app.models.analysis import Analysis
from app.models.enums import AnalysisStatus
from app.queue.redis_queue import RedisQueue
from app.queue.tasks import TaskEnvelope
from app.utils.pdf_ocr import validate_pdf_bytes
from app.logger import logger

ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/png",
    "application/pdf"
]

UPLOAD_DIR = "./uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 25 MB
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024  # 1 MB


class DocumentService:
    def __init__(self, db: AsyncSession, queue: RedisQueue | None = None):
        self.db = db
        # `queue` is only required for operations that enqueue work
        # (analyze_document). Other callers (list/get/delete) don't need it.
        self.queue = queue

    # ====================== DOCUMENT ======================================

    async def upload_document(self, file: UploadFile):
        logger.info(
            "Uploading document",
            filename=file.filename,
            content_type=file.content_type,
        )
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning("Unsupported content type", content_type=file.content_type)
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type",
            )

        contents = await self._read_upload_within_limit(file)
        if not contents:
            logger.warning("Empty file uploaded", filename=file.filename)
            raise HTTPException(
                status_code=400,
                detail="Empty file provided",
            )

        self._validate_file_bytes(contents, file.content_type)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        document_id = uuid4()
        ext = os.path.splitext(file.filename)[1]
        stored_filename = f"{document_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, stored_filename)

        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)
        except OSError as e:
            logger.error("File write error", error=str(e), file_path=file_path)
            raise HTTPException(
                status_code=400,
                detail="File writing error",
            )

        try:
            document = Document(
                document_id=document_id,
                original_filename=file.filename,
                stored_filename=stored_filename,
                file_path=file_path,
                content_type=file.content_type,
                file_size=len(contents),
                checksum=hashlib.sha256(contents).hexdigest()
            )
            self.db.add(document)
            
            await audit(
                db = self.db,
                event_type="document:upload",
                document_id=document_id,
                audit_metadata={
                    "filename":file.filename,
                    "content_type":file.content_type,
                    "file_size": len(contents)
                }
            )
            
            await self.db.commit()
            await self.db.refresh(document)

            logger.info(
                "Document uploaded successfully",
                document_id=str(document_id),
                file_size=len(contents),
                content_type=file.content_type,
            )
            return document

        except Exception as e:
            await self.db.rollback()
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error("Database error during document upload", error=str(e))
            raise HTTPException(
                status_code=500,
                detail="Failed to save document to database",
            )

    @staticmethod
    async def _read_upload_within_limit(file: UploadFile) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await file.read(UPLOAD_READ_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File too large",
                )
            chunks.append(chunk)
        await file.seek(0)
        return b"".join(chunks)

    @staticmethod
    def _validate_file_bytes(contents: bytes, content_type: str) -> None:
        if content_type == "application/pdf":
            try:
                validate_pdf_bytes(contents)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            DocumentService._validate_image_bytes(contents)

    @staticmethod
    def _validate_image_bytes(contents: bytes) -> None:
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(BytesIO(contents)) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(
                status_code=400,
                detail="File content does not match a valid image",
            )

    async def list_documents(self, skip: int, limit: int):
        logger.info("Listing documents", skip=skip, limit=limit)
        result = await self.db.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        docs = result.scalars().all()
        logger.info("Retrieved document list", count=len(docs))
        return docs

    async def get_document(self, document_id: UUID) -> Document:
        logger.info("Fetching document", document_id=str(document_id))
        result = await self.db.execute(
            select(Document)
            .where(Document.document_id == document_id)
        )
        document = result.scalars().first()
        if not document:
            logger.warning("Document not found", document_id=str(document_id))
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )
        logger.info("Document retrieved", document_id=str(document_id))
        return document

    async def delete_document(self, document_id: UUID):
        logger.info("Deleting document", document_id=str(document_id))
        document = await self.get_document(document_id)  # raises 404 if not found
        await audit(
                db = self.db,
                event_type="document:delete",
                document_id=document_id,
                audit_metadata={
                    "file":document.original_filename,
                    "content_type":document.content_type,
                    "file_size":document.file_size
                }
        )

        await self.db.delete(document)
        await self.db.commit()
        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)
        logger.info("Document deleted", document_id=str(document_id))

    # ===================== ANALYSIS =======================================

    async def list_analyses(self, document_id: UUID, skip: int, limit: int):
        logger.info("Listing analyses for document", document_id=str(document_id), skip=skip, limit=limit)
        document = await self.db.get(Document,document_id)
        if not document:
            raise HTTPException(
                404,"Document not found"
            )
        result = await self.db.execute(
            select(Analysis)
            .where(
                Analysis.document_id == document_id,
                Analysis.status != AnalysisStatus.DELETED
            )
            .order_by(Analysis.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        analyses = result.scalars().all()
        logger.info("Retrieved analysis list", document_id=str(document_id), count=len(analyses))
        return analyses

    IN_PROGRESS_ANALYSIS_STATUSES = (
        AnalysisStatus.ANALYZING,
        AnalysisStatus.CHUNKING,
        AnalysisStatus.EMBEDDING,
    )

    async def analyze_document(
        self,
        document_id: UUID,
    ):
        logger.info("Initiating document analysis", document_id=str(document_id))
        await self.get_document(document_id)

        existing = await self.db.execute(
            select(Analysis)
            .where(Analysis.document_id == document_id)
            .where(Analysis.status.in_(self.IN_PROGRESS_ANALYSIS_STATUSES))
        )
        if existing.scalars().first() is not None:
            logger.warning("Analysis already in progress", document_id=str(document_id))
            raise HTTPException(
                status_code=409,
                detail="Resource Conflict: an analysis is already in progress for this document",
            )

        analysis_id = uuid4()
        try:
            model_name, _, model_version = settings.ANALYSIS_MODEL.partition(":")
            model_version = model_version or "v1"
        except AttributeError:
            logger.error("ANALYSIS_MODEL not configured correctly")
            raise HTTPException(
                status_code=500,
                detail="ANALYSIS_MODEL is not configured correctly",
            )

        analysis = Analysis(
            analysis_id=analysis_id,
            document_id=document_id,
            model_name=model_name,
            model_version=model_version,
            status=AnalysisStatus.ANALYZING
        )
        document = await self.get_document(analysis.document_id)

        try:
            self.db.add(analysis)
            await audit(
                db = self.db,
                event_type="document:analysis",
                document_id=document_id,
                audit_metadata={
                    "file":document.original_filename,
                    "content_type":document.content_type,
                    "file_size":document.file_size
                }
            )
            await self.db.commit()
            await self.db.refresh(analysis)
        except IntegrityError:
            await self.db.rollback()
            logger.warning("Integrity error creating analysis (likely duplicate)", document_id=str(document_id))
            raise HTTPException(
                status_code=409,
                detail="Resource Conflict: an analysis is already in progress for this document",
            )
        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to create analysis record", error=str(e))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create analysis record: {type(e).__name__}",
            )

        if document.content_type in ("application/pdf", "image/png", "image/jpeg"):
            task = TaskEnvelope.for_analysis(
                analysis_id=analysis.analysis_id,
                document_type=document.content_type,
            )
            await self.queue.add(task)
            logger.info(
                "Queued analysis task",
                analysis_id=str(analysis_id),
                task_id=str(task.task_id),
                content_type=document.content_type,
            )
        else:
            logger.warning("Unsupported content type for analysis", content_type=document.content_type)

        logger.info("Analysis created and task queued", analysis_id=str(analysis_id), document_id=str(document_id))
        return analysis

    async def delete_document_analyses(self, document_id: UUID):
        logger.info("Deleting all analyses for document", document_id=str(document_id))
        await self.get_document(document_id)

        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.updated_at.desc())
        )
        analyses = result.scalars().all()
        if not analyses:
            return
        try:
            for analysis in analyses:
                await self.db.delete(analysis)
            await self.db.commit()
            logger.info("Deleted all analyses", document_id=str(document_id), count=len(analyses))
        except Exception as e:
            await self.db.rollback()
            logger.error("Failed to delete analyses", error=str(e))
            raise HTTPException(500, f"Failed to delete analyses: {type(e).__name__}")