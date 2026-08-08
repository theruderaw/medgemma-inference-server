import os
from uuid import UUID, uuid4

import aiofiles
# pyrefly: ignore [missing-import]
from fastapi import HTTPException, UploadFile, BackgroundTasks
# pyrefly: ignore [missing-import]
from sqlalchemy import select
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.analysis import Analysis
from app.models.document import Document

from app.inference.analysis import ImageAnalysisService

ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/png",
]

UPLOAD_DIR = "./uploads"

# A generous but bounded cap -- prevents unbounded memory use from
# `await file.read()` buffering an arbitrarily large upload in full.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ====================== DOCUMENT ======================================

    async def upload_document(self, file: UploadFile):
        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported Media Type",
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file provided",
            )

        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large",
            )

        # Content-type header is client-supplied and not trustworthy on its
        # own -- verify the bytes actually decode as an image before we
        # persist them and later feed them into cv2/PIL/YOLO.
        self._validate_image_bytes(contents)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        document_id = uuid4()
        ext = os.path.splitext(file.filename)[1]
        stored_filename = f"{document_id}{ext}"
        file_path = os.path.join(UPLOAD_DIR, stored_filename)

        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(contents)

        except OSError:
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
            )

            self.db.add(document)
            await self.db.commit()
            await self.db.refresh(document)

            return document

        except Exception:
            await self.db.rollback()

            if os.path.exists(file_path):
                os.remove(file_path)

            raise HTTPException(
                status_code=500,
                detail="Failed to save document to database",
            )

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
        result = await self.db.execute(
            select(Document)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_document(self, document_id: UUID):
        result = await self.db.execute(
            select(Document)
            .where(Document.document_id == document_id)
        )

        document = result.scalars().first()

        if not document:
            raise HTTPException(
                status_code=404,
                detail="Document not found",
            )

        return document

    async def delete_document(self, document_id: UUID):
        document = await self.get_document(document_id)

        # Commit the DB delete first, then remove the file. If the file
        # removal fails after a successful commit, at worst you have an
        # orphaned file on disk (recoverable via cleanup job) rather than
        # an orphaned DB row pointing at a file that no longer exists.
        await self.db.delete(document)
        await self.db.commit()

        if document.file_path and os.path.exists(document.file_path):
            os.remove(document.file_path)

    # ===================== ANALYSIS =======================================

    async def list_analyses(self, document_id: UUID, skip: int, limit: int):
        await self.get_document(document_id)

        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )

        return result.scalars().all()


    async def analyze_document(
        self,
        document_id: UUID,
        background_tasks: BackgroundTasks,
    ):
        await self.get_document(document_id)

        analysis_id = uuid4()

        try:
            model_name, _, model_version = settings.ANALYSIS_MODEL.partition(":")
            model_version = model_version or "v1"
        except AttributeError:
            raise HTTPException(
                status_code=500,
                detail="ANALYSIS_MODEL is not configured correctly",
            )

        analysis = Analysis(
            analysis_id=analysis_id,
            document_id=document_id,
            model_name=model_name,
            model_version=model_version,
        )

        try:
            self.db.add(analysis)
            await self.db.commit()
            await self.db.refresh(analysis)
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create analysis record: {type(e).__name__}",
            )

        background_tasks.add_task(
            ImageAnalysisService.run,
            analysis.analysis_id,
        )

        return analysis
    async def delete_document_analyses(self, document_id: UUID):
        await self.get_document(document_id)

        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.updated_at.desc())
        )
        analyses = result.scalars().all()
        if not analyses:
            raise HTTPException(404, "Document doesn't have any analyses")
        try:
            for analysis in analyses:
                await self.db.delete(analysis)
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(500, f"Failed to delete analyses: {type(e).__name__}")