import json
import re
import traceback
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.inference.llm import chat, embed as ollama_embed
from app.inference.prompts import EXTRACT_PROMPT, IMG_PROCESS_PROMPT
from app.inference.types import ChestXrayEntity
from app.models.analysis import Analysis
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import AnalysisStatus, ChunkType
from app.utils.pdf_ocr import extract_pdf_text, render_pdf_images
from app.logger import logger

EXTRACTION_ENTITIES = [entity.value for entity in ChestXrayEntity]


class ImageAnalysisService:
    def __init__(
        self,
        db: AsyncSession,
        analysis_model: str,
        text_model: str,
    ):
        self.db = db
        self.analysis_model = analysis_model or settings.ANALYSIS_MODEL
        self.text_model = text_model or settings.TEXT_MODEL

    @staticmethod
    def _clean_response(content: str) -> str:
        return re.sub(
            r".*?<unused95>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()

    @staticmethod
    def _regex_extract_entities(report_text: str) -> list[str]:
        report_lower = report_text.lower()
        found = []
        for entity in EXTRACTION_ENTITIES:
            pattern = rf"\b{re.escape(entity.lower())}\b"
            if re.search(pattern, report_lower):
                found.append(entity)
        return found

    @staticmethod
    def _extract_summary_fallback(raw_output: str) -> str:
        summary_match = re.search(
            r"(?:\*\*Summary:?\*\*|Summary:?)\s*\n?(.+?)(?:\n\n|\n\*\*|\Z)",
            raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            return summary_match.group(1).strip()
        paragraphs = [p.strip() for p in raw_output.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else raw_output.strip()

    async def analyse(self, analysis_id: UUID) -> None:
        logger.info("Starting image analysis", analysis_id=str(analysis_id))
        analysis = await self.db.get(Analysis, analysis_id)
        if analysis is None:
            logger.error("Analysis not found", analysis_id=str(analysis_id))
            raise ValueError("Analysis not found")

        document = await self.db.get(Document, analysis.document_id)
        if document is None:
            logger.error("Document not found", document_id=str(analysis.document_id))
            raise ValueError("Document not found")

        image = Path(document.file_path).read_bytes()
        logger.debug("Read image file", file_path=document.file_path, file_size=len(image))

        res = await chat(
            model=self.analysis_model,
            messages=[
                {"role": "user", "content": IMG_PROCESS_PROMPT}
            ],
            images=[image],
            stream=False,
        )

        content = self._clean_response(res["message"]["content"])
        analysis.raw_output = content
        analysis.summary = content
        analysis.prompt_template = IMG_PROCESS_PROMPT
        analysis.extract_prompt_template = EXTRACT_PROMPT

        await self.db.commit()
        logger.info("Image analysis completed", analysis_id=str(analysis_id))

    async def extract(self, analysis_id: UUID) -> None:
        logger.info("Starting extraction for analysis", analysis_id=str(analysis_id))
        analysis = await self.db.get(Analysis, analysis_id)
        if analysis is None:
            logger.error("Analysis not found for extraction", analysis_id=str(analysis_id))
            raise ValueError("Analysis not found")

        if not analysis.raw_output:
            logger.warning("No raw output available for extraction", analysis_id=str(analysis_id))
            raise ValueError("Analysis has no raw output")

        analysis.status = AnalysisStatus.CHUNKING
        await self.db.commit()

        summary = None
        entities = None
        notes = {}

        # Attempt LLM extraction
        try:
            res = await chat(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": analysis.raw_output},
                ],
                stream=False,
            )
            content = res["message"]["content"].strip()
            if content and content != "{}":
                try:
                    extracted = json.loads(content)
                    summary = extracted.get("summary")
                    entities = extracted.get("entities")
                    notes = extracted.get("notes", {})
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        # Fallback to regex if needed
        if not summary:
            summary = self._extract_summary_fallback(analysis.raw_output)
        if not entities:
            entities = self._regex_extract_entities(analysis.raw_output)

        if not summary or not entities:
            logger.warning(
                "Extraction failed (no summary or entities)",
                analysis_id=str(analysis_id),
                has_summary=bool(summary),
                has_entities=bool(entities),
            )
            analysis.status = AnalysisStatus.FAILED
            await self.db.commit()
            return

        analysis.summary = summary
        chunk = Chunk(
            chunk_id=uuid4(),
            document_id=analysis.document_id,
            analysis_id=analysis_id,
            chunk_content=summary,
            chunk_index=1,
            page_number=0,  
            chunk_type=ChunkType.IMAGE,
            entities=entities,
            notes=notes,
            embedding_model=settings.EMBED_MODEL,
        )
        self.db.add(chunk)
        await self.db.commit()
        logger.info(
            "Extraction successful",
            analysis_id=str(analysis_id),
            entity_count=len(entities),
        )

    async def embed(self, analysis_id: UUID) -> None:
        logger.info("Starting embedding for analysis", analysis_id=str(analysis_id))
        analysis = await self.db.get(Analysis, analysis_id)
        if analysis is None:
            logger.error("Analysis not found for embedding", analysis_id=str(analysis_id))
            raise ValueError("Analysis not found")

        analysis.status = AnalysisStatus.EMBEDDING
        await self.db.commit()

        result = await self.db.execute(
            select(Chunk).where(Chunk.analysis_id == analysis_id)
        )
        chunks = result.scalars().all()
        if not chunks:
            logger.warning("No chunks found for embedding", analysis_id=str(analysis_id))
            raise ValueError("No chunks found")

        for chunk in chunks:
            if not chunk.chunk_content:
                continue
            res = await ollama_embed(
                model=settings.EMBED_MODEL,
                input=chunk.chunk_content,
            )
            chunk.embedding = res["embeddings"][0]
            chunk.embedding_model = settings.EMBED_MODEL

        analysis.status = AnalysisStatus.COMPLETE
        await self.db.commit()
        logger.info(
            "Embedding completed",
            analysis_id=str(analysis_id),
            chunk_count=len(chunks),
        )

    @staticmethod
    async def run(analysis_id: UUID):
        logger.info("Running full image analysis pipeline", analysis_id=str(analysis_id))
        async with AsyncSessionLocal() as db:
            try:
                service = ImageAnalysisService(
                    db=db,
                    analysis_model=settings.ANALYSIS_MODEL,
                    text_model=settings.TEXT_MODEL,
                )
                await service.analyse(analysis_id)
                await service.extract(analysis_id)
                await service.embed(analysis_id)
                logger.info("Image analysis pipeline completed", analysis_id=str(analysis_id))
            except Exception as e:
                logger.error(
                    "Image analysis pipeline failed",
                    analysis_id=str(analysis_id),
                    error=str(e),
                    traceback=traceback.format_exc(),
                )
                analysis = await db.get(Analysis, analysis_id)
                if analysis:
                    analysis.summary = ""
                    analysis.raw_output = ""
                    analysis.status = AnalysisStatus.FAILED
                    await db.commit()
                raise