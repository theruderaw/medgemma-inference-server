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
from app.inference.prompts import EXTRACT_PROMPT, PDF_PAGE_ANALYSIS_PROMPT
from app.models.analysis import Analysis
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import AnalysisStatus, ChunkType
from app.utils.pdf_ocr import extract_pdf_text, render_pdf_images
from app.logger import logger

_PAGE_MARKER_RE = re.compile(r"--- Page (\d+) ---\n?")


class PDFAnalysisService:
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
        from app.inference.types import ChestXrayEntity
        entities = [e.value for e in ChestXrayEntity]
        report_lower = report_text.lower()
        found = []
        for entity in entities:
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

    @staticmethod
    def _split_pages(raw_output: str) -> list[tuple[int, str]]:
        parts = _PAGE_MARKER_RE.split(raw_output)
        pages = []
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                pages.append((page_num, content))
        return pages

    async def analyze(self, analysis_id: UUID) -> None:
        logger.info("Starting PDF analysis", analysis_id=str(analysis_id))
        analysis = await self.db.get(Analysis, analysis_id)
        if not analysis:
            logger.error("Analysis not found", analysis_id=str(analysis_id))
            raise ValueError("Analysis not found")

        document = await self.db.get(Document, analysis.document_id)
        if not document:
            logger.error("Document not found", document_id=str(analysis.document_id))
            raise ValueError("Document not found")

        pdf_bytes = Path(document.file_path).read_bytes()
        logger.debug("Read PDF file", file_path=document.file_path, file_size=len(pdf_bytes))

        pages = extract_pdf_text(
            pdf_bytes,
            per_page=True,
            ocr_fallback=True,
            layout=True,
        )
        logger.debug("Extracted text from PDF", page_count=len(pages))

        page_results = []
        for page_num, page_image in render_pdf_images(
            pdf_bytes,
            dpi=300,
            mode="b64",
        ):
            page_text = pages[page_num - 1]
            prompt = PDF_PAGE_ANALYSIS_PROMPT.format(page_text=page_text)
            res = await chat(
                model=self.analysis_model,
                messages=[
                    {"role": "user", "content": prompt, "images": [page_image]}
                ],
                stream=False,
            )
            content = self._clean_response(res["message"]["content"])
            page_results.append(f"--- Page {page_num} ---\n{content}")

        analysis.raw_output = "\n\n".join(page_results)
        analysis.summary = analysis.raw_output
        analysis.prompt_template = PDF_PAGE_ANALYSIS_PROMPT
        analysis.extract_prompt_template = EXTRACT_PROMPT

        await self.db.commit()
        logger.info(
            "PDF analysis completed",
            analysis_id=str(analysis_id),
            page_count=len(page_results),
        )

    async def extract(self, analysis_id: UUID) -> None:
        logger.info("Starting extraction for PDF analysis", analysis_id=str(analysis_id))
        analysis = await self.db.get(Analysis, analysis_id)
        if analysis is None:
            logger.error("Analysis not found for extraction", analysis_id=str(analysis_id))
            raise ValueError("Analysis not found")

        if not analysis.raw_output:
            logger.warning("No raw output available for extraction", analysis_id=str(analysis_id))
            raise ValueError("Analysis has no raw output")

        analysis.status = AnalysisStatus.CHUNKING
        await self.db.commit()

        pages = self._split_pages(analysis.raw_output)
        if not pages:
            logger.warning("No page-marked output, treating as single chunk", analysis_id=str(analysis_id))
            pages = [(1, analysis.raw_output)]

        doc_summaries = []
        any_chunk_created = False
        failed_pages = 0

        for page_num, page_content in pages:
            summary = None
            entities = None
            notes = {}

            # Attempt LLM extraction
            try:
                res = await chat(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": EXTRACT_PROMPT},
                        {"role": "user", "content": page_content},
                    ],
                    stream=False,
                    format_json=True,
                )
                content = res["message"]["content"].strip()
                if content and content != "{}":
                    try:
                        extracted = json.loads(content)
                        summary = extracted.get("summary")
                        entities = extracted.get("entities")
                        notes = extracted.get("notes", {})
                    except json.JSONDecodeError:
                        logger.debug("JSON decode error in LLM extraction, falling back", page=page_num)
            except Exception:
                logger.debug("LLM extraction failed, falling back to regex", page=page_num)

            # Fallback to regex
            if not summary:
                summary = self._extract_summary_fallback(page_content)
            if not entities:
                entities = self._regex_extract_entities(page_content)

            if not summary:
                logger.warning(
                    "No summary for page, skipping chunk",
                    analysis_id=str(analysis_id),
                    page=page_num,
                )
                failed_pages += 1
                continue

            doc_summaries.append(summary)
            chunk = Chunk(
                chunk_id=uuid4(),
                document_id=analysis.document_id,
                analysis_id=analysis_id,
                chunk_content=summary,
                chunk_index=page_num,
                page_number=page_num,
                chunk_type=ChunkType.TEXT,
                entities=entities,
                notes=notes,
                embedding_model=settings.EMBED_MODEL,
            )
            self.db.add(chunk)
            any_chunk_created = True

        if not any_chunk_created:
            logger.warning(
                "No chunks created during extraction, marking as failed",
                analysis_id=str(analysis_id),
                total_pages=len(pages),
            )
            analysis.status = AnalysisStatus.FAILED
            await self.db.commit()
            return

        analysis.summary = "\n\n".join(doc_summaries)
        await self.db.commit()
        logger.info(
            "Extraction completed",
            analysis_id=str(analysis_id),
            chunks_created=len(doc_summaries),
            failed_pages=failed_pages,
        )

    async def embed(self, analysis_id: UUID) -> None:
        logger.info("Starting embedding for PDF analysis", analysis_id=str(analysis_id))
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

        embeddable = [c for c in chunks if c.chunk_content]
        if not embeddable:
            logger.warning("No embeddable chunks (empty content)", analysis_id=str(analysis_id))
            analysis.status = AnalysisStatus.COMPLETE
            await self.db.commit()
            return

        res = await ollama_embed(
            model=settings.EMBED_MODEL,
            input=[c.chunk_content for c in embeddable],
        )
        embeddings = res["embeddings"]
        if len(embeddable) != len(embeddings):
            logger.error(
                "Embedding count mismatch",
                analysis_id=str(analysis_id),
                expected=len(embeddable),
                received=len(embeddings),
            )
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} "
                f"for {len(embeddable)} chunks"
            )

        for chunk, embedding in zip(embeddable, embeddings):
            chunk.embedding = embedding
            chunk.embedding_model = settings.EMBED_MODEL

        analysis.status = AnalysisStatus.COMPLETE
        await self.db.commit()
        logger.info(
            "Embedding completed",
            analysis_id=str(analysis_id),
            chunks_embedded=len(embeddable),
        )

    @staticmethod
    async def run(analysis_id: UUID):
        logger.info("Running full PDF analysis pipeline", analysis_id=str(analysis_id))
        async with AsyncSessionLocal() as db:
            try:
                service = PDFAnalysisService(
                    db=db,
                    analysis_model=settings.ANALYSIS_MODEL,
                    text_model=settings.TEXT_MODEL,
                )
                await service.analyze(analysis_id)
                await service.extract(analysis_id)
                await service.embed(analysis_id)
                logger.info("PDF analysis pipeline completed", analysis_id=str(analysis_id))
            except Exception as e:
                logger.error(
                    "PDF analysis pipeline failed",
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