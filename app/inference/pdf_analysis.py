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

# Matches the "--- Page N ---" markers written by analyze()
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
        """Fallback: literal substring matching against known entity names."""
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
        """Extract summary section if present, otherwise use last paragraph."""
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
        """
        Recover (page_num, page_content) pairs from the joined raw_output
        produced by analyze(), which looks like:

            --- Page 1 ---
            <content>

            --- Page 2 ---
            <content>
        """
        parts = _PAGE_MARKER_RE.split(raw_output)
        # parts == ["", "1", "<content>", "2", "<content>", ...]
        pages = []
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                pages.append((page_num, content))
        return pages

    async def analyze(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if not analysis:
            raise ValueError("Analysis not found")

        document = await self.db.get(Document, analysis.document_id)

        if not document:
            raise ValueError("Document not found")

        pdf_bytes = Path(document.file_path).read_bytes()

        pages = extract_pdf_text(
            pdf_bytes,
            per_page=True,
            ocr_fallback=True,
            layout=True,
        )

        page_results = []

        for page_num, page_image in render_pdf_images(
            pdf_bytes,
            dpi=300,
            mode="b64",
        ):
            page_text = pages[page_num - 1]

            prompt = PDF_PAGE_ANALYSIS_PROMPT.format(
                page_text=page_text,
            )

            res = await chat(
                model=self.analysis_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [page_image],
                    }
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

    async def extract(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if analysis is None:
            raise ValueError("Analysis not found")

        if not analysis.raw_output:
            raise ValueError("Analysis has no raw output")

        analysis.status = AnalysisStatus.CHUNKING

        await self.db.commit()

        pages = self._split_pages(analysis.raw_output)

        if not pages:
            # Not page-marked output for some reason — treat as a single chunk
            pages = [(1, analysis.raw_output)]

        doc_summaries = []
        any_chunk_created = False

        for page_num, page_content in pages:
            summary = None
            entities = None
            notes = {}

            # --- Attempt 1: LLM extraction ---
            try:
                res = await chat(
                    model=self.text_model,
                    messages=[
                        {
                            "role": "system",
                            "content": EXTRACT_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": page_content,
                        },
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
                        pass  # Fall through to regex fallback

            except Exception:
                pass  # LLM call failed, fall through to regex fallback

            # --- Attempt 2: Regex fallback if LLM failed or returned empty ---
            if not summary:
                summary = self._extract_summary_fallback(page_content)

            if not entities:
                entities = self._regex_extract_entities(page_content)

            if not summary:
                # Skip an unusable page rather than failing the whole document
                continue

            doc_summaries.append(summary)

            chunk = Chunk(
                chunk_id=uuid4(),
                document_id=analysis.document_id,
                analysis_id=analysis_id,
                chunk_content=summary,
                chunk_index=page_num,
                page_number=page_num,
                chunk_type=ChunkType.TEXT,  # NOTE: adjust to your actual enum member
                entities=entities,
                notes=notes,
                embedding_model=settings.EMBED_MODEL,
            )

            self.db.add(chunk)
            any_chunk_created = True

        if not any_chunk_created:
            analysis.status = AnalysisStatus.FAILED
            await self.db.commit()
            return

        analysis.summary = "\n\n".join(doc_summaries)

        await self.db.commit()

    async def embed(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if analysis is None:
            raise ValueError("Analysis not found")

        analysis.status = AnalysisStatus.EMBEDDING

        result = await self.db.execute(
            select(Chunk).where(Chunk.analysis_id == analysis_id)
        )

        chunks = result.scalars().all()

        if not chunks:
            raise ValueError("No chunks found")

        embeddable = [c for c in chunks if c.chunk_content]
        if not embeddable:
            analysis.status = AnalysisStatus.COMPLETE
            await self.db.commit()
            return

        res = await ollama_embed(
            model=settings.EMBED_MODEL,
            input=[c.chunk_content for c in embeddable],
        )
        
        embeddings = res["embeddings"]

        if len(embeddable) != len(embeddings):
            raise ValueError(
                f"Embedding count mismatch: got {len(embeddings)} "
                f"for {len(embeddable)} chunks"
            )
            
        for chunk,embedding in zip(embeddable,embeddings):
            chunk.embedding = embedding
            chunk.embedding_model = settings.EMBED_MODEL
            
        analysis.status = AnalysisStatus.COMPLETE

        await self.db.commit()

    @staticmethod
    async def run(analysis_id: UUID):
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

            except Exception:
                traceback.print_exc()

                analysis = await db.get(Analysis, analysis_id)

                if analysis:
                    analysis.summary = ""
                    analysis.raw_output = ""
                    analysis.status = AnalysisStatus.FAILED

                    await db.commit()

                raise