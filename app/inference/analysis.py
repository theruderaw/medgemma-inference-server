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
from app.inference.prompts import EXTRACT_PROMPT, SYS_PROMPT_INGESTION
from app.inference.types import ChestXrayEntity
from app.models.analysis import Analysis
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.enums import AnalysisStatus, ChunkType


# Fixed entity set for regex fallback
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
        """Fallback: literal substring matching against known entity names."""
        report_lower = report_text.lower()
        found = []

        for entity in EXTRACTION_ENTITIES:
            pattern = rf"\b{re.escape(entity.lower())}\b"

            if re.search(pattern, report_lower):
                found.append(entity)

    @staticmethod
    def _extract_summary_fallback(raw_output: str) -> str:
        """Extract summary section if present, otherwise use last paragraph."""
        # Try to find a "Summary:" or "**Summary:**" section
        summary_match = re.search(
            r"(?:\*\*Summary:?\*\*|Summary:?)\s*\n?(.+?)(?:\n\n|\n\*\*|\Z)",
            raw_output,
            re.DOTALL | re.IGNORECASE,
        )
        if summary_match:
            return summary_match.group(1).strip()

        # Fallback: take the last non-empty paragraph
        paragraphs = [p.strip() for p in raw_output.split("\n\n") if p.strip()]
        return paragraphs[-1] if paragraphs else raw_output.strip()

    async def analyse(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if analysis is None:
            raise ValueError("Analysis not found")

        await self.db.commit()

        document = await self.db.get(
            Document,
            analysis.document_id
        )

        if document is None:
            raise ValueError("Document not found")

        image = Path(document.file_path).read_bytes()

        res = await chat(
            model=self.analysis_model,
            messages=[
                {
                    "role": "user",
                    "content": SYS_PROMPT_INGESTION,
                }
            ],
            images=[image],
            stream=False,
        )

        content = self._clean_response(
            res["message"]["content"]
        )

        analysis.raw_output = content
        analysis.summary = content

        await self.db.commit()

    async def extract(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if analysis is None:
            raise ValueError("Analysis not found")

        if not analysis.raw_output:
            raise ValueError("Analysis has no raw output")

        analysis.status = AnalysisStatus.CHUNKING

        await self.db.commit()

        summary = None
        entities = None
        notes = {}

        # --- Attempt 1: LLM extraction (qwen) ---
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
                        "content": analysis.raw_output,
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
            summary = self._extract_summary_fallback(analysis.raw_output)

        if not entities:
            entities = self._regex_extract_entities(analysis.raw_output)

        # --- Save results ---
        if not summary or not entities:
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
            chunk_type=ChunkType.IMAGE,
            entities=entities,
            notes=notes,
            embedding_model=settings.EMBED_MODEL,
        )

        self.db.add(chunk)

        await self.db.commit()

    async def embed(self, analysis_id: UUID) -> None:
        analysis = await self.db.get(Analysis, analysis_id)

        if analysis is None:
            raise ValueError("Analysis not found")

        analysis.status = AnalysisStatus.EMBEDDING

        result = await self.db.execute(
            select(Chunk)
            .where(
                Chunk.analysis_id == analysis_id
            )
        )

        chunks = result.scalars().all()

        if not chunks:
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

    @staticmethod
    async def run(analysis_id: UUID):
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

            except Exception as e:
                traceback.print_exc()

                analysis = await db.get(
                    Analysis,
                    analysis_id,
                )

                if analysis:
                    analysis.summary = ""
                    analysis.raw_output = ""
                    analysis.status = AnalysisStatus.FAILED

                    await db.commit()

                raise