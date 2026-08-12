from dataclasses import dataclass, field
from uuid import UUID

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import aliased
from sqlmodel import select

from app.models.analysis import Analysis
from app.models.chat_document import ChatDocument
from app.models.chunk import Chunk
from app.inference.types import ChestXrayEntity
from app.logger import logger


@dataclass
class SimilarChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    entities: list
    similarity: float


@dataclass
class DocumentSummaryContext:
    document_id: UUID
    summary: str | None
    notes: list[ChestXrayEntity] = field(default_factory=list)


@dataclass
class ContextBundle:
    similar_chunks: list[SimilarChunk]
    previous_documents: list[DocumentSummaryContext]
    current_document_raw_output: str | None


class ContextEngine:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def build_context(
        self,
        chat_id: UUID,
        query_embedding: list[float],
        current_document_id: UUID | None = None,
        top_k: int = 5,
    ) -> ContextBundle:
        logger.info(
            "Building context for chat",
            chat_id=str(chat_id),
            top_k=top_k,
            has_current_doc=current_document_id is not None,
        )
        similar_chunks = await self._get_similar_chunks(query_embedding, top_k)
        previous_documents = await self._get_previous_document_summaries(chat_id)
        current_document_raw_output = (
            await self._get_current_document_raw_output(current_document_id)
            if current_document_id
            else None
        )

        bundle = ContextBundle(
            similar_chunks=similar_chunks,
            previous_documents=previous_documents,
            current_document_raw_output=current_document_raw_output,
        )
        logger.info(
            "Context built",
            similar_chunk_count=len(similar_chunks),
            previous_doc_count=len(previous_documents),
            has_raw_output=current_document_raw_output is not None,
        )
        return bundle

    async def _get_similar_chunks(
        self, query_embedding: list[float], top_k: int
    ) -> list[SimilarChunk]:
        logger.debug("Fetching similar chunks", top_k=top_k)
        distance = Chunk.embedding.cosine_distance(query_embedding)
        result = await self.db.execute(
            select(
                Chunk.chunk_id,
                Chunk.document_id,
                Chunk.chunk_content,
                Chunk.entities,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(top_k)
        )
        chunks = [
            SimilarChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.chunk_content,
                entities=row.entities,
                similarity=1 - row.distance,
            )
            for row in result.all()
        ]
        logger.debug("Similar chunks retrieved", count=len(chunks))
        return chunks

    async def _get_previous_document_summaries(
        self, chat_id: UUID
    ) -> list[DocumentSummaryContext]:
        logger.debug("Fetching previous document summaries", chat_id=str(chat_id))
        latest_analysis = (
            select(Analysis)
            .distinct(Analysis.document_id)
            .order_by(Analysis.document_id, Analysis.created_at.desc())
            .subquery()
        )
        latest_analysis_alias = aliased(Analysis, latest_analysis)

        result = await self.db.execute(
            select(
                ChatDocument.document_id,
                latest_analysis_alias.analysis_id,
                latest_analysis_alias.summary,
            )
            .join(
                latest_analysis_alias,
                latest_analysis_alias.document_id == ChatDocument.document_id,
            )
            .where(ChatDocument.chat_id == chat_id)
        )
        rows = result.all()
        if not rows:
            logger.debug("No previous documents found for chat", chat_id=str(chat_id))
            return []

        analysis_ids = [row.analysis_id for row in rows]
        chunk_result = await self.db.execute(
            select(Chunk.analysis_id, Chunk.notes).where(
                Chunk.analysis_id.in_(analysis_ids)
            )
        )

        notes_by_analysis: dict[UUID, set[ChestXrayEntity]] = {}
        for chunk_row in chunk_result.all():
            bucket = notes_by_analysis.setdefault(chunk_row.analysis_id, set())
            for note in chunk_row.notes or []:
                try:
                    bucket.add(ChestXrayEntity(note))
                except ValueError:
                    continue

        contexts = [
            DocumentSummaryContext(
                document_id=row.document_id,
                summary=row.summary,
                notes=list(notes_by_analysis.get(row.analysis_id, set())),
            )
            for row in rows
        ]
        logger.debug("Previous document summaries retrieved", count=len(contexts))
        return contexts

    async def _get_current_document_raw_output(
        self, document_id: UUID
    ) -> str | None:
        logger.debug("Fetching current document raw output", document_id=str(document_id))
        result = await self.db.execute(
            select(Analysis.raw_output)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        row = result.first()
        raw = row.raw_output if row else None
        logger.debug("Current document raw output fetched", has_raw=raw is not None)
        return raw