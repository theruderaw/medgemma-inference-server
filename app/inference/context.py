from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import func
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
        query: str,
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
        semantic_chunks = await self._get_similar_chunks(query_embedding, top_k)
        
        lexical_chunks = await self._get_lexical_chunks(query,top_k)
        
        similar_chunks = self._merge_chunks(
            semantic_chunks,
            lexical_chunks,
            top_k
        )
        
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
        try:
            distance = Chunk.embedding.cosine_distance(query_embedding)
            result = await self.db.exec(
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
            rows = result.all()
            chunks = [
                SimilarChunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.chunk_content,
                    entities=row.entities,
                    similarity=1 - row.distance,
                )
                for row in rows
            ]
            logger.debug(
                "Similar chunks retrieved",
                top_k=top_k,
                count=len(chunks),
            )
            return chunks
        except Exception as e:
            logger.error(
                "Error fetching similar chunks",
                top_k=top_k,
                error=str(e),
            )
            raise
        
    async def _get_lexical_chunks(
        self,
        query: str,
        top_k: int
    ) -> list[SimilarChunk]:
        logger.debug(
            "Fetching lexical chunks",top_k = top_k
        )
        try:
            search_query = func.websearch_to_tsquery(
                "english",
                query
            )
            
            rank = func.ts_rank(
                Chunk.search_vector,search_query
            )
            
            result = await self.db.exec(
                select(
                    Chunk.chunk_id,
                    Chunk.document_id,
                    Chunk.chunk_content,
                    Chunk.entities,
                    rank.label("rank")
                )
                .where(
                    Chunk.search_vector.op("@@")(search_query)
                )
                .order_by(rank.desc())
                .limit(top_k)
            )
            
            rows = result.all()
            
            chunks = [
                SimilarChunk(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.chunk_content,
                    entities=row.entities,
                    similarity=float(row.rank),
                )
                for row in rows
            ]
            
            logger.debug(
                "Lexical chunks retrieved",
                count=len(chunks),
            )

            return chunks
            
        except Exception as e:
            logger.error(
                "Error fetching lexical chunks",
                top_k=top_k,
                error=str(e),
            )
            raise
    
    def _merge_chunks(
        self,
        semantic_chunks: list[SimilarChunk],
        lexical_chunks: list[SimilarChunk],
        top_k: int,
    ) -> list[SimilarChunk]:
        scores: dict[UUID, float] = {}
        chunks: dict[UUID, SimilarChunk] = {}

        k = 60

        for rank, chunk in enumerate(semantic_chunks, start=1):
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = (
                scores.get(chunk.chunk_id, 0.0)
                + 1 / (k + rank)
            )

        for rank, chunk in enumerate(lexical_chunks, start=1):
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = (
                scores.get(chunk.chunk_id, 0.0)
                + 1 / (k + rank)
            )

        ranked_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:top_k]

        return [chunks[chunk_id] for chunk_id in ranked_ids]
    async def _get_previous_document_summaries(
        self, chat_id: UUID
    ) -> list[DocumentSummaryContext]:
        logger.debug("Fetching previous document summaries", chat_id=str(chat_id))
        try:
            latest_analysis = (
                select(Analysis)
                .distinct(Analysis.document_id)
                .order_by(Analysis.document_id, Analysis.created_at.desc())
                .subquery()
            )
            latest_analysis_alias = aliased(Analysis, latest_analysis)

            result = await self.db.exec(
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
                logger.debug(
                    "No previous documents found for chat", chat_id=str(chat_id)
                )
                return []

            analysis_ids = [row.analysis_id for row in rows]
            logger.debug(
                "Fetching notes for previous analyses",
                chat_id=str(chat_id),
                analysis_count=len(analysis_ids),
            )
            chunk_result = await self.db.exec(
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
                        logger.debug(
                            "Skipping unrecognized note value",
                            chat_id=str(chat_id),
                            note=note,
                        )
                        continue

            contexts = [
                DocumentSummaryContext(
                    document_id=row.document_id,
                    summary=row.summary,
                    notes=list(notes_by_analysis.get(row.analysis_id, set())),
                )
                for row in rows
            ]
            logger.debug(
                "Previous document summaries retrieved",
                chat_id=str(chat_id),
                count=len(contexts),
            )
            return contexts
        except Exception as e:
            logger.error(
                "Error fetching previous document summaries",
                chat_id=str(chat_id),
                error=str(e),
            )
            raise

    async def _get_current_document_raw_output(
        self, document_id: UUID
    ) -> str | None:
        logger.debug("Fetching current document raw output", document_id=str(document_id))
        result = await self.db.exec(
            select(Analysis.raw_output)
            .where(Analysis.document_id == document_id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        row = result.first()
        raw = row.raw_output if row else None
        logger.debug("Current document raw output fetched", has_raw=raw is not None)
        return raw