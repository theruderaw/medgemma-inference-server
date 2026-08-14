from collections import defaultdict
from uuid import UUID, uuid4

from sqlalchemy.orm import aliased
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.inference.context import ContextBundle, ContextEngine
from app.inference.llm import chat, embed
from app.inference.prompts import GENERATE_PROMPT, QUERY_PROMPT
from app.inference.types import ChestXrayEntity
from app.models.analysis import Analysis
from app.models.chat import ChatMessage
from app.models.chat_document import ChatDocument
from app.models.chunk import Chunk
from app.models.enums import MessageRole
from app.schemas.context import ChatDocumentSummary
from app.logger import logger


class RAGService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.context_engine = ContextEngine(db)

    # ---- query embedding -------------------------------------------------

    async def embed_query(self, query: str) -> list[float]:
        if not query or not query.strip():
            logger.warning("Empty query provided for embedding")
            raise ValueError("Query can't be empty")

        logger.debug("Embedding query", query_length=len(query))
        res = await embed(model=settings.EMBED_MODEL, input=query)
        embedding = res["embeddings"][0]
        logger.debug("Query embedding completed", embedding_dim=len(embedding))
        return embedding

    # ---- prompt assembly --------------------------------------------------

    async def augment(self, query: str, context: ContextBundle) -> str:
        logger.debug("Augmenting query with context", query_length=len(query))
        similar_chunks_section = "\n".join(
            f"""
            Chunk {index}:
            Content: {chunk.content}
            Entities: {chunk.entities}
            """
            for index, chunk in enumerate(context.similar_chunks, start=1)
        ) or "None"

        previous_docs_section = "\n".join(
            f"- Document {doc.document_id}: {doc.summary or 'No summary'} "
            f"(Findings: {', '.join(n.value for n in doc.notes) or 'None'})"
            for doc in context.previous_documents
        ) or "None"

        current_doc_section = context.current_document_raw_output or "None"

        combined_context = f"""
        Similar Chunks:
        {similar_chunks_section}

        Previously Attached Documents:
        {previous_docs_section}

        Current Document (raw analysis output):
        {current_doc_section}
        """

        prompt = QUERY_PROMPT.format(context=combined_context, query=query)
        logger.debug("Prompt assembled", prompt_length=len(prompt))
        return prompt

    # ---- generation ---------------------------------------------------

    async def generate(self, chat_id: UUID, prompt: str):
        logger.info("Generating response for chat", chat_id=str(chat_id))
        res = await chat(
            model=settings.TEXT_MODEL,
            messages=[
                {"role": "system", "content": GENERATE_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        message = ChatMessage(
            message_id=uuid4(),
            chat_id=chat_id,
            role=MessageRole.ASSISTANT,
            content=res["message"]["content"],
            message_metadata={},
        )

        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        logger.info(
            "Response generated and saved",
            chat_id=str(chat_id),
            message_id=str(message.message_id),
            content_length=len(message.content),
        )

    # ---- entry points -----------------------------------------------------

    @staticmethod
    async def run(
        chat_id: UUID,
        query: str,
        current_document_id: UUID | None = None,
    ):
        logger.info(
            "Running RAG pipeline",
            chat_id=str(chat_id),
            query_length=len(query),
            has_current_doc=current_document_id is not None,
        )
        async with AsyncSessionLocal() as db:
            service = RAGService(db=db)

            # Small-talk guard
            simple_greetings = {
                "hi", "hello", "hey", "hey there", "hi there", "good morning",
                "good afternoon", "good evening", "howdy", "yo", "sup"
            }
            normalized_query = query.strip().lower().rstrip("!.?")
            if normalized_query in simple_greetings:
                logger.info("Small talk detected, responding directly", chat_id=str(chat_id))
                message = ChatMessage(
                    message_id=uuid4(),
                    chat_id=chat_id,
                    role=MessageRole.ASSISTANT,
                    content="Hello! I can help answer questions about your medical documents. Please attach a document and ask a medical question.",
                    message_metadata={},
                )
                db.add(message)
                await db.commit()
                await db.refresh(message)
                return

            query_embedding = await service.embed_query(query)
            context = await service.context_engine.build_context(
                chat_id=chat_id,
                query=query,
                query_embedding=query_embedding,
                current_document_id=current_document_id,
            )
            prompt = await service.augment(query, context)
            await service.generate(chat_id, prompt)
            logger.info("RAG pipeline completed", chat_id=str(chat_id))
            
            
    @staticmethod
    async def getDocumentSummary(chat_id: UUID) -> list[ChatDocumentSummary]:
        logger.info("Fetching document summaries for chat", chat_id=str(chat_id))
        async with AsyncSessionLocal() as db:
            try:
                latest_analysis = (
                    select(Analysis)
                    .distinct(Analysis.document_id)
                    .order_by(
                        Analysis.document_id,
                        Analysis.created_at.desc()
                    )
                    .subquery()
                )

                latest_analysis_alias = aliased(Analysis, latest_analysis)

                result = await db.exec(
                    select(
                        ChatDocument.chat_id,
                        ChatDocument.document_id,
                        ChatDocument.attached_at,
                        latest_analysis_alias.analysis_id,
                        latest_analysis_alias.summary,
                        latest_analysis_alias.status,
                    )
                    .join(
                        latest_analysis_alias,
                        latest_analysis_alias.document_id == ChatDocument.document_id,
                    )
                    .where(ChatDocument.chat_id == chat_id)
                )

                rows = result.all()

                if not rows:
                    logger.info("No document summaries found for chat", chat_id=str(chat_id))
                    return []

                analysis_ids = [row.analysis_id for row in rows]

                chunk_result = await db.exec(
                    select(Chunk.analysis_id, Chunk.chunk_id, Chunk.notes)
                    .where(Chunk.analysis_id.in_(analysis_ids))
                )

                chunk_rows = chunk_result.all()

                chunks_by_analysis: dict[UUID, list[UUID]] = defaultdict(list)
                notes_by_analysis: dict[UUID, set[ChestXrayEntity]] = defaultdict(set)

                for chunk_row in chunk_rows:
                    chunks_by_analysis[chunk_row.analysis_id].append(
                        chunk_row.chunk_id
                    )

                    for note in (chunk_row.notes or []):
                        try:
                            notes_by_analysis[chunk_row.analysis_id].add(
                                ChestXrayEntity(note)
                            )
                        except ValueError:
                            continue

                summaries = [
                    ChatDocumentSummary(
                        chat_id=row.chat_id,
                        document_id=row.document_id,
                        analysis_id=row.analysis_id,
                        summary=row.summary,
                        status=row.status,
                        attached_at=row.attached_at,
                        chunks=chunks_by_analysis.get(row.analysis_id, []),
                        notes=list(
                            notes_by_analysis.get(row.analysis_id, set())
                        ),
                    )
                    for row in rows
                ]

                logger.info(
                    "Document summaries fetched",
                    chat_id=str(chat_id),
                    count=len(summaries),
                )
                return summaries

            except Exception as e:
                logger.error(
                    "Error fetching document summaries",
                    chat_id=str(chat_id),
                    error=str(e),
                )
                raise