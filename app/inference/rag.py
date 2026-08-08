from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.inference.llm import chat, embed
from app.core.config import settings
from app.inference.prompts import GENERATE_PROMPT, QUERY_PROMPT
from app.models.chat import ChatMessage
from app.models.chunk import Chunk
from app.models.enums import MessageRole

class RAGService:
    def __init__(self,db:AsyncSession):
        self.db = db
    async def retrieve(
        self,
        query: str,
        top_k: int = 4
    ):
        if not query or not query.strip():
            raise ValueError("Query can't be empty")

        res = await embed(
            model=settings.EMBED_MODEL,
            input=query
        )
        
        query_embedding = res["embeddings"][0]
        
        distance = Chunk.embedding.cosine_distance(
            query_embedding
        )
        
        statement = (
            select(
                Chunk,
                distance.label("distance")
            )
            .where(
                Chunk.embedding.is_not(None)
            )
            .order_by(distance)
            .limit(top_k)
        )

        result = await self.db.execute(statement)
        rows = result.all()
        
        return [
            {
                "chunk":chunk,
                "distance":float(distance)
            } for chunk,distance in rows
        ]
        
    async def augment(
        self,
        query: str,
        context: list[dict]
    ):
        contexts = [
            f"""
            CONTEXT {index}
            Content: {item["chunk"].chunk_content}
            Entities: {item["chunk"].entities}
            Notes: {item["chunk"].notes}
            """
            for index,item in enumerate(context,start=1)
        ]
        return QUERY_PROMPT.format(
            context=f"\n{'-'*40}\n".join(contexts),
            query=query
        )
        
    async def generate(
        self,
        chat_id: UUID,
        prompt: str
    ):
        res = await chat(
            model=settings.TEXT_MODEL,
            messages=[
                {
                    "role":"system",
                    "content":GENERATE_PROMPT
                },
                {
                    "role":"user",
                    "content":prompt
                }
            ],
        )
        
        message = ChatMessage(
            message_id=uuid4(),
            chat_id=chat_id,
            role= MessageRole.ASSISTANT,
            content = res["message"]["content"],
            message_metadata={}
        ) 
        
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        
        
    @staticmethod
    async def run(chat_id:UUID,query: str):
        async with AsyncSessionLocal() as db:
            service = RAGService(
                db=db
            )
        
            result = await service.retrieve(query, 4)
            prompt = await service.augment(query,result)
            await service.generate(chat_id,prompt)