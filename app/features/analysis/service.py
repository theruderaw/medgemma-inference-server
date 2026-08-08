from uuid import UUID

# pyrefly: ignore [missing-import]
from fastapi import HTTPException
# pyrefly: ignore [missing-import]
from sqlalchemy import select
# pyrefly: ignore [missing-import]
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.analysis import Analysis
from app.schemas.analysis import AnalysisMetadata, AnalysisRead, AnalysisStatusResponse


class AnalysisService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_analysis(self, analysis_id: UUID):
        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
            .options(selectinload(Analysis.chunks))
        )

        analysis = result.scalars().first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not present")

        return AnalysisRead(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            raw_output=analysis.raw_output,
            summary=analysis.summary,
            entities=list(set().union(*(chunk.entities for chunk in analysis.chunks))),
            status=analysis.status,
            analysis_metadata=AnalysisMetadata(
                model_name=analysis.model_name,
                model_version=analysis.model_version,
                created_at=analysis.created_at
            )
        )

    async def get_analysis_status(self, analysis_id: UUID):
        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
        )

        analysis = result.scalars().first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not present")
        return analysis

    async def delete_analysis(self, analysis_id: UUID):
        result = await self.db.execute(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
        )

        analysis = result.scalars().first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not present")
        
        await self.db.delete(analysis)
        await self.db.commit()

        return