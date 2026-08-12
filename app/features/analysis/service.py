from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import selectinload
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models.analysis import Analysis
from app.models.enums import AnalysisStatus
from app.schemas.analysis import (
    AnalysisMetadata,
    AnalysisRead,
)
from app.logger import logger


class AnalysisService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_analysis(self, analysis_id: UUID):
        logger.info("Fetching analysis", analysis_id=str(analysis_id))
        result = await self.db.exec(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
            .options(selectinload(Analysis.chunks))
        )

        analysis = result.first()

        if not analysis:
            logger.warning("Analysis not found", analysis_id=str(analysis_id))
            raise HTTPException(
                status_code=404,
                detail="Analysis not present",
            )

        entities = list(
            set().union(*(chunk.entities for chunk in analysis.chunks))
        )
        response = AnalysisRead(
            analysis_id=analysis.analysis_id,
            document_id=analysis.document_id,
            raw_output=analysis.raw_output,
            summary=analysis.summary,
            entities=entities,
            status=analysis.status,
            analysis_metadata=AnalysisMetadata(
                model_name=analysis.model_name,
                model_version=analysis.model_version,
                created_at=analysis.created_at,
            ),
        )
        logger.info(
            "Analysis fetched successfully",
            analysis_id=str(analysis_id),
            status=analysis.status,
            entity_count=len(entities),
        )
        return response

    async def get_analysis_status(self, analysis_id: UUID):
        logger.info("Fetching analysis status", analysis_id=str(analysis_id))
        result = await self.db.exec(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
        )

        analysis = result.first()

        if not analysis:
            logger.warning("Analysis not found for status", analysis_id=str(analysis_id))
            raise HTTPException(
                status_code=404,
                detail="Analysis not present",
            )

        logger.info(
            "Analysis status fetched",
            analysis_id=str(analysis_id),
            status=analysis.status,
        )
        return analysis

    async def delete_analysis(self, analysis_id: UUID):
        logger.info("Deleting analysis", analysis_id=str(analysis_id))
        result = await self.db.exec(
            select(Analysis)
            .where(Analysis.analysis_id == analysis_id)
        )

        analysis = result.first()

        if not analysis:
            logger.warning("Analysis not found for deletion", analysis_id=str(analysis_id))
            raise HTTPException(
                status_code=404,
                detail="Analysis not present",
            )

        analysis.status = AnalysisStatus.DELETED
        await self.db.commit()
        logger.info(
            "Analysis deleted (soft)",
            analysis_id=str(analysis_id),
            status=analysis.status,
        )