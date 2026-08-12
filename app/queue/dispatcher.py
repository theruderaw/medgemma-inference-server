"""
Maps a consumed TaskEnvelope to the existing service entry points.

This module intentionally contains no pipeline logic of its own -- it only
identifies the task type and calls into `ImageAnalysisService.run`,
`PDFAnalysisService.run`, or `RAGService.run`, all of which already create
their own DB session and already handle marking `Analysis.status` (or
saving the `ChatMessage`) on success/failure.
"""
from __future__ import annotations

from app.inference.image_analysis import ImageAnalysisService
from app.inference.pdf_analysis import PDFAnalysisService
from app.inference.rag import RAGService
from app.logger import logger
from app.queue.tasks import TaskEnvelope, TaskType


async def dispatch(task: TaskEnvelope) -> None:
    """Execute the service call for a given task. Raises on failure."""
    if task.task_type == TaskType.ANALYSIS_PDF:
        payload = task.analysis_payload()
        logger.info(
            "Dispatching PDF analysis task",
            task_id=str(task.task_id),
            analysis_id=str(payload.analysis_id),
        )
        await PDFAnalysisService.run(payload.analysis_id)

    elif task.task_type == TaskType.ANALYSIS_IMAGE:
        payload = task.analysis_payload()
        logger.info(
            "Dispatching image analysis task",
            task_id=str(task.task_id),
            analysis_id=str(payload.analysis_id),
        )
        await ImageAnalysisService.run(payload.analysis_id)

    elif task.task_type == TaskType.RAG:
        payload = task.rag_payload()
        logger.info(
            "Dispatching RAG task",
            task_id=str(task.task_id),
            chat_id=str(payload.chat_id),
        )
        await RAGService.run(
            chat_id=payload.chat_id,
            query=payload.query,
            current_document_id=payload.current_document_id,
        )

    else:
        # Should be unreachable given TaskType is a closed enum, but guard
        # against a payload written by a future/older worker version.
        logger.error(
            "Unknown task type, dropping task",
            task_id=str(task.task_id),
            task_type=str(task.task_type),
        )
        raise ValueError(f"Unknown task type: {task.task_type}")