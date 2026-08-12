"""
Worker process entry point.

Run with:

    python -m app.worker

This process is independent of the FastAPI API process. It does not expose
any HTTP endpoints. It waits for tasks on the Redis queue and executes them
via the existing analysis/RAG service `run()` entry points.

Concurrency: a single worker processes one task at a time, deliberately --
the inference pipeline shares local model resources (vision/text/embedding
models via Ollama). Run additional worker processes (not additional
in-process concurrency) if throughput requires it later; the queue already
supports multiple consumers safely since BRPOP is atomic per-item.
"""
from __future__ import annotations

import asyncio
import signal
import traceback

from app.core.config import settings
from app.logger import logger
from app.logging_config import configure_structlog, configure_uvicorn_logging
from app.queue.dispatcher import dispatch
from app.queue.redis_queue import DEFAULT_POLL_TIMEOUT_SECONDS, RedisQueue


class Worker:
    def __init__(self, queue: RedisQueue):
        self._queue = queue
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        logger.info("Worker stop requested; will finish current task and exit")
        self._stop_event.set()

    async def run(self) -> None:
        logger.info("Worker started, connecting to Redis", redis_url=settings.REDIS_URL)
        await self._queue.ping()
        logger.info("Worker connected to Redis, waiting for tasks")

        while not self._stop_event.is_set():
            try:
                task = await self._queue.get(timeout=DEFAULT_POLL_TIMEOUT_SECONDS)
            except Exception:
                logger.error(
                    "Error polling queue, backing off",
                    traceback=traceback.format_exc(),
                )
                await asyncio.sleep(DEFAULT_POLL_TIMEOUT_SECONDS)
                continue

            if task is None:
                # Timed out waiting for work; loop back and re-check for
                # a shutdown request.
                continue

            logger.info(
                "Task received",
                task_id=str(task.task_id),
                task_type=str(task.task_type),
            )
            try:
                await dispatch(task)
                await self._queue.mark_completed(task.task_id)
                logger.info("Task completed", task_id=str(task.task_id))
            except Exception as e:
                # The underlying service `run()` methods already mark
                # Analysis.status = FAILED (application state lives in
                # PostgreSQL). Here we only record that the *queue entry*
                # failed, and keep the worker alive for the next task.
                await self._queue.mark_failed(task.task_id)
                logger.error(
                    "Task failed",
                    task_id=str(task.task_id),
                    task_type=str(task.task_type),
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

        logger.info("Worker loop exited, closing Redis connection")
        await self._queue.close()
        logger.info("Worker shut down cleanly")


async def _amain() -> None:
    queue = RedisQueue(settings.REDIS_URL)
    worker = Worker(queue)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, worker.request_stop)
        except NotImplementedError:
            # add_signal_handler isn't available on some platforms (e.g.
            # Windows); fall back to default KeyboardInterrupt handling.
            pass

    await worker.run()


def main() -> None:
    configure_structlog(truncate_at=1000)
    configure_uvicorn_logging()
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        logger.info("Worker interrupted, exiting")


if __name__ == "__main__":
    main()