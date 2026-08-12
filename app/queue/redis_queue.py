"""
A small Redis-backed queue abstraction.

Mental model: a persistent deque of work, consumed by a separate worker
process. This is intentionally *not* a priority queue or a task-management
system -- see the design doc. It supports exactly the operations the
worker/API need:

    add()     -- API enqueues a task
    get()     -- worker consumes the next task (blocking)
    check()   -- inspect whether a task is queued/processing/completed/failed
    remove()  -- cancel a task that hasn't started executing yet

Redis layout (all keys prefixed with `settings.REDIS_URL`'s namespace via
the `key_prefix` given to RedisQueue):

    {prefix}:pending                 LIST of task_id strings (the deque)
    {prefix}:payload:{task_id}       STRING, JSON-encoded TaskEnvelope
    {prefix}:status:{task_id}        STRING, one of TaskStatus values

The database (PostgreSQL) remains the source of truth for *analysis*
state (`Analysis.status`). The `status` key here only tracks whether the
*queue entry itself* is waiting, has been picked up, or has finished --
it is not a second analysis-state database.
"""
from __future__ import annotations

from enum import Enum
from uuid import UUID

import redis.asyncio as redis

from app.queue.tasks import TaskEnvelope

# How long a terminal status (completed/failed) or a stale payload is kept
# around before Redis expires it. This is just for `check()` inspection /
# debugging -- it is not relied on for correctness.
_TERMINAL_STATUS_TTL_SECONDS = 60 * 60  # 1 hour

# How long the worker blocks on each BRPOP call before looping again. This
# bounds how quickly the worker notices a shutdown request; it does not
# affect how quickly a task is picked up (BRPOP returns immediately when
# work is pushed).
DEFAULT_POLL_TIMEOUT_SECONDS = 5


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RedisQueue:
    def __init__(self, redis_url: str, key_prefix: str = "queue"):
        self._redis_url = redis_url
        self._prefix = key_prefix
        self._client: redis.Redis = redis.from_url(
            redis_url,
            decode_responses=True,
        )

    # -- key helpers ---------------------------------------------------

    @property
    def _pending_key(self) -> str:
        return f"{self._prefix}:pending"

    def _payload_key(self, task_id: UUID | str) -> str:
        return f"{self._prefix}:payload:{task_id}"

    def _status_key(self, task_id: UUID | str) -> str:
        return f"{self._prefix}:status:{task_id}"

    # -- lifecycle -------------------------------------------------------

    async def ping(self) -> bool:
        return await self._client.ping()

    async def close(self) -> None:
        await self._client.aclose()

    # -- operations --------------------------------------------------------

    async def add(self, task: TaskEnvelope) -> UUID:
        """Add a task to the queue. Returns the task id."""
        task_id = task.task_id
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.set(self._payload_key(task_id), task.to_json())
            pipe.set(self._status_key(task_id), TaskStatus.QUEUED.value)
            pipe.lpush(self._pending_key, str(task_id))
            await pipe.execute()
        return task_id

    async def get(
        self, timeout: int = DEFAULT_POLL_TIMEOUT_SECONDS
    ) -> TaskEnvelope | None:
        """
        Block waiting for the next task, up to `timeout` seconds.

        Returns None on timeout (no work available), so the worker loop can
        wake periodically and check for a shutdown request. Marks the task
        as `processing` once it has been popped off the pending list, since
        at that point it is no longer merely queued.
        """
        result = await self._client.brpop(self._pending_key, timeout=timeout)
        if result is None:
            return None

        _, task_id = result
        payload_raw = await self._client.get(self._payload_key(task_id))
        if payload_raw is None:
            # Payload vanished (e.g. expired or removed out of band).
            # Nothing meaningful to execute; let the worker move on.
            await self._client.delete(self._status_key(task_id))
            return None

        await self._client.set(self._status_key(task_id), TaskStatus.PROCESSING.value)
        return TaskEnvelope.from_json(payload_raw)

    async def check(self, task_id: UUID | str) -> TaskStatus | None:
        """Return the queue-entry status for a task, or None if unknown."""
        status = await self._client.get(self._status_key(task_id))
        if status is None:
            return None
        return TaskStatus(status)

    async def remove(self, task_id: UUID | str) -> bool:
        """
        Remove a *queued* task before a worker starts it.

        Returns True if the task was found and removed from the pending
        list, False otherwise (already consumed, already finished, or
        never existed). This does not attempt to cancel work that a
        worker has already picked up -- that is a separate concern.
        """
        removed_count = await self._client.lrem(self._pending_key, 0, str(task_id))
        if removed_count:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.delete(self._payload_key(task_id))
                pipe.delete(self._status_key(task_id))
                await pipe.execute()
            return True
        return False

    async def mark_completed(self, task_id: UUID | str) -> None:
        await self._client.set(
            self._status_key(task_id),
            TaskStatus.COMPLETED.value,
            ex=_TERMINAL_STATUS_TTL_SECONDS,
        )
        await self._client.delete(self._payload_key(task_id))

    async def mark_failed(self, task_id: UUID | str) -> None:
        await self._client.set(
            self._status_key(task_id),
            TaskStatus.FAILED.value,
            ex=_TERMINAL_STATUS_TTL_SECONDS,
        )
        await self._client.delete(self._payload_key(task_id))