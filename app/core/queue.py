from app.core.config import settings
from app.queue.redis_queue import RedisQueue

# A single RedisQueue instance for the API process's lifetime. redis.asyncio
# manages its own internal connection pool, so this is safe to share across
# requests -- same pattern as the SQLAlchemy `engine` in core/database.py.
task_queue = RedisQueue(settings.REDIS_URL)


def get_queue() -> RedisQueue:
    return task_queue