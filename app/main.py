from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.queue import task_queue

from app.features.documents.router import (
    router as document_router,
)

from app.features.analysis.router import (
    router as analysis_router,
)

from app.features.chats.router import (
    router as chats_general_router,
)

from app.features.chat_messages.router import (
    router as chats_message_router,
)

from app.logging_config import (
    configure_structlog,
    configure_uvicorn_logging,
)

from app.middleware import LoggingMiddleware
from app.logger import logger


# =========================================================
# LOGGING
# =========================================================

configure_structlog(
    truncate_at=1000,
)

configure_uvicorn_logging()


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

    # The API process only ever *adds* to the queue;
    # closing the Redis connection here does not affect
    # the worker process, which manages its own connection
    # independently.

    await task_queue.close()


# =========================================================
# APP
# =========================================================

app = FastAPI(
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

origins = [
    "http://localhost:5173",
    "http://localhost:5000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MIDDLEWARE
# =========================================================

app.add_middleware(
    LoggingMiddleware,
)


# =========================================================
# ROUTERS
# =========================================================

app.include_router(
    document_router,
)

app.include_router(
    analysis_router,
)

app.include_router(
    chats_general_router,
)

app.include_router(
    chats_message_router,
)


# =========================================================
# LOG FILE
# =========================================================

LOG_FILE = Path(
    "logs/app.jsonl"
)


# =========================================================
# LOG API
# =========================================================

from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.core.queue import task_queue

from app.features.documents.router import (
    router as document_router,
)

from app.features.analysis.router import (
    router as analysis_router,
)

from app.features.chats.router import (
    router as chats_general_router,
)

from app.features.chat_messages.router import (
    router as chats_message_router,
)

from app.logging_config import (
    configure_structlog,
    configure_uvicorn_logging,
)

from app.middleware import LoggingMiddleware
from app.logger import logger


# =========================================================
# LOGGING
# =========================================================

configure_structlog(
    truncate_at=1000,
)

configure_uvicorn_logging()


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

    # The API process only ever *adds* to the queue;
    # closing the Redis connection here does not affect
    # the worker process, which manages its own connection
    # independently.

    await task_queue.close()


# =========================================================
# APP
# =========================================================

app = FastAPI(
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

origins = [
    "http://localhost:5173",
    "http://localhost:5000",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5000",
    "http://127.0.0.1:3000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MIDDLEWARE
# =========================================================

app.add_middleware(
    LoggingMiddleware,
)


# =========================================================
# ROUTERS
# =========================================================

app.include_router(
    document_router,
)

app.include_router(
    analysis_router,
)

app.include_router(
    chats_general_router,
)

app.include_router(
    chats_message_router,
)


# =========================================================
# LOG FILE
# =========================================================

LOG_FILE = Path(
    "logs/app.jsonl"
)


# =========================================================
# LOG API
# =========================================================

@app.get(
    "/logs",
    include_in_schema=False,
)
async def get_logs(
    level: str | None = Query(
        default=None,
    ),
    event: str | None = Query(
        default=None,
    ),
    limit: int = Query(
        default=500,
        ge=1,
        le=5000,
    ),
):
    if not LOG_FILE.exists():
        return []

    logs = []

    with LOG_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                log = json.loads(line)

            except json.JSONDecodeError:
                continue

            # -------------------------------------------------
            # Level filter
            # -------------------------------------------------

            if (
                level
                and log.get("level") != level
            ):
                continue

            # -------------------------------------------------
            # Event filter
            # -------------------------------------------------

            if (
                event
                and event.lower()
                not in str(
                    log.get("event", "")
                ).lower()
            ):
                continue

            logs.append(
                log
            )

    # Newest first
    logs.reverse()

    return logs[:limit]


# =========================================================
# LOG UI
# =========================================================

LOGS_HTML = (
    Path(__file__).resolve().parent
    / "static"
    / "logs.html"
)


@app.get(
    "/logs/ui",
    include_in_schema=False,
)
async def logs_ui():
    return FileResponse(
        LOGS_HTML
    )


# =========================================================
# STARTUP
# =========================================================

logger.info(
    "Application started",
    event_type="startup",
)