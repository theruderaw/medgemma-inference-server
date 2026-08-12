from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.features.documents.router import router as document_router
from app.features.analysis.router import router as analysis_router
from app.features.chats.router import router as chats_general_router
from app.features.chat_messages.router import router as chats_message_router

from app.logging_config import configure_structlog,configure_uvicorn_logging
from app.middleware import LoggingMiddleware
from app.logger import logger

configure_structlog(truncate_at=1000)
configure_uvicorn_logging()

app = FastAPI()

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

app.add_middleware(LoggingMiddleware)

app.include_router(document_router)
app.include_router(analysis_router)
app.include_router(chats_general_router)
app.include_router(chats_message_router)

logger.info("Application started", event_type="startup")