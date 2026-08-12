import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from structlog.stdlib import ProcessorFormatter


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.jsonl"


class IgnoreLogsFilter(logging.Filter):
    """
    Prevent /logs requests from appearing in Uvicorn access logs.

    This is necessary because /logs requests are logged by
    uvicorn.access independently of our LoggingMiddleware.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True

        try:
            # Uvicorn access log args are typically:
            # (client_addr, request_line, status_code)
            #
            # Depending on the Uvicorn version, the request line
            # can be found in a different position, so inspect
            # all string arguments.
            args = record.args

            if isinstance(args, tuple):
                for arg in args:
                    if isinstance(arg, str):
                        if arg.startswith("/logs"):
                            return False

        except Exception:
            # Never let logging break the application.
            pass

        return True


def truncate_processor(truncate_at: int = None):
    def processor(_, __, event_dict):
        if truncate_at is None:
            return event_dict

        for key, value in event_dict.items():
            if isinstance(value, str) and len(value) > truncate_at:
                event_dict[key] = value[:truncate_at] + "..."

        return event_dict

    return processor


def configure_structlog(truncate_at: int = None):
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if truncate_at is not None:
        processors.append(
            truncate_processor(truncate_at)
        )

    processors.append(
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter
    )

    formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # File
    # ---------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # Root logger
    # ---------------------------------------------------------

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    root_logger.setLevel(
        logging.INFO
    )

    root_logger.addHandler(
        console_handler
    )

    root_logger.addHandler(
        file_handler
    )

    # ---------------------------------------------------------
    # Structlog
    # ---------------------------------------------------------

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )


def configure_uvicorn_logging():
    """
    Configure Uvicorn loggers to output JSON via structlog.

    Logs are written to:
        - terminal
        - logs/app.jsonl

    Requests to /logs are excluded from Uvicorn access logs.
    """

    formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # File
    # ---------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    # ---------------------------------------------------------
    # Ignore /logs in Uvicorn access logging
    # ---------------------------------------------------------

    logs_filter = IgnoreLogsFilter()

    console_handler.addFilter(
        logs_filter
    )

    file_handler.addFilter(
        logs_filter
    )

    # ---------------------------------------------------------
    # Uvicorn
    # ---------------------------------------------------------

    for name in (
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
    ):
        logger = logging.getLogger(name)

        logger.handlers = [
            console_handler,
            file_handler,
        ]

        logger.setLevel(
            logging.INFO
        )

        logger.propagate = False