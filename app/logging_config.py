import logging
import structlog
from structlog.stdlib import ProcessorFormatter


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
        processors.append(truncate_processor(truncate_at))
    processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )


def configure_uvicorn_logging():
    """
    Configure uvicorn loggers to output JSON via structlog,
    using the same renderer and timestamp format as the app.
    """
    # Build a formatter that mimics the app's processors,
    # but for standard logging records (foreign logs).
    # The 'foreign_pre_chain' processes the log record before
    # the final processor (JSONRenderer) is applied.
    formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Replace handlers on uvicorn loggers
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False