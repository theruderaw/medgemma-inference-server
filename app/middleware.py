from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
)

from uuid import uuid4


class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):
        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        path = scope.get(
            "path",
            "",
        )

        # Don't create logging context for the
        # log dashboard or log API.
        if path.startswith("/logs"):
            await self.app(
                scope,
                receive,
                send,
            )
            return

        bind_contextvars(
            request_id=str(uuid4()),
            path=path,
            method=scope.get("method"),
        )

        try:
            await self.app(
                scope,
                receive,
                send,
            )

        finally:
            clear_contextvars()