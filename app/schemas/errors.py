from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str = Field(
        ...,
        description="Machine-readable error code.",
        examples=["DOCUMENT_NOT_FOUND"],
    )

    message: str = Field(
        ...,
        description="Human-readable error message.",
        examples=["The requested document does not exist."],
    )

    details: dict[str, Any] | None = Field(
        default=None,
        description="Additional error-specific information.",
    )


ERROR_400 = {
    "model": ErrorResponse,
    "description": "Bad Request",
}

ERROR_404 = {
    "model": ErrorResponse,
    "description": "Resource Not Found",
}

ERROR_409 = {
    "model": ErrorResponse,
    "description": "Resource Conflict",
}

ERROR_415 = {
    "model": ErrorResponse,
    "description": "Unsupported Media Type",
}

ERROR_500 = {
    "model": ErrorResponse,
    "description": "Internal Server Error",
}

ERROR_503 = {
    "model": ErrorResponse,
    "description": "Service Unavailable",
}