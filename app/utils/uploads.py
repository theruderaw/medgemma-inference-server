"""
Generic upload-handling utilities (not specific to any file type).
"""

# pyrefly: ignore [missing-import]
from fastapi import HTTPException, UploadFile
from app.logger import logger


async def read_upload_within_limit(
    file: UploadFile,
    max_bytes: int,
    chunk_size: int,
) -> bytes:
    """Read an upload in bounded chunks, rejecting it as soon as `max_bytes`
    is exceeded instead of buffering an arbitrarily large body in full
    before its length is ever checked.

    Raises:
        HTTPException(413): If the upload exceeds `max_bytes`.
    """
    logger.debug(
        "Reading upload within limit",
        filename=file.filename,
        max_bytes=max_bytes,
        chunk_size=chunk_size,
    )

    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            logger.warning(
                "Upload exceeded size limit",
                filename=file.filename,
                total=total,
                max_bytes=max_bytes,
            )
            raise HTTPException(
                status_code=413,
                detail="File too large",
            )

        chunks.append(chunk)

    await file.seek(0)
    result = b"".join(chunks)
    logger.debug(
        "Upload read completed",
        filename=file.filename,
        total_bytes=len(result),
    )
    return result