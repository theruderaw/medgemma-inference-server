"""
Generic upload-handling utilities (not specific to any file type).
"""

# pyrefly: ignore [missing-import]
from fastapi import HTTPException, UploadFile


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
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break

        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="File too large",
            )

        chunks.append(chunk)

    await file.seek(0)
    return b"".join(chunks)