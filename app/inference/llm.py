import base64
from io import BytesIO
from typing import Any, AsyncIterator

import httpx
from fastapi import UploadFile
from PIL import Image

from app.core.config import settings
from app.inference.types import ImageInput
from app.logger import logger


OLLAMA_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=120.0,
    write=40.0,
    pool=5.0,
)

# Target resolution for Vision-Language Models to reduce hallucinations
TARGET_IMAGE_SIZE = (448, 448)


def _url(path: str) -> str:
    return f"{settings.OLLAMA_URL.rstrip('/')}{path}"


def _resize_and_encode(image_bytes: bytes) -> str:
    """Resize image bytes to TARGET_IMAGE_SIZE and return base64 string."""
    with Image.open(BytesIO(image_bytes)) as img:
        # Convert to RGB to ensure compatibility across formats (PNG/JPEG)
        img_resized = img.convert("RGB").resize(
            TARGET_IMAGE_SIZE, Image.Resampling.LANCZOS
        )
        buffer = BytesIO()
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def _to_base64(image: ImageInput) -> str:
    """Normalize bytes / base64 str / UploadFile into a 448x448 base64 string."""
    if isinstance(image, UploadFile):
        data = await image.read()
        return _resize_and_encode(data)

    if isinstance(image, bytes):
        return _resize_and_encode(image)

    if isinstance(image, str):
        # Decode base64 str, resize, and re-encode
        try:
            data = base64.b64decode(image)
            return _resize_and_encode(data)
        except Exception:
            # Fallback if raw string cannot be parsed as base64 bytes
            return image

    raise TypeError(f"Unsupported image type: {type(image)!r}")


async def _encode_images(
    images: list[ImageInput] | None,
) -> list[str] | None:
    if not images:
        return None

    return [await _to_base64(img) for img in images]


def _raise_timeout(e: httpx.TimeoutException) -> None:
    """Convert HTTPX timeout errors into a controlled client-layer error."""
    logger.error("Ollama request timed out", error=str(e))
    raise RuntimeError("Ollama request timed out") from e


async def chat(
    model: str,
    messages: list[dict],
    images: list[ImageInput] | None = None,
    stream: bool = False,
    format_json: bool = False,
    **kwargs: Any,
) -> dict | AsyncIterator[dict]:
    """
    Call /api/chat.

    Returns a dict for non-streaming requests, or an async iterator of dicts
    when stream=True.

    If images are given, they are attached to the last message.
    """
    if images:
        b64_images = await _encode_images(images)
        messages = [
            *messages[:-1],
            {
                **messages[-1],
                "images": b64_images,
            },
        ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs,
    }

    if format_json:
        payload["format"] = "json"

    logger.debug(
        "Calling Ollama chat",
        model=model,
        stream=stream,
        format_json=format_json,
        message_count=len(messages),
        has_images=bool(images),
    )

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    _url("/api/chat"),
                    json=payload,
                )

                resp.raise_for_status()
                result = resp.json()
                logger.debug("Ollama chat response received", model=model)
                return result

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    async def _iter() -> AsyncIterator[dict]:
        logger.debug("Starting Ollama chat stream", model=model)
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    _url("/api/chat"),
                    json=payload,
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if line:
                            yield httpx.Response(
                                200,
                                content=line,
                            ).json()

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    return _iter()


async def generate(
    model: str,
    prompt: str,
    images: list[ImageInput] | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict | AsyncIterator[dict]:
    """Call /api/generate."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        **kwargs,
    }

    b64_images = await _encode_images(images)

    if b64_images:
        payload["images"] = b64_images

    logger.debug(
        "Calling Ollama generate",
        model=model,
        stream=stream,
        prompt_length=len(prompt),
        has_images=bool(b64_images),
    )

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    _url("/api/generate"),
                    json=payload,
                )

                resp.raise_for_status()
                result = resp.json()
                logger.debug("Ollama generate response received", model=model)
                return result

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    async def _iter() -> AsyncIterator[dict]:
        logger.debug("Starting Ollama generate stream", model=model)
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    _url("/api/generate"),
                    json=payload,
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if line:
                            yield httpx.Response(
                                200,
                                content=line,
                            ).json()

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    return _iter()


async def embed(
    model: str,
    input: str | list[str],
    **kwargs: Any,
) -> dict:
    """Call /api/embed."""

    payload = {
        "model": model,
        "input": input,
        **kwargs,
    }

    logger.debug(
        "Calling Ollama embed",
        model=model,
        input_type=type(input).__name__,
    )

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                _url("/api/embed"),
                json=payload,
            )

            resp.raise_for_status()
            result = resp.json()
            logger.debug("Ollama embed response received", model=model)
            return result

    except httpx.TimeoutException as e:
        _raise_timeout(e)