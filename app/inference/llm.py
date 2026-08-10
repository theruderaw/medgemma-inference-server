"""
Functions that talk to Ollama — no server, just a client module.

Images may be passed as raw bytes, base64 strings, or FastAPI UploadFile
objects (e.g. straight from an endpoint's `file: UploadFile` param).

Usage:
from app.core.ollama_client import chat, embed, generate

await chat(model="llama3", messages=[{"role": "user", "content": "hi"}])

# with an image, e.g. from a FastAPI endpoint:
# async def route(file: UploadFile):
#     await generate(model="llava", prompt="describe this", images=[file])
"""

import base64
from typing import Any, AsyncIterator

import httpx
from fastapi import UploadFile

from app.core.config import settings
from app.inference.types import ImageInput


OLLAMA_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=120.0,
    write=40.0,
    pool=5.0,
)


def _url(path: str) -> str:
    return f"{settings.OLLAMA_URL.rstrip('/')}{path}"


async def _to_base64(image: ImageInput) -> str:
    """Normalize bytes / base64 str / UploadFile into a base64 string."""
    if isinstance(image, UploadFile):
        data = await image.read()
        return base64.b64encode(data).decode()

    if isinstance(image, bytes):
        return base64.b64encode(image).decode()

    if isinstance(image, str):
        # Assume it's already base64-encoded.
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

    if not stream:
        try:
            async with httpx.AsyncClient(
                timeout=OLLAMA_TIMEOUT
            ) as client:
                resp = await client.post(
                    _url("/api/chat"),
                    json=payload,
                )

                resp.raise_for_status()
                return resp.json()

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    async def _iter() -> AsyncIterator[dict]:
        try:
            async with httpx.AsyncClient(
                timeout=OLLAMA_TIMEOUT
            ) as client:
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

    if not stream:
        try:
            async with httpx.AsyncClient(
                timeout=OLLAMA_TIMEOUT
            ) as client:
                resp = await client.post(
                    _url("/api/generate"),
                    json=payload,
                )

                resp.raise_for_status()
                return resp.json()

        except httpx.TimeoutException as e:
            _raise_timeout(e)

    async def _iter() -> AsyncIterator[dict]:
        try:
            async with httpx.AsyncClient(
                timeout=OLLAMA_TIMEOUT
            ) as client:
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

    try:
        async with httpx.AsyncClient(
            timeout=OLLAMA_TIMEOUT
        ) as client:
            resp = await client.post(
                _url("/api/embed"),
                json=payload,
            )

            resp.raise_for_status()
            return resp.json()

    except httpx.TimeoutException as e:
        _raise_timeout(e)