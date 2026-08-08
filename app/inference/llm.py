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
from app.inference.types import ImageInput

from app.core.config import settings



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


async def _encode_images(images: list[ImageInput] | None) -> list[str] | None:
    if not images:
        return None
    return [await _to_base64(img) for img in images]


async def chat(
    model: str,
    messages: list[dict],
    images: list[ImageInput] | None = None,
    stream: bool = False,
    format_json: bool = False,
    **kwargs: Any,
) -> dict | AsyncIterator[dict]:
    """Call /api/chat. Returns a dict, or an async iterator of dicts if stream=True.

    If `images` is given, they're attached to the last message in `messages`
    (Ollama expects per-message "images", so this covers the common single-turn
    vision case). For multi-turn image attachment, add "images" to a message
    dict directly instead.
    """
    if images:
        b64_images = await _encode_images(images)
        messages = [*messages[:-1], {**messages[-1], "images": b64_images}]
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **kwargs}
    # print("\n"*5,payload,"\n"*5)
    if format_json:
        payload["format"] = 'json'
    if not stream:
        async with httpx.AsyncClient(timeout=None) as client:
            # print(payload)
            resp = await client.post(_url("/api/chat"), json=payload)
            print(resp.status_code,":",resp.text)
            resp.raise_for_status()
            return resp.json()

    async def _iter() -> AsyncIterator[dict]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", _url("/api/chat"), json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield httpx.Response(200, content=line).json()

    return _iter()


async def generate(
    model: str,
    prompt: str,
    images: list[ImageInput] | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict | AsyncIterator[dict]:
    """Call /api/generate. Returns a dict, or an async iterator of dicts if stream=True."""
    payload = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

    b64_images = await _encode_images(images)
    if b64_images:
        payload["images"] = b64_images

    if not stream:
        async with httpx.AsyncClient(timeout=None) as client:
            print("/n"*5,payload,"/n"*5)
            resp = await client.post(_url("/api/generate"), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def _iter() -> AsyncIterator[dict]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", _url("/api/generate"), json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield httpx.Response(200, content=line).json()

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

    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            _url("/api/embed"),
            json=payload,
        )

        resp.raise_for_status()

        return resp.json()