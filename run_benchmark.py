"""
setup_chats.py

Ingestion step that must run BEFORE run_benchmark.py.

For every image file in a directory:
  1. POST /documents/upload            -> get document_id
  2. POST /documents/{id}/analysis     -> kick off analysis (returns analysis_id)
  3. Poll GET /analysis/{id}/status    -> wait until status == "complete" (or "failed")
  4. POST /chats                       -> create a new chat session
  5. POST /chats/{chat_id}/document/{document_id}
                                        -> attach the document (this is the
                                           "first message" of the conversation --
                                           everything after is a text query)

Writes a manifest (chat_id, document_id, analysis_id, filename, status) to
JSON so you can see what succeeded/failed before running the benchmark.

Usage:
    python3 setup_chats.py --base-url http://localhost:8000 --images-dir ./datasets --out manifest.json
    python3 setup_chats.py --base-url http://localhost:8000 --images-dir ./datasets --out manifest.json --limit 5
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

POLL_INTERVAL_SECONDS = 2.0
ANALYSIS_TIMEOUT_SECONDS = 300.0  # analysis can be slow (image model inference)
TERMINAL_STATUSES = {"complete", "failed", "deleted"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


async def upload_document(client: httpx.AsyncClient, base_url: str, path: Path) -> str:
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "image/png")}
        resp = await client.post(f"{base_url}/documents/upload", files=files)
    resp.raise_for_status()
    return resp.json()["document_id"]


async def trigger_analysis(client: httpx.AsyncClient, base_url: str, document_id: str) -> str:
    resp = await client.post(f"{base_url}/documents/{document_id}/analysis")
    resp.raise_for_status()
    return resp.json()["analysis_id"]


async def wait_for_analysis(
    client: httpx.AsyncClient, base_url: str, analysis_id: str
) -> str:
    """Polls until analysis reaches a terminal status. Returns the final status."""
    deadline = time.monotonic() + ANALYSIS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        resp = await client.get(f"{base_url}/analysis/{analysis_id}/status")
        resp.raise_for_status()
        status = resp.json()["status"]
        if status in TERMINAL_STATUSES:
            return status
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
    return "timeout"


async def create_chat(client: httpx.AsyncClient, base_url: str, title: str) -> str:
    resp = await client.post(f"{base_url}/chats", json={"title": title})
    resp.raise_for_status()
    return resp.json()["chat_id"]


async def attach_document(
    client: httpx.AsyncClient, base_url: str, chat_id: str, document_id: str
) -> None:
    resp = await client.post(f"{base_url}/chats/{chat_id}/document/{document_id}")
    resp.raise_for_status()


async def ingest_one(
    client: httpx.AsyncClient, base_url: str, path: Path
) -> dict:
    record = {"filename": path.name}
    try:
        document_id = await upload_document(client, base_url, path)
        record["document_id"] = document_id

        analysis_id = await trigger_analysis(client, base_url, document_id)
        record["analysis_id"] = analysis_id

        status = await wait_for_analysis(client, base_url, analysis_id)
        record["analysis_status"] = status

        if status != "complete":
            record["error"] = f"analysis ended with status={status}, not 'complete'"
            return record

        chat_id = await create_chat(client, base_url, title=path.stem)
        record["chat_id"] = chat_id

        await attach_document(client, base_url, chat_id, document_id)

        record["error"] = None
        return record

    except httpx.HTTPStatusError as e:
        record["error"] = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        return record
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        return record


async def main(base_url: str, images_dir: str, out_path: str, limit: int | None):
    image_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        image_paths = image_paths[:limit]

    print(f"Found {len(image_paths)} image(s) to ingest.", file=sys.stderr)

    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, path in enumerate(image_paths, start=1):
            print(f"[{i}/{len(image_paths)}] {path.name} ...", file=sys.stderr, end=" ")
            record = await ingest_one(client, base_url, path)
            status = "OK" if not record.get("error") else f"ERROR: {record['error']}"
            print(status, file=sys.stderr)
            results.append(record)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_ok = sum(1 for r in results if not r.get("error"))
    n_err = len(results) - n_ok
    print(
        f"\nDone. {n_ok} ingested successfully, {n_err} failed.\n"
        f"Manifest written to {out_path}\n"
        f"Once this looks good, run run_benchmark.py against the same --base-url.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--images-dir", required=True, help="Directory containing xray_*.png files")
    parser.add_argument("--out", default="manifest.json")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only ingest the first N images (smoke test)"
    )
    args = parser.parse_args()

    asyncio.run(main(args.base_url, args.images_dir, args.out, args.limit))