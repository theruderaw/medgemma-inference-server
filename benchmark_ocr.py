#!/usr/bin/env python3

import base64
import json
import re
import time
from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "medgemma1.5:4b"

IMAGE_DIR = Path(".")
OUTPUT_FILE = Path("results.txt")
JSONL_FILE = Path("results.jsonl")

OLLAMA_TIMEOUT = 300

PROMPT = """Transcribe this handwritten medical document exactly as written.

Rules:
- Preserve the original wording, spelling, abbreviations, numbers, and formatting as closely as possible.
- Do not summarize.
- Do not interpret or correct unclear medical terms.
- Do not add information that is not visible.
- If something is genuinely illegible, write [ILLEGIBLE].
- Return ONLY the transcription.
"""


def image_to_base64(path: Path) -> str:
    """Read an image and return its base64 representation."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_records():
    """
    Group images into AMR records.

    Supports:
        AMR_001_HTR.png
        AMR_002_HTR_p1.png
        AMR_002_HTR_p2.png
        AMR_002_HTR_p3.png

    Multi-page records are returned in page order.
    """
    records = {}

    pattern = re.compile(
        r"^(AMR_\d+)_HTR(?:_p(\d+))?\.png$"
    )

    for path in IMAGE_DIR.glob("AMR_*_HTR*.png"):
        match = pattern.match(path.name)

        if not match:
            continue

        pair_id = match.group(1)

        # Single-page image = page 1
        page = int(match.group(2)) if match.group(2) else 1

        records.setdefault(pair_id, []).append(
            (page, path)
        )

    for pair_id in sorted(records):
        pages = sorted(
            records[pair_id],
            key=lambda item: item[0],
        )

        yield pair_id, [path for _, path in pages]


def get_completed_records():
    """
    Read results.txt and return AMR IDs that have
    successfully completed.
    """
    if not OUTPUT_FILE.exists():
        return set()

    completed = set()

    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if re.fullmatch(r"AMR_\d+", line):
                completed.add(line)

    return completed


def get_completed_json_records():
    """
    Read results.jsonl and return successfully completed
    AMR IDs.

    This provides a second checkpoint in case results.txt
    is manually modified.
    """
    if not JSONL_FILE.exists():
        return set()

    completed = set()

    with JSONL_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                if (
                    record.get("status") == "success"
                    and record.get("pair_id")
                ):
                    completed.add(record["pair_id"])

            except json.JSONDecodeError:
                continue

    return completed


def transcribe(images):
    """
    Send one AMR record to MedGemma.

    All pages belonging to a record are sent together.
    """
    encoded_images = [
        image_to_base64(path)
        for path in images
    ]

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": encoded_images,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if "response" not in data:
        raise RuntimeError(
            f"Ollama response did not contain 'response': {data}"
        )

    transcription = data["response"].strip()

    if not transcription:
        raise RuntimeError(
            "MedGemma returned an empty transcription."
        )

    return transcription


def save_text_result(
    file,
    pair_id,
    images,
    transcription,
    elapsed_seconds,
):
    """Append a human-readable result to results.txt."""

    file.write(f"{'=' * 80}\n")
    file.write(f"{pair_id}\n")
    file.write(
        f"Images: {', '.join(path.name for path in images)}\n"
    )
    file.write(
        f"Time: {elapsed_seconds:.2f}s\n"
    )
    file.write(f"Model: {MODEL}\n")
    file.write(f"{'=' * 80}\n")
    file.write(transcription)
    file.write("\n\n")
    file.flush()


def save_json_result(
    file,
    pair_id,
    images,
    transcription,
    elapsed_seconds,
):
    """Append a machine-readable result to results.jsonl."""

    record = {
        "pair_id": pair_id,
        "status": "success",
        "model": MODEL,
        "images": [
            path.name
            for path in images
        ],
        "num_pages": len(images),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "transcription": transcription,
    }

    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    file.flush()


def save_error_result(
    file,
    pair_id,
    images,
    error,
    elapsed_seconds,
):
    """
    Save failures to results.jsonl.

    IMPORTANT:
    Failed records are NOT written to results.txt,
    so they will be retried on the next run.
    """

    record = {
        "pair_id": pair_id,
        "status": "error",
        "model": MODEL,
        "images": [
            path.name
            for path in images
        ],
        "num_pages": len(images),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "error": str(error),
    }

    file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    file.flush()


def main():
    records = list(get_records())

    if not records:
        print(
            f"No AMR images found in {IMAGE_DIR.resolve()}"
        )
        return

    completed_text = get_completed_records()
    completed_json = get_completed_json_records()

    completed = completed_text | completed_json

    remaining = [
        (pair_id, images)
        for pair_id, images in records
        if pair_id not in completed
    ]

    print()
    print(f"Found:           {len(records)} records")
    print(f"Already done:    {len(completed)} records")
    print(f"Remaining:       {len(remaining)} records")
    print(f"Model:           {MODEL}")
    print(f"Ollama:          {OLLAMA_URL}")
    print()

    if not remaining:
        print("Nothing to analyze.")
        return

    # Append so previous results survive restarts.
    with (
        OUTPUT_FILE.open("a", encoding="utf-8") as text_file,
        JSONL_FILE.open("a", encoding="utf-8") as json_file,
    ):
        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}"
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:

            task = progress.add_task(
                "Starting...",
                total=len(remaining),
            )

            for pair_id, images in remaining:
                progress.update(
                    task,
                    description=(
                        f"Transcribing {pair_id} "
                        f"({len(images)} page(s))"
                    ),
                )

                start = time.perf_counter()

                try:
                    transcription = transcribe(images)

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    save_text_result(
                        text_file,
                        pair_id,
                        images,
                        transcription,
                        elapsed,
                    )

                    save_json_result(
                        json_file,
                        pair_id,
                        images,
                        transcription,
                        elapsed,
                    )

                except Exception as error:
                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    save_error_result(
                        json_file,
                        pair_id,
                        images,
                        error,
                        elapsed,
                    )

                    # Print after the progress display
                    # has a chance to refresh.
                    progress.console.print(
                        f"[red]✗ {pair_id} failed:[/red] "
                        f"{error}"
                    )

                progress.advance(task)

    print()
    print("Done.")
    print(f"Text results: {OUTPUT_FILE}")
    print(f"JSONL results: {JSONL_FILE}")


if __name__ == "__main__":
    main()