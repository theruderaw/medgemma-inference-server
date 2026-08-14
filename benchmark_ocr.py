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

IMAGE_DIR = Path("./datasets/african-medical-records/htr")
OUTPUT_FILE = Path("results.txt")
JSONL_FILE = Path("results.jsonl")

OLLAMA_TIMEOUT = 300

PROMPT = r"""
Transcribe the handwritten medical document exactly as it appears in the image.

This is a transcription task, NOT a medical interpretation task.

STRICT RULES:

1. Transcribe ONLY text that is visibly present in the image.
2. Do NOT summarize, explain, interpret, infer, expand, or correct anything.
3. Do NOT replace an unclear word with a medically plausible word.
4. Do NOT use medical knowledge to guess handwriting.
5. Preserve names, patient IDs, hospital numbers, dates, ages, measurements,
   drug names, strengths, doses, routes, frequencies, durations, diagnoses,
   abbreviations, and other values exactly as written.
6. Preserve the document's line structure and ordering as closely as possible.
7. Preserve abbreviations exactly. For example:
   "bd" must remain "bd", "tds" must remain "tds", "12hrly" must remain
   "12hrly". Do not expand abbreviations.
8. Preserve numbers exactly. Never change a number because another value
   would be medically more likely.
9. Preserve drug names exactly as visually written. Never substitute one
   medication for another.
10. Preserve patient names and identifiers exactly. Never invent missing
    characters or digits.
11. Do not normalize spelling.
12. Do not convert dates into another format.
13. Do not convert units into another format.
14. Do not add punctuation that is not reasonably visible.
15. If a word, number, character, or section is genuinely impossible to read,
    write [ILLEGIBLE].
16. If only part of a word is readable, transcribe the readable portion and
    use [ILLEGIBLE] for the unreadable portion.
17. Do not output commentary such as "I cannot read this", "likely", "probably",
    or explanations.
18. Return ONLY the transcription.

IMPORTANT:
When uncertain between two possible medical terms, DO NOT choose the medically
more likely term. Transcribe what is visually present, or use [ILLEGIBLE].
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
        r"^(AMR_\d+)_HTR(?:_p(\d+))?\.png$",
        re.IGNORECASE,
    )

    for path in IMAGE_DIR.glob("AMR_*_HTR*.png"):
        match = pattern.match(path.name)

        if not match:
            continue

        pair_id = match.group(1).upper()

        page = (
            int(match.group(2))
            if match.group(2)
            else 1
        )

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
    Read results.jsonl and return AMR IDs that have a successful
    result for the CURRENT model and CURRENT prompt.

    This prevents old benchmark results from incorrectly causing
    new runs to be skipped.
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

            except json.JSONDecodeError:
                continue

            if (
                record.get("status") == "success"
                and record.get("pair_id")
                and record.get("model") == MODEL
            ):
                completed.add(record["pair_id"])

    return completed


def clean_transcription(text: str) -> str:
    """
    Remove accidental model wrappers without changing the actual
    transcription content.
    """
    text = text.strip()

    # Remove common markdown code fences if the model ignores
    # the "return only transcription" instruction.
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()

        if len(lines) >= 2:
            lines = lines[1:-1]

        text = "\n".join(lines).strip()

    return text


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

    transcription = clean_transcription(
        data["response"]
    )

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

    Failed records are intentionally NOT written to results.txt
    and are retried on the next run.
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

    completed = get_completed_records()

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