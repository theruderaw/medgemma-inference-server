import asyncio
import json
import logging
from pathlib import Path

import httpx
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000/api/v1"

DATASET_DIR = Path("datasets")
RESULTS_DIR = Path("results")

LIMIT = 100
POLL_INTERVAL = 2.0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            console=console,
            show_path=False,
        )
    ],
)

# Suppress httpx request/response logs (including the status polling calls)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=30.0,
    write=60.0,
    pool=10.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_images() -> list[Path]:
    images = sorted(
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    return images[:LIMIT]


def get_content_type(path: Path) -> str:
    if path.suffix.lower() == ".png":
        return "image/png"

    return "image/jpeg"


async def upload_document(
    client: httpx.AsyncClient,
    path: Path,
) -> str:
    with path.open("rb") as file:
        response = await client.post(
            f"{BASE_URL}/documents/upload",
            files={
                "file": (
                    path.name,
                    file,
                    get_content_type(path),
                )
            },
        )

    response.raise_for_status()

    return response.json()["document_id"]


async def start_analysis(
    client: httpx.AsyncClient,
    document_id: str,
) -> str:
    response = await client.post(
        f"{BASE_URL}/documents/{document_id}/analysis"
    )

    response.raise_for_status()

    return response.json()["analysis_id"]


async def wait_for_analysis(
    client: httpx.AsyncClient,
    analysis_id: str,
) -> dict:
    while True:
        response = await client.get(
            f"{BASE_URL}/analysis/{analysis_id}/status"
        )

        response.raise_for_status()

        status = response.json()["status"]

        if status == "complete":
            break

        if status == "failed":
            raise RuntimeError(
                f"Analysis failed: {analysis_id}"
            )

        if status == "deleted":
            raise RuntimeError(
                f"Analysis deleted: {analysis_id}"
            )

        await asyncio.sleep(POLL_INTERVAL)

    response = await client.get(
        f"{BASE_URL}/analysis/{analysis_id}"
    )

    response.raise_for_status()

    return response.json()


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------

async def process_image(
    client: httpx.AsyncClient,
    path: Path,
) -> dict:
    document_id = await upload_document(
        client,
        path,
    )

    analysis_id = await start_analysis(
        client,
        document_id,
    )

    analysis = await wait_for_analysis(
        client,
        analysis_id,
    )

    return {
        "filename": path.name,
        "path": str(path),
        "document_id": document_id,
        "analysis_id": analysis_id,
        "raw_output": analysis.get("raw_output"),
        "summary": analysis.get("summary"),
        "entities": analysis.get("entities", []),
        "status": analysis.get("status"),
        "analysis_metadata": analysis.get(
            "analysis_metadata"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_images()

    if not images:
        logger.error(
            "No images found in %s",
            DATASET_DIR.resolve(),
        )
        return

    logger.info(
        "Found %d images — evaluating first %d",
        len(images),
        min(len(images), LIMIT),
    )

    results = []
    failed = []

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=TIMEOUT,
    ) as client:

        with Progress(
            SpinnerColumn(),
            TextColumn(
                "[progress.description]{task.description}"
            ),
            BarColumn(),
            TextColumn(
                "{task.completed}/{task.total}"
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:

            task = progress.add_task(
                "Evaluating...",
                total=len(images),
            )

            for path in images:
                progress.update(
                    task,
                    description=f"Evaluating {path.name}",
                )

                try:
                    result = await process_image(
                        client,
                        path,
                    )

                    results.append(result)

                except Exception as exc:
                    failed.append(
                        {
                            "filename": path.name,
                            "error": str(exc),
                        }
                    )

                    logger.error(
                        "%s: %s",
                        path.name,
                        exc,
                    )

                finally:
                    progress.advance(task)

    output = {
        "total": len(images),
        "successful": len(results),
        "failed": len(failed),
        "results": results,
        "failures": failed,
    }

    output_path = RESULTS_DIR / "evaluation.json"

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    console.print()
    console.print("[bold]Evaluation complete[/bold]")
    console.print(
        f"  Successful : {len(results)}"
    )
    console.print(
        f"  Failed     : {len(failed)}"
    )
    console.print(
        f"  Total      : {len(images)}"
    )
    console.print(
        f"  Results    : {output_path}"
    )


if __name__ == "__main__":
    asyncio.run(main())