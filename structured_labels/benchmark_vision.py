import base64
import json
import time
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
from typing import List, Dict, Any

import requests
import textwrap

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("datasets")
OUTPUT_DIR = Path("benchmarks")

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "medgemma1.5:4b",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"
}

# CheXpert label set (only for instruction, not for parsing)
CHEXPERT_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# Base observational prompt
PROMPT_BASE = textwrap.dedent("""
    Analyze this medical image carefully.

    Your task is to provide a structured observational description of what
    is visible in the image and identify anything that appears potentially
    concerning, inconsistent, unusual, or worthy of further review.

    Focus on:
    - Image quality and limitations
    - Visible anatomical structures
    - Apparent abnormalities or unusual findings
    - Internal inconsistencies between visible findings
    - Potential alerts or findings that warrant human review
    - Important negative observations when reasonably supported
    - Uncertainty or ambiguity in the image

    IMPORTANT SAFETY CONSTRAINTS:
    - Do NOT provide a definitive diagnosis.
    - Do NOT claim that the patient has a specific disease or condition.
    - Do NOT present uncertain findings as confirmed findings.
    - Do NOT recommend treatment.
    - Clearly distinguish observations from possible interpretations.
    - If something cannot be determined from the image, explicitly say so.
    - This is an observational/alerting task intended to support human review,
      not replace clinical judgment.

    Use cautious language such as:
    "appears to", "may represent", "raises concern for", "could be consistent
    with", or "requires further review" when appropriate.

    Provide a concise but sufficiently detailed report.
""").strip()

# Additional instruction for CheXpert labelling (informs the model, but we do not parse)
CHEXPERT_INSTRUCTION = textwrap.dedent(f"""
    In addition, please evaluate the following CheXpert findings and label each
    as "positive", "negative", or "uncertain" using the "more likely than not"
    criterion:
    - If the finding is more likely present than absent, label it "positive".
    - If the finding is more likely absent than present, label it "negative".
    - If the evidence is equivocal or insufficient, label it "uncertain".

    The findings are: {', '.join(CHEXPERT_LABELS)}.

    Provide your labels along with your narrative report – there is no need to
    format them as JSON; plain text is acceptable.
""")

PROMPT = PROMPT_BASE + "\n\n" + CHEXPERT_INSTRUCTION

# ============================================================
# Logging setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def encode_image(image_path: Path) -> str:
    """Read image and return base64 encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_ollama(model: str, image_base64: str) -> str:
    """Send request to Ollama and return the raw response text."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_base64],
            }
        ],
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def find_images(dataset_dir: Path) -> List[Path]:
    """Recursively find all image files in dataset_dir."""
    images = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def safe_model_name(model: str) -> str:
    """Sanitise model name for use in filenames."""
    return model.replace(":", "_").replace("/", "_").replace("\\", "_")


def check_ollama_available() -> bool:
    """Quick check if Ollama server is responsive."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


# ============================================================
# Main benchmark
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run medical image benchmark against Ollama models."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of appending.",
    )
    args = parser.parse_args()

    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory does not exist: {DATASET_DIR}")
        sys.exit(1)

    if not check_ollama_available():
        logger.error(
            "Ollama server not reachable at %s. Please start Ollama.",
            OLLAMA_URL
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = find_images(DATASET_DIR)
    logger.info("Found %d images.", len(images))
    logger.info("Models: %s", ", ".join(MODELS))

    if not images:
        logger.warning("No images found. Exiting.")
        return

    # Prepare output files (append or overwrite)
    output_files = {}
    for model in MODELS:
        out_path = OUTPUT_DIR / f"{safe_model_name(model)}.jsonl"
        mode = "w" if args.overwrite else "a"
        output_files[model] = open(out_path, mode, encoding="utf-8")

    with ExitStack() as stack:
        for f in output_files.values():
            stack.push(f)

        # --- LOOP ORDER CHANGED: Models OUTER, Images INNER ---
        for model in MODELS:
            logger.info("=" * 70)
            logger.info(f"STARTING MODEL: {model}")
            
            # Create a progress bar for this model's batch
            iterator = tqdm(images, desc=f"{model}") if tqdm else images

            for image_path in iterator:
                # Encode image (skip if fails)
                try:
                    image_base64 = encode_image(image_path)
                except Exception as e:
                    logger.error("Failed to encode image %s: %s", image_path, e)
                    continue

                relative_path = str(image_path.relative_to(DATASET_DIR))
                started = time.perf_counter()

                result: Dict[str, Any] = {
                    "image": relative_path,
                    "image_path": str(image_path),
                    "model": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt": PROMPT,
                }

                try:
                    output = call_ollama(model, image_base64)
                    elapsed = time.perf_counter() - started
                    result.update({
                        "status": "success",
                        "elapsed_seconds": round(elapsed, 3),
                        "response": output,
                    })
                    logger.debug("  OK (%.2fs)", elapsed)
                except Exception as e:
                    elapsed = time.perf_counter() - started
                    result.update({
                        "status": "error",
                        "elapsed_seconds": round(elapsed, 3),
                        "error": str(e),
                    })
                    logger.error("  ERROR: %s", e)

                # Write to the corresponding file
                output_files[model].write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_files[model].flush()

            logger.info(f"FINISHED MODEL: {model}")
            # Optional: Force unload the model after batch to free VRAM before loading next
            # (Ollama will unload it automatically after 5 mins, but you can manually stop it)
            # os.system(f"ollama stop {model}")  # Uncomment if you want to force unload

    logger.info("Benchmark complete. Results saved to: %s", OUTPUT_DIR)
    
if __name__ == "__main__":
    main()