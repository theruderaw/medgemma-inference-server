#!/usr/bin/env python3

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
MODEL = "qwen2.5:3b"

INPUT_FILE = Path("results.jsonl")
TRUTH_DIR = Path("./datasets/african-medical-records/truth")
OUTPUT_FILE = Path("evaluation.jsonl")

OLLAMA_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text only for metric calculation.

    We preserve the original text for Qwen evaluation.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# CER / WER
# ---------------------------------------------------------------------------

def levenshtein_distance(a, b):
    """
    Standard Levenshtein edit distance.
    """
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def calculate_cer(reference, hypothesis):
    """
    Character Error Rate.

    CER = edit distance / number of reference characters
    """
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    if not reference:
        return 0.0 if not hypothesis else 1.0

    distance = levenshtein_distance(
        reference,
        hypothesis,
    )

    return distance / len(reference)


def calculate_wer(reference, hypothesis):
    """
    Word Error Rate.

    WER = word-level edit distance / number of reference words
    """
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    distance = levenshtein_distance(
        reference_words,
        hypothesis_words,
    )

    return distance / len(reference_words)


# ---------------------------------------------------------------------------
# Truth loading
# ---------------------------------------------------------------------------

def load_truth(pair_id):
    """
    AMR_001 -> ./datasets/truth/AMR_001.txt
    """
    path = TRUTH_DIR / f"{pair_id}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Truth file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    ).strip()


# ---------------------------------------------------------------------------
# Qwen evaluation
# ---------------------------------------------------------------------------

def build_prompt(reference, prediction):
    """
    Build the Qwen evaluation prompt.

    No .format() is used, so JSON braces cannot accidentally
    become Python formatting placeholders.
    """

    return f"""You are evaluating a handwritten medical-document transcription.

Compare the REFERENCE transcription with the PREDICTED transcription.

Evaluate ONLY the transcription quality.
Do not rewrite the transcription.
Do not diagnose the patient.
Do not invent information.

Consider:
- missing words or information
- incorrect words
- incorrect numbers
- incorrect medical terms
- incorrect abbreviations
- whether the prediction preserves the meaning of the reference

Return ONLY valid JSON in exactly this structure:

{{
  "text_accuracy": 0.0,
  "missing_information": [],
  "incorrect_information": [],
  "explanation": ""
}}

Rules for text_accuracy:
- 1.0 = essentially identical
- 0.9 = extremely accurate
- 0.8 = mostly accurate with minor errors
- 0.7 = generally understandable but noticeable errors
- 0.5 = many errors
- 0.3 = substantial information is wrong or missing
- 0.0 = unusable

REFERENCE:
<<<
{reference}
>>>

PREDICTED:
<<<
{prediction}
>>>
"""


def extract_json(text):
    """
    Extract JSON from Qwen's response.

    Handles cases where Qwen accidentally wraps the JSON
    in markdown code fences or adds surrounding text.
    """

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    # First try the entire response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Then find the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"Could not find JSON object in Qwen response: {text}"
        )

    return json.loads(
        text[start:end + 1]
    )


def evaluate_with_qwen(reference, prediction):
    prompt = build_prompt(
        reference,
        prediction,
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
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

    raw_response = data["response"].strip()

    if not raw_response:
        raise RuntimeError(
            "Qwen returned an empty response."
        )

    evaluation = extract_json(raw_response)

    if not isinstance(evaluation, dict):
        raise ValueError(
            "Qwen evaluation was not a JSON object."
        )

    return evaluation


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def get_completed_records():
    """
    Read evaluation.jsonl and return AMR IDs that successfully
    completed evaluation.
    """

    if not OUTPUT_FILE.exists():
        return set()

    completed = set()

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

                if (
                    record.get("status") == "success"
                    and record.get("pair_id")
                ):
                    completed.add(
                        record["pair_id"]
                    )

            except json.JSONDecodeError:
                continue

    return completed


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_results():
    """
    Load successful MedGemma transcriptions from results.jsonl.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    records = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"Warning: invalid JSON on line "
                    f"{line_number}: {error}"
                )
                continue

            if record.get("status") != "success":
                continue

            pair_id = record.get("pair_id")
            prediction = record.get("transcription")

            if not pair_id or prediction is None:
                continue

            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_results()
    completed = get_completed_records()

    remaining = [
        record
        for record in records
        if record["pair_id"] not in completed
    ]

    print()
    print(f"Input records:    {len(records)}")
    print(f"Already evaluated: {len(completed)}")
    print(f"Remaining:        {len(remaining)}")
    print(f"Evaluator model:  {MODEL}")
    print(f"Truth directory:  {TRUTH_DIR}")
    print(f"Output:            {OUTPUT_FILE}")
    print()

    if not remaining:
        print("Nothing to evaluate.")
        return

    with OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
    ) as output:

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

            for record in remaining:
                pair_id = record["pair_id"]
                prediction = record["transcription"]

                progress.update(
                    task,
                    description=f"Evaluating {pair_id}",
                )

                start = time.perf_counter()

                try:
                    reference = load_truth(pair_id)

                    # Objective metrics.
                    cer = calculate_cer(
                        reference,
                        prediction,
                    )

                    wer = calculate_wer(
                        reference,
                        prediction,
                    )

                    # Qualitative LLM evaluation.
                    qwen_evaluation = evaluate_with_qwen(
                        reference,
                        prediction,
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    result = {
                        "pair_id": pair_id,
                        "status": "success",
                        "evaluator_model": MODEL,
                        "elapsed_seconds": round(
                            elapsed,
                            2,
                        ),
                        "cer": round(
                            cer,
                            4,
                        ),
                        "wer": round(
                            wer,
                            4,
                        ),
                        "qwen_evaluation": qwen_evaluation,
                    }

                    output.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    output.flush()

                    progress.console.print(
                        f"[green]✓ {pair_id}[/green] "
                        f"CER={cer:.3f} "
                        f"WER={wer:.3f} "
                        f"Qwen={qwen_evaluation.get('text_accuracy', 'N/A')}"
                    )

                except Exception as error:
                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    result = {
                        "pair_id": pair_id,
                        "status": "error",
                        "evaluator_model": MODEL,
                        "elapsed_seconds": round(
                            elapsed,
                            2,
                        ),
                        "error": str(error),
                    }

                    output.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    output.flush()

                    progress.console.print(
                        f"[red]✗ {pair_id} failed:[/red] "
                        f"{error}"
                    )

                progress.advance(task)

    print()
    print("Done.")
    print(f"Evaluation results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()