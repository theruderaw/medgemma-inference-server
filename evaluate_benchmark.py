#!/usr/bin/env python3
"""
compare_entities.py

Compares extracted entities from 6 (source_model, parser_model) jsonl files
against ground-truth labels in labels.csv.

Usage:
    python compare_entities.py --dir /path/to/structured_labels

Expects:
    <dir>/labels.csv
    <dir>/*.jsonl   (any number of jsonl files matching the parsed-output schema)

Output:
    <dir>/results.json
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical vocabulary (ChestX-ray14 style, matches labels.csv)
# ---------------------------------------------------------------------------
CANONICAL_LABELS = [
    "No Finding", "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

# Normalize a raw string -> canonical form.
# Handles case, whitespace/underscore variants, and known exact synonyms.
# NOTE: this list is intentionally conservative. Terms that are clinically
# distinct from the canonical vocabulary (e.g. "Enlarged Cardiomediastinum",
# "Lung Opacity", "Lung Lesion", "Fracture", "Support Devices") are NOT
# auto-mapped onto a canonical label -- they will surface as "extra" so you
# can see them and decide whether to add an alias below.
SYNONYM_MAP = {
    "no finding": "No Finding",
    "normal": "No Finding",
    "pleural thickening": "Pleural_Thickening",
    "pleural_thickening": "Pleural_Thickening",
}

def normalize(label: str) -> str:
    if not isinstance(label, str):
        return ""
    key = label.strip().lower().replace("_", " ")
    key = re.sub(r"\s+", " ", key)
    if key in SYNONYM_MAP:
        return SYNONYM_MAP[key]
    # Try direct match against canonical list (case-insensitive, underscore-insensitive)
    for canon in CANONICAL_LABELS:
        if key == canon.lower().replace("_", " "):
            return canon
    # Unknown / unmapped term -- keep a cleaned-up version, title-cased,
    # so it's still comparable/groupable in the "extra" report.
    return label.strip()


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def load_ground_truth(labels_csv: Path) -> dict:
    gt = {}
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            raw_labels = row["labels"].strip()
            labels = {normalize(l) for l in raw_labels.split(";") if l.strip()}
            gt[fname] = labels
    return gt


# ---------------------------------------------------------------------------
# Entity extraction from a parsed record
# ---------------------------------------------------------------------------
def extract_entities(record: dict) -> tuple[set, str, str]:
    """
    Returns (entity_set, extraction_method, error_message)
    extraction_method is:
      - 'structured'   if parsed.entities was present and usable
      - 'parse_failed' if parsed.entities was missing (upstream JSON parse failure) --
                        no keyword fallback is attempted; predicted set is empty
                        and the record is flagged as an error rather than scored.
    error_message is the parser's own error text when available, else a generic one.
    """
    parsed = record.get("parsed", {})

    if isinstance(parsed, dict):
        ents = parsed.get("entities")
        if isinstance(ents, list) and len(ents) > 0:
            entities = {normalize(e) for e in ents if isinstance(e, str) and e.strip()}
            return entities, "structured", ""

    # No usable entities list -> this is a parse failure, not a prediction.
    error_message = "parse_failed: no 'entities' field found"
    if isinstance(parsed, dict):
        exc = parsed.get("_exception")
        if exc:
            error_message = f"parse_failed: {exc}"

    return set(), "parse_failed", error_message


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare(predicted: set, truth: set) -> dict:
    exact = sorted(predicted & truth)
    missing = sorted(truth - predicted)
    extra = sorted(predicted - truth)
    return {"exact_matches": exact, "missing": missing, "extra": extra}


def prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compare extracted entities against ground truth labels.")
    ap.add_argument("--dir", type=str, default=".", help="Folder containing labels.csv and *.jsonl files")
    args = ap.parse_args()

    base = Path(args.dir)
    labels_csv = base / "labels.csv"
    if not labels_csv.exists():
        sys.exit(f"labels.csv not found at {labels_csv}")

    jsonl_files = sorted(base.glob("*.jsonl"))
    if not jsonl_files:
        sys.exit(f"No .jsonl files found in {base}")

    ground_truth = load_ground_truth(labels_csv)
    print(f"Loaded ground truth for {len(ground_truth)} images.")
    print(f"Found {len(jsonl_files)} jsonl files: {[f.name for f in jsonl_files]}")

    all_results = {}

    for jf in jsonl_files:
        combo_name = jf.stem  # e.g. medgemma-vision_latest_osmosis_0.6b
        per_image = {}
        tp_total = fp_total = fn_total = 0
        parse_method_counts = defaultdict(int)
        missing_counter = defaultdict(int)
        extra_counter = defaultdict(int)
        images_with_no_gt = []
        parse_failures = 0

        with open(jf, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [warn] {jf.name} line {lineno}: could not parse JSON line ({e}) -- skipped")
                    continue

                image = record.get("image", f"<line {lineno}>")
                if image not in ground_truth:
                    images_with_no_gt.append(image)
                    continue

                truth = ground_truth[image]
                predicted, method, error_message = extract_entities(record)
                parse_method_counts[method] += 1

                if method == "parse_failed":
                    parse_failures += 1
                    per_image[image] = {
                        "error": error_message,
                        "exact_matches": None,
                        "missing": None,
                        "extra": None,
                        "predicted": None,
                        "ground_truth": sorted(truth),
                        "extraction_method": method,
                    }
                    # Not scored: excluded from precision/recall totals below.
                    continue

                cmp = compare(predicted, truth)
                per_image[image] = {
                    **cmp,
                    "predicted": sorted(predicted),
                    "ground_truth": sorted(truth),
                    "extraction_method": method,
                }

                tp_total += len(cmp["exact_matches"])
                fp_total += len(cmp["extra"])
                fn_total += len(cmp["missing"])
                for m in cmp["missing"]:
                    missing_counter[m] += 1
                for e in cmp["extra"]:
                    extra_counter[e] += 1

        summary = {
            "num_images_compared": len(per_image),
            "num_scored": len(per_image) - parse_failures,
            "num_parse_failures": parse_failures,
            "images_missing_ground_truth": images_with_no_gt,
            "extraction_method_counts": dict(parse_method_counts),
            "totals": {"true_positive": tp_total, "false_positive_extra": fp_total, "false_negative_missing": fn_total},
            **prf1(tp_total, fp_total, fn_total),
            "note": "precision/recall/f1 computed only over num_scored images; parse_failed images are excluded, not counted as 0",
            "top_missing_labels": sorted(missing_counter.items(), key=lambda x: -x[1]),
            "top_extra_labels": sorted(extra_counter.items(), key=lambda x: -x[1]),
        }

        all_results[combo_name] = {
            "summary": summary,
            "per_image": per_image,
        }

        print(f"  {combo_name}: n={summary['num_images_compared']} "
              f"P={summary['precision']} R={summary['recall']} F1={summary['f1']}")

    out_path = base / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()