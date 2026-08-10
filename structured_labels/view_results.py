#!/usr/bin/env python3
"""
summarize_results.py

Reads the full results.json produced by compare_entities.py and writes a
compact basic_results.json containing only the per-combo aggregate stats
(no per-image breakdown).

Usage:
    python summarize_results.py --dir structured_labels
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=".", help="Folder containing results.json")
    args = ap.parse_args()

    base = Path(args.dir)
    in_path = base / "results.json"
    if not in_path.exists():
        raise SystemExit(f"results.json not found at {in_path}")

    with open(in_path, encoding="utf-8") as f:
        full = json.load(f)

    basic = {}
    for combo, data in full.items():
        s = data["summary"]
        basic[combo] = {
            "num_images": s["num_images_compared"],
            "num_scored": s["num_scored"],
            "num_parse_failures": s["num_parse_failures"],
            "parse_failure_rate": round(s["num_parse_failures"] / s["num_images_compared"], 4)
                                  if s["num_images_compared"] else None,
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "top_missing": s["top_missing_labels"][:5],
            "top_extra": s["top_extra_labels"][:5],
        }

    # sort by f1 descending so the best combo is on top
    basic = dict(sorted(basic.items(), key=lambda kv: -kv[1]["f1"]))

    out_path = base / "basic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(basic, f, indent=2)

    print(f"Wrote {out_path}\n")
    print(f"{'combo':45s} {'scored':>7s} {'fail%':>7s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for combo, s in basic.items():
        fail_pct = f"{s['parse_failure_rate']*100:.0f}%" if s['parse_failure_rate'] is not None else "-"
        print(f"{combo:45s} {s['num_scored']:>7d} {fail_pct:>7s} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f}")


if __name__ == "__main__":
    main()