import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

# ============================================================
# Configuration
# ============================================================

BENCHMARK_DIR = Path("benchmarks")
OUTPUT_DIR = Path("structured_labels")

OLLAMA_URL = "http://localhost:11434/api/chat"

# The small, text-only models you want to use as parsers
PARSER_MODELS = [
    "osmosis:0.6b",
    "qwen2.5:3b",
]

# The original JSONL files you generated from the vision models
INPUT_FILES = [
    "medgemma1.5_4b.jsonl",
]

# ============================================================
# Your strict extraction prompt (copy-pasted from your message)
# ============================================================

EXTRACT_PROMPT = """
OUTPUT FORMAT IS STRICT.

You are a deterministic JSON extraction engine.

Your ONLY task is to extract structured information from a single chest X-ray report.

You MUST output ONLY valid JSON.
Your response MUST:
- Start with {
- End with }
- Contain nothing except JSON

DO NOT output:
- markdown
- ```json fences
- explanations
- reasoning
- analysis
- comments
- apologies
- recommendations
- medical advice
- any text outside JSON

You are NOT a doctor.
You are NOT providing clinical interpretation.
You are NOT generating a diagnosis.
You are ONLY extracting information explicitly present in the input report.

---

TASK:

Convert the chest X-ray report below into a single structured JSON object.

Return ONLY ONE JSON object.

Do not return an array.

The object MUST follow this schema exactly:

{
  "summary": "string",
  "entities": ["string"],
  "notes": ["string"]
}

Do not add new fields.
Do not remove fields.
Do not rename fields.

---

SUMMARY RULES:

- Produce a concise summary of the report.
- Preserve all clinically relevant findings.
- Preserve uncertainty words such as:
  - possible
  - may represent
  - suggests
  - cannot exclude
  - likely

- Remove repetition.
- Do not add information.
- Do not remove information.
- Do not reinterpret findings.

---

ENTITIES RULES:

Extract ONLY pathology terms that appear in the input report.

Valid entities are ONLY:

["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

MATCHING:

- A valid entity is considered present if its name appears in the input
  text, ignoring case, underscores, and spacing differences.
- The value you RETURN must always be copied verbatim from the valid
  entity list above.
- Do not match on partial words or unrelated terms.
- Do not map observations to related diagnoses.
- Do not use medical knowledge or clinical reasoning to infer an entity
  from a description that does not name it.
- Terms that describe a visual pattern but are NOT in the valid entity
  list (e.g. "reticular pattern", "hazy opacity", "linear scarring") are
  NOT extracted as entities.

SUMMARY/ENTITIES CONSISTENCY:

- If a valid entity term is named in the input report, it MUST also
  appear in `entities`.
- Do not describe a finding in the summary while omitting its matching entity.
- Do not include an entity that is unsupported by the summary.

If no valid entity name appears in the input:

Return ["No Finding"] for the entities field.

---

NOTES RULES:

Extract ONLY information explicitly mentioned in the report.

Examples:
- image limitations
- technical limitations
- comparison information
- positioning issues

Rules:
- Use [] when no notes or limitations exist.
- Never invent notes.

---

FORBIDDEN OUTPUT:

Never generate:
- diagnosis_suggestion
- recommended_action
- treatment
- follow-up instructions
- differential diagnosis
- possible causes
- clinical reasoning
- additional investigations
- prognosis

Only extract what exists in the report.

---

FINAL VALIDATION BEFORE ANSWERING:

Check:

1. Is the output valid JSON?
2. Is the output a single JSON object (not an array)?
3. Does the object match the provided schema?
4. Are there any keys not in the schema?
5. Did you add any information not explicitly present?
6. Is every entity string copied verbatim from the valid entity list?
7. Does every valid entity term named in the report appear in `entities`?
8. Does `summary` agree with `entities` (no contradictions)?

If any answer is yes (for 1-5) or no (for 6-8), fix the output before responding.

Return ONLY the JSON object.
"""
# ============================================================
# Helper functions
# ============================================================

def call_ollama_text(model: str, prompt: str) -> str:
    """Send a text-only request to Ollama and return the raw response."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format":"json"
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def parse_extraction_output(raw: str) -> Dict[str, Any]:
    """Attempt to parse the model's output as a single JSON object."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        else:
            raise ValueError("Output is not a JSON object")
    except json.JSONDecodeError as e:
        return {"_parse_error": raw, "_exception": str(e)}

def process_file(input_path: Path, parser_model: str, output_path: Path) -> None:
    """Read input JSONL, send each response to the parser model, save structured output."""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"  📄 Processing {input_path.name} with {parser_model} ({len(lines)} entries)")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines):
            try:
                original = json.loads(line)
            except json.JSONDecodeError:
                print(f"    ⚠️ Skipping malformed line {idx+1}")
                continue

            # Get the original vision model's response
            report_text = original.get("response", "")
            if not report_text:
                print(f"    ⚠️ No response field in line {idx+1}")
                continue

            # Truncate if too long (small models have limited context)
            if len(report_text) > 6000:
                report_text = report_text[:6000] + "... (truncated)"

            # Build the full prompt: extraction instructions + the report
            user_prompt = EXTRACT_PROMPT + "\n\n---\n\n" + report_text

            try:
                start = time.perf_counter()
                raw_output = call_ollama_text(parser_model, user_prompt)
                elapsed = time.perf_counter() - start

                # Try to parse the structured JSON
                parsed = parse_extraction_output(raw_output)

                # Prepare the output record
                result = {
                    "image": original["image"],
                    "source_model": original["model"],
                    "parser_model": parser_model,
                    "parsed": parsed,             # The extracted JSON array (or error object)
                    "raw_parser_output": raw_output,  # Keep for debugging
                    "elapsed_seconds": round(elapsed, 3),
                }

                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

                if (idx + 1) % 10 == 0:
                    print(f"      Processed {idx+1}/{len(lines)}")

            except Exception as e:
                print(f"      ❌ Error on {original.get('image', 'unknown')}: {e}")
                error_result = {
                    "image": original.get("image", "unknown"),
                    "source_model": original.get("model", "unknown"),
                    "parser_model": parser_model,
                    "error": str(e),
                }
                out_f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print(f"    ✅ Saved to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for input_filename in INPUT_FILES:
        input_path = BENCHMARK_DIR / input_filename
        if not input_path.exists():
            print(f"⚠️ Skipping {input_filename} (not found)")
            continue

        for model in PARSER_MODELS:
            # Build output filename: original_basename_parser_model.jsonl
            base = input_path.stem  # without .jsonl
            safe_model = model.replace(":", "_").replace("/", "_")
            output_filename = f"{base}_{safe_model}.jsonl"
            output_path = OUTPUT_DIR / output_filename

            process_file(input_path, model, output_path)

    print("\n🎉 All done! Check the 'structured_labels' folder.")


if __name__ == "__main__":
    main()