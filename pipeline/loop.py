from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import anthropic

from . import config
from .orchestrator import extract
from .evaluator import evaluate
from .reflection import propose


def run(
    doc_id: str,
    pdf_path: Path,
    gt_path: Path,
    iterations: int,
    output_dir: Path | None = None,
    extraction_prompt: str | None = None,
) -> Path:
    """Run the GEPA optimization loop for a single document.

    Each iteration:
      1. Extract JSON from PDF using current extraction prompt
      2. Evaluate against ground truth (deterministic + LLM)
      3. Reflect → propose new extraction prompt
      4. Save row to CSV

    Returns path to the results CSV.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    config.ensure_dirs()

    out_dir = Path(output_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if extraction_prompt is None:
        extraction_prompt = (config.PROMPTS_DIR / "extraction.txt").read_text(encoding="utf-8").strip()

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"{doc_id}_optimization_{timestamp}.csv"

    fieldnames = [
        "iteration", "score", "structure", "numbers", "text",
        "top_issues", "qualitative_feedback", "extraction_prompt_path",
    ]

    rows = []
    prev_score: float | None = None

    for i in range(1, iterations + 1):
        print(f"\n[{doc_id}] Iteration {i}/{iterations} — extracting...")
        prediction = extract(pdf_path, extraction_prompt, client)

        print(f"[{doc_id}] Iteration {i}/{iterations} — evaluating...")
        eval_result = evaluate(prediction, ground_truth, client)
        eval_report = eval_result["eval_report"]
        score = eval_report.get("score", 0.0)
        subscores = eval_report.get("subscores", {})
        top_issues = eval_report.get("issues", [])[:5]

        delta_str = ""
        if prev_score is not None:
            delta = score - prev_score
            delta_str = f" (Δ {delta:+.4f})"
        print(f"[{doc_id}] Iteration {i}: score={score:.4f}{delta_str}")
        print(f"         structure={subscores.get('structure'):.4f}, "
              f"numbers={subscores.get('numbers'):.4f}, "
              f"text={subscores.get('text'):.4f}")
        prev_score = score

        ext_prompt_path = prompts_dir / f"{doc_id}_iter{i}_extraction.txt"
        ext_prompt_path.write_text(extraction_prompt, encoding="utf-8")

        pred_path = out_dir / f"{doc_id}_iter{i}_prediction.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2, default=str)

        rows.append({
            "iteration": i,
            "score": f"{score:.4f}",
            "structure": f"{subscores.get('structure', 0):.4f}",
            "numbers": f"{subscores.get('numbers', 0):.4f}",
            "text": f"{subscores.get('text', 0):.4f}",
            "top_issues": json.dumps(
                [{"path": x["path"], "kind": x["kind"], "penalty": x["penalty"]} for x in top_issues]
            ),
            "qualitative_feedback": eval_result.get("qualitative_feedback", "")[:500],
            "extraction_prompt_path": str(ext_prompt_path),
        })

        if i < iterations:
            print(f"[{doc_id}] Iteration {i}/{iterations} — reflecting...")
            new_prompts = propose(extraction_prompt, eval_result, client)
            extraction_prompt = new_prompts["extraction_prompt"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[{doc_id}] Done. Results: {csv_path}")
    return csv_path
