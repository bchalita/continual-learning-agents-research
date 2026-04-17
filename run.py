#!/usr/bin/env python3
"""CLI entry point for the repair order extraction + GEPA optimization pipeline.

Usage:
  # Single document — eval only (no optimization loop)
  python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json

  # Single document — optimization loop
  python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json --iterations 4

  # All 6 sample documents — eval only
  python run.py --all

  # All 6 sample documents — optimization loop
  python run.py --all --iterations 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure pipeline package is importable from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config
from pipeline.orchestrator import extract
from pipeline.evaluator import evaluate
from pipeline.loop import run as run_loop
import anthropic


def _run_single_eval(doc_id: str, pdf_path: Path, gt_path: Path, client: anthropic.Anthropic) -> None:
    """Extract + evaluate once (no prompt optimization)."""
    extraction_prompt = (config.PROMPTS_DIR / "extraction.txt").read_text(encoding="utf-8").strip()

    print(f"[{doc_id}] Extracting...")
    prediction = extract(pdf_path, extraction_prompt, client)

    config.ensure_dirs()
    pred_path = config.RESULTS_DIR / f"{doc_id}_prediction.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, default=str)
    print(f"[{doc_id}] Prediction saved: {pred_path}")

    with open(gt_path, encoding="utf-8") as f:
        ground_truth = json.load(f)

    print(f"[{doc_id}] Evaluating...")
    eval_result = evaluate(prediction, ground_truth, client)
    eval_report = eval_result["eval_report"]
    score = eval_report.get("score", 0.0)
    subscores = eval_report.get("subscores", {})

    print(f"[{doc_id}] Score: {score:.4f}")
    print(f"         structure={subscores.get('structure'):.4f}, "
          f"numbers={subscores.get('numbers'):.4f}, "
          f"text={subscores.get('text'):.4f}")
    print(f"[{doc_id}] Diagnosis: {eval_result.get('qualitative_feedback', '')}")

    report_path = config.RESULTS_DIR / f"{doc_id}_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, default=str)
    print(f"[{doc_id}] Eval report saved: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair order extraction + GEPA optimization pipeline")
    parser.add_argument("--doc", type=Path, help="Path to a single PDF file")
    parser.add_argument("--gt", type=Path, help="Path to the ground-truth JSON for --doc")
    parser.add_argument("--all", action="store_true", help="Run on all 6 samples in Data/Samples/")
    parser.add_argument("--iterations", type=int, default=0,
                        help="Number of optimization iterations (0 = eval only, no reflection)")
    args = parser.parse_args()

    if not args.doc and not args.all:
        parser.error("Provide --doc or --all")
    if args.doc and not args.gt:
        parser.error("--gt is required when using --doc")

    if not config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    jobs: list[tuple[str, Path, Path]] = []
    if args.all:
        for pdf in sorted(config.SAMPLES_DIR.glob("*.pdf")):
            gt = pdf.with_suffix(".json")
            if gt.exists():
                jobs.append((pdf.stem, pdf, gt))
    else:
        jobs.append((args.doc.stem, args.doc, args.gt))

    for doc_id, pdf_path, gt_path in jobs:
        if args.iterations > 0:
            run_loop(doc_id, pdf_path, gt_path, iterations=args.iterations)
        else:
            _run_single_eval(doc_id, pdf_path, gt_path, client)

    return 0


if __name__ == "__main__":
    sys.exit(main())
