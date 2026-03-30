#!/usr/bin/env python3
"""
Stack AI → eval.py → GEPA-style reflection pipeline.

Flow:
  1. Load Stack AI export (extracted results + optional evaluator feedback).
  2. For each row: run eval.py (programmatic comparison vs ground truth).
  3. Aggregate scores and collect issues.
  4. Run GEPA-style reflection (LLM) to propose a new system prompt.
  5. Write new prompt and summary to results/.

Usage:
  # Single doc: ground truth and extraction JSON paths
  python run_pipeline.py --gt Samples/201414.json --pred Samples/201414_extracted.json --prompt "You are an expert..."

  # From Stack AI export (CSV with columns: doc_id, extraction_result, ground_truth, system_prompt [, evaluator_feedback])
  python run_pipeline.py --stack-ai-export results/batch_export.csv --prompt-col system_prompt

  # Reflection only (you already have eval report)
  python run_pipeline.py --eval-report results/201414_eval_report.json --prompt "You are an expert..."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure this script's dir is on path so we can import config and reflection
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config  # noqa: E402
from reflection import propose_prompt  # noqa: E402

# Add repo Scripts so we can import eval
_eval_dir = config.REPO_ROOT / "Scripts"
if _eval_dir.exists() and str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))
from eval import evaluate_extraction  # noqa: E402


def run_eval_single(gt_path: Path, pred_path: Path, eval_config: Optional[dict] = None) -> dict:
    """Load GT and prediction JSONs, run evaluate_extraction, return report."""
    with open(gt_path) as f:
        gt = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)
    return evaluate_extraction(gt, pred, eval_config)


def run_eval_from_stack_ai_export(
    export_path: Path,
    prompt_col: str = "system_prompt",
    extraction_col: str = "extraction_result",
    ground_truth_col: str = "ground_truth",
    feedback_col: Optional[str] = "evaluator_feedback",
    doc_id_col: str = "doc_id",
) -> Tuple[List[dict], Optional[str], Optional[str]]:
    """
    Read CSV from Stack AI Batch/Evaluator export.
    Each row: doc_id, extraction (JSON string or path), ground_truth (JSON string or path), system_prompt [, evaluator_feedback].
    Run eval for each row where we have both extraction and ground_truth.
    Returns (list of eval reports per row, aggregated prompt text, aggregated feedback text).
    """
    reports = []
    prompt_used = None
    feedback_parts = []

    with open(export_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_used = row.get(prompt_col) or prompt_used
            ext_raw = row.get(extraction_col) or row.get("out-0")
            gt_raw = row.get(ground_truth_col)
            fb = row.get(feedback_col or "")

            if not ext_raw or not gt_raw:
                continue

            # If values look like paths, load from file
            try:
                if gt_raw.strip().startswith("{"):
                    gt = json.loads(gt_raw)
                else:
                    with open(Path(gt_raw).expanduser()) as fp:
                        gt = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue
            try:
                if ext_raw.strip().startswith("{"):
                    pred = json.loads(ext_raw)
                else:
                    with open(Path(ext_raw).expanduser()) as fp:
                        pred = json.load(fp)
            except (json.JSONDecodeError, OSError):
                continue

            report = evaluate_extraction(gt, pred, None)
            report["_doc_id"] = row.get(doc_id_col, "")
            reports.append(report)
            if fb:
                feedback_parts.append(f"[{report.get('_doc_id', '')}] {fb}")

    aggregated_feedback = "\n".join(feedback_parts) if feedback_parts else None
    return reports, prompt_used, aggregated_feedback


def aggregate_reports(reports: List[dict]) -> dict:
    """Combine multiple eval reports into one for reflection (avg score, merged top issues)."""
    if not reports:
        return {"score": 0.0, "subscores": {}, "issues": []}

    total_score = sum(r.get("score", 0) for r in reports) / len(reports)
    subscores = reports[0].get("subscores", {})
    all_issues = []
    for r in reports:
        for issue in r.get("issues", [])[:15]:
            issue = dict(issue)
            path = issue.get("path", "")
            doc_id = r.get("_doc_id", "")
            if doc_id:
                issue["path"] = f"{doc_id}:{path}"
            all_issues.append(issue)
    all_issues.sort(key=lambda x: (-x.get("penalty", 0), x.get("path", "")))
    return {
        "score": total_score,
        "subscores": subscores,
        "issues": all_issues[:50],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stack AI + eval + GEPA reflection pipeline")
    ap.add_argument("--gt", help="Path to ground-truth JSON (single-doc mode)")
    ap.add_argument("--pred", help="Path to extracted/prediction JSON (single-doc mode)")
    ap.add_argument("--stack-ai-export", help="Path to Stack AI Batch/Evaluator CSV export")
    ap.add_argument("--eval-report", help="Path to existing eval report JSON (reflection only)")
    ap.add_argument("--prompt", default="", help="Current system prompt (for reflection; or from CSV if --stack-ai-export)")
    ap.add_argument("--prompt-col", default="system_prompt", help="CSV column name for prompt")
    ap.add_argument("--out-dir", default=None, help="Write results here (default: config.RESULTS_DIR)")
    ap.add_argument("--no-reflection", action="store_true", help="Only run eval, do not call reflection LLM")
    ap.add_argument(
        "--reflection-backend",
        choices=("openai", "stack_ai"),
        default="openai",
        help="Where to run reflection: openai (local API key) or stack_ai (Stack AI workflow, free tokens)",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    out_dir = Path(args.out_dir or config.RESULTS_DIR)

    eval_report: dict
    stack_ai_feedback: Optional[str] = None

    if args.eval_report:
        with open(args.eval_report) as f:
            eval_report = json.load(f)
    elif args.gt and args.pred:
        eval_report = run_eval_single(Path(args.gt), Path(args.pred))
        out_report = out_dir / "eval_report.json"
        with open(out_report, "w") as f:
            json.dump(eval_report, f, indent=2, default=str)
        print(f"Eval report written to {out_report}")
    elif args.stack_ai_export:
        reports, prompt_used, stack_ai_feedback = run_eval_from_stack_ai_export(
            Path(args.stack_ai_export),
            prompt_col=args.prompt_col,
        )
        eval_report = aggregate_reports(reports)
        if prompt_used:
            args.prompt = prompt_used
        if not args.prompt and not args.no_reflection:
            print("No prompt in CSV and --prompt not set; reflection needs current prompt.", file=sys.stderr)
            return 1
        out_report = out_dir / "eval_report_batch.json"
        with open(out_report, "w") as f:
            json.dump(eval_report, f, indent=2, default=str)
        print(f"Eval report (batch) written to {out_report}")
    else:
        print("Provide --gt + --pred, or --stack-ai-export, or --eval-report.", file=sys.stderr)
        return 1

    if not args.prompt and not args.no_reflection:
        print("--prompt required for reflection (or use --stack-ai-export with prompt column).", file=sys.stderr)
        return 1

    print(f"Score: {eval_report.get('score')}")

    if args.no_reflection:
        return 0

    if args.reflection_backend == "stack_ai":
        from reflection import propose_prompt_via_stack_ai  # noqa: E402
        new_prompt = propose_prompt_via_stack_ai(
            args.prompt,
            eval_report,
            stack_ai_feedback=stack_ai_feedback,
            org_id=config.STACK_AI_REFLECTION_ORG_ID,
            flow_id=config.STACK_AI_REFLECTION_FLOW_ID,
            base_url=config.STACK_AI_BASE_URL,
            api_key_env=config.STACK_AI_API_KEY_ENV,
        )
    else:
        new_prompt = propose_prompt(
            args.prompt,
            eval_report,
            stack_ai_feedback=stack_ai_feedback,
            model=config.REFLECTION_MODEL,
            api_key_env=config.OPENAI_API_KEY_ENV,
        )
    out_prompt = out_dir / "proposed_prompt.txt"
    with open(out_prompt, "w") as f:
        f.write(new_prompt)
    print(f"Proposed prompt written to {out_prompt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
