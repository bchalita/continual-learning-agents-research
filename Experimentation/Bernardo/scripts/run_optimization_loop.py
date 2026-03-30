#!/usr/bin/env python3
"""
Run the Stack AI flow N times on the same document, using the Reflection-proposed prompt
as the next run's input. Tracks baseline → iter1 → iter2 → ... and saves scores,
evaluator feedback, and proposed prompts to a CSV (and prompt files).

Usage:
  cd Experimentation/Bernardo/scripts
  export STACK_AI_API_KEY="your-token"
  python3 run_optimization_loop.py \\
    --doc-url "https://...?dl=1" \\
    --ground-truth-file ../Samples/201414.json \\
    --prompt-file baseline_prompt.txt \\
    --iterations 4 \\
    --output-csv ../results/optimization_run.csv

CSV columns: iteration, eval_score, llm_evaluator_score, marginal_eval, marginal_llm,
evaluator_feedback, proposed_prompt_path, log_path
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config
from run_extraction_api import run_extraction, _normalize_doc_url


def _parse_full_response(response):
    """
    Extract eval score, LLM evaluator score, evaluator feedback text, and proposed prompt
    from the Stack AI response. Handles combined out-0 string or separate output keys.
    Returns (eval_score, llm_evaluator_score, evaluator_feedback, proposed_prompt).
    """
    eval_score = None
    llm_evaluator_score = None
    evaluator_feedback = ""
    proposed_prompt = ""

    out = response.get("outputs") or {}
    raw = out.get("out-0") if isinstance(out, dict) else response.get("out-0")
    if not isinstance(raw, str):
        return (eval_score, llm_evaluator_score, evaluator_feedback, proposed_prompt)

    # Eval score (deterministic): 'score': 0.8155 or "score": 0.8155
    m = re.search(r"['\"]score['\"]\s*:\s*([\d.]+)", raw)
    if m:
        try:
            eval_score = float(m.group(1))
        except ValueError:
            pass

    # LLM Evaluator score: "Score: 0.0" or "Score: 1.0" before "Feedback:"
    llm_score_m = re.search(r"\bScore:\s*([\d.]+)", raw)
    if llm_score_m:
        try:
            llm_evaluator_score = float(llm_score_m.group(1))
        except ValueError:
            pass

    # Evaluator: "Score: 0.0\n\nFeedback: ..." or "Feedback: ..."
    idx = raw.find("Feedback:")
    if idx != -1:
        evaluator_feedback = raw[idx + len("Feedback:") :].strip()
        if len(evaluator_feedback) > 2000:
            evaluator_feedback = evaluator_feedback[:2000] + "..."

    # Proposed prompt: long block starting with "You are an expert at extracting"
    start_marker = "You are an expert at extracting"
    start = raw.find(start_marker)
    if start != -1:
        end = raw.find(" Score: ", start)
        if end == -1:
            end = raw.find("Feedback:", start)
        if end != -1:
            proposed_prompt = raw[start:end].strip()
        else:
            proposed_prompt = raw[start:].strip()
        if len(proposed_prompt) > 50000:
            proposed_prompt = proposed_prompt[:50000]

    return (eval_score, llm_evaluator_score, evaluator_feedback, proposed_prompt)


def _write_log_and_get_path(response, payload_sent, logs_dir, marginal_eval=None, marginal_llm=None):
    """Write request/response to a timestamped log file; optionally include marginal improvement. Return path."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = logs_dir / f"api_run_{ts}.json"
    log_data = {
        "timestamp_utc": ts,
        "request": payload_sent,
        "response": response,
    }
    if marginal_eval is not None or marginal_llm is not None:
        log_data["marginal_improvement"] = {
            "eval_score": marginal_eval,
            "llm_evaluator_score": marginal_llm,
        }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)
    return log_path


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run N iterations: same document, prompt evolves from Reflection output; track scores and save CSV."
    )
    ap.add_argument("--doc-url", required=True, help="Public URL of the PDF (e.g. Dropbox with dl=1).")
    ap.add_argument("--ground-truth-file", type=Path, required=True, help="Path to ground truth JSON (in-1).")
    ap.add_argument("--prompt-file", type=Path, default=_here / "baseline_prompt.txt", help="Baseline system prompt file.")
    ap.add_argument("--iterations", type=int, default=4, help="Number of iterations (default 4).")
    ap.add_argument("--output-csv", type=Path, default=None, help="Output CSV path (default: results/optimization_<timestamp>.csv).")
    ap.add_argument("--output-dir", type=Path, default=None, help="Directory for CSV and prompt files (default: config.RESULTS_DIR).")
    ap.add_argument("--doc-as-string", action="store_true", help="Send doc-0 as a single URL string instead of [url]. Use if the flow ignores the document when sent as an array.")
    args = ap.parse_args()

    config.ensure_dirs()
    out_dir = args.output_dir or config.RESULTS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _here / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.ground_truth_file.exists():
        print(f"ERROR: Ground truth file not found: {args.ground_truth_file}", file=sys.stderr)
        sys.exit(1)
    ground_truth = args.ground_truth_file.read_text(encoding="utf-8")

    if not args.prompt_file.exists():
        print(f"ERROR: Prompt file not found: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)
    current_prompt = args.prompt_file.read_text(encoding="utf-8").strip()

    doc_url = _normalize_doc_url(args.doc_url) if "dropbox" in args.doc_url.lower() else args.doc_url

    csv_path = args.output_csv or (out_dir / f"optimization_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
    rows = []
    header = [
        "iteration",
        "eval_score",
        "llm_evaluator_score",
        "marginal_eval",
        "marginal_llm",
        "evaluator_feedback",
        "proposed_prompt_path",
        "log_path",
    ]

    prev_eval_score = None
    prev_llm_score = None

    print(f"Baseline prompt: {args.prompt_file} ({len(current_prompt)} chars)")
    print(f"Document: {doc_url[:60]}...")
    print(f"Iterations: {args.iterations}")
    print(f"Output CSV: {csv_path}")
    print("---")

    for i in range(1, args.iterations + 1):
        print(f"[Iteration {i}/{args.iterations}] Calling API with current prompt...")
        try:
            response, payload_sent = run_extraction(
                current_prompt,
                doc_url=doc_url,
                ground_truth=ground_truth,
                doc_as_list=not args.doc_as_string,
            )
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            rows.append([i, "", "", "", "", f"API error: {e}", "", ""])
            break

        eval_score, llm_evaluator_score, evaluator_feedback, proposed_prompt = _parse_full_response(response)

        marginal_eval = None
        marginal_llm = None
        if prev_eval_score is not None and eval_score is not None:
            marginal_eval = round(eval_score - prev_eval_score, 4)
        if prev_llm_score is not None and llm_evaluator_score is not None:
            marginal_llm = round(llm_evaluator_score - prev_llm_score, 4)

        log_path = _write_log_and_get_path(
            response, payload_sent, logs_dir,
            marginal_eval=marginal_eval,
            marginal_llm=marginal_llm,
        )
        print(f"  Log: {log_path}")
        print(f"  Eval score: {eval_score}  |  LLM evaluator score: {llm_evaluator_score}")
        if marginal_eval is not None or marginal_llm is not None:
            print(f"  Marginal improvement: eval {marginal_eval:+g}  |  LLM evaluator {marginal_llm:+g}")

        prompt_path = ""
        if proposed_prompt:
            prompt_path = prompts_dir / f"iter_{i}_proposed.txt"
            prompt_path.write_text(proposed_prompt, encoding="utf-8")
            prompt_path = str(prompt_path)

        row = [
            i,
            str(eval_score) if eval_score is not None else "",
            str(llm_evaluator_score) if llm_evaluator_score is not None else "",
            str(marginal_eval) if marginal_eval is not None else "",
            str(marginal_llm) if marginal_llm is not None else "",
            (evaluator_feedback or "")[:500],
            prompt_path,
            str(log_path),
        ]
        rows.append(row)
        print(f"  Proposed prompt saved: {prompt_path or 'no'}")

        if not proposed_prompt.strip():
            print("  No proposed prompt; stopping.")
            break

        prev_eval_score = eval_score
        prev_llm_score = llm_evaluator_score
        current_prompt = proposed_prompt

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"Done. CSV: {csv_path}")


if __name__ == "__main__":
    main()
