"""
Optimization loop: single-doc (Erwin's original) and multi-doc (GEPA).

Single-doc mode (`run`): iterate on one document, linearly replacing prompts.
Multi-doc mode (`run_gepa`): evaluate each prompt on ALL documents, maintain
a Pareto frontier of non-dominated prompts, select parents via win-frequency.

API call routing:
  - Vision calls (structure, extraction) → Anthropic API (Haiku 4.5)
  - Text-only calls (eval diagnosis, reflection, merge) → Stack AI (Opus 4.6)

Logging:
  - CSV: compact per-iteration scores (spreadsheet-friendly)
  - JSON log: rich per-iteration details — prompt diffs, per-doc deltas,
    what improved/worsened, reflection reasoning, top issues
"""

from __future__ import annotations

import csv
import difflib
import json
from datetime import datetime
from pathlib import Path

import anthropic

from . import config
from .orchestrator import extract
from .evaluator import evaluate
from .reflection import propose
from .gepa import PromptPool


def _prompt_diff_summary(old_text: str, new_text: str) -> dict:
    """Generate a human-readable summary of what changed between two prompts."""
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, n=1))

    added = [l.rstrip() for l in diff if l.startswith("+") and not l.startswith("+++")]
    removed = [l.rstrip() for l in diff if l.startswith("-") and not l.startswith("---")]

    return {
        "lines_added": len(added),
        "lines_removed": len(removed),
        "added_lines": added[:20],  # cap at 20 for readability
        "removed_lines": removed[:20],
        "full_diff": "".join(diff) if len(diff) < 200 else "(diff too large, see prompt files)",
    }


def _score_deltas(
    candidate_scores: dict[str, float],
    candidate_subscores: dict[str, dict],
    parent_scores: dict[str, float],
    parent_subscores: dict[str, dict],
) -> dict:
    """Compute per-doc score deltas between candidate and parent."""
    deltas = {}
    for doc_id in candidate_scores:
        c_score = candidate_scores[doc_id]
        p_score = parent_scores.get(doc_id, 0)
        delta = c_score - p_score

        c_sub = candidate_subscores.get(doc_id, {})
        p_sub = parent_subscores.get(doc_id, {})
        sub_deltas = {}
        for key in ["structure", "numbers", "text"]:
            sub_deltas[key] = round(c_sub.get(key, 0) - p_sub.get(key, 0), 4)

        status = "improved" if delta > 0.005 else ("worsened" if delta < -0.005 else "unchanged")
        deltas[doc_id] = {
            "score": round(c_score, 4),
            "parent_score": round(p_score, 4),
            "delta": round(delta, 4),
            "status": status,
            "subscores": c_sub,
            "subscore_deltas": sub_deltas,
        }
    return deltas


def run(
    doc_id: str,
    pdf_path: Path,
    gt_path: Path,
    iterations: int,
    output_dir: Path | None = None,
    extraction_prompt: str | None = None,
) -> Path:
    """Run the single-doc optimization loop (Erwin's original).

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


def run_gepa(
    doc_jobs: list[tuple[str, Path, Path]],
    iterations: int,
    output_dir: Path | None = None,
    extraction_prompt: str | None = None,
) -> Path:
    """Run multi-document GEPA optimization loop.

    Evaluates each prompt candidate on ALL documents, maintains a Pareto
    frontier, and selects parents via win-frequency for reflective mutation.

    Args:
        doc_jobs: List of (doc_id, pdf_path, gt_path) tuples.
        iterations: Number of GEPA iterations (each produces one new prompt).
        output_dir: Where to save results.
        extraction_prompt: Starting prompt (reads extraction.txt if None).

    Pipeline per iteration:
      1. Select parent from Pareto front (win-frequency weighted)
      2. Reflect on parent's worst doc → propose new prompt
      3. Evaluate new prompt on ALL documents
      4. Add to pool, update Pareto frontier
      5. Log scores

    API cost per iteration:
      - len(doc_jobs) × extraction calls (Haiku 4.5 vision via Anthropic)
      - len(doc_jobs) × eval diagnosis calls (Opus 4.6 text via Stack AI)
      - 1 × reflection call (Opus 4.6 text via Stack AI)

    Returns path to the results directory.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    config.ensure_dirs()

    out_dir = Path(output_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if extraction_prompt is None:
        extraction_prompt = (config.PROMPTS_DIR / "extraction.txt").read_text(encoding="utf-8").strip()

    # Load all ground truths
    ground_truths: dict[str, dict] = {}
    for doc_id, _, gt_path in doc_jobs:
        with open(gt_path, encoding="utf-8") as f:
            ground_truths[doc_id] = json.load(f)

    doc_ids = [doc_id for doc_id, _, _ in doc_jobs]

    # Initialize prompt pool with the base prompt
    pool = PromptPool()
    pool.add(extraction_prompt, parent_id=None, iteration=0)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"gepa_optimization_{timestamp}.csv"
    log_path = out_dir / f"gepa_log_{timestamp}.json"

    fieldnames = (
        ["iteration", "prompt_id", "parent_id", "mean_score", "min_score"]
        + [f"{d}_score" for d in doc_ids]
        + ["pareto_front_size", "pool_size", "prompt_path"]
    )

    rows: list[dict] = []
    iteration_logs: list[dict] = []  # Rich log for analysis

    for i in range(1, iterations + 1):
        iter_log: dict = {
            "iteration": i,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # ── Select parent + generate candidate ──
        if i == 1:
            # First iteration: evaluate the base prompt as-is
            candidate = pool.candidates[0]
            print(f"\n{'='*60}")
            print(f"GEPA Iteration {i}/{iterations} — evaluating base prompt ({candidate.prompt_id})")
            iter_log["type"] = "baseline"
            iter_log["prompt_id"] = candidate.prompt_id
        else:
            # Select parent from Pareto front
            parent = pool.select_parent()
            worst_doc = parent.worst_doc()
            print(f"\n{'='*60}")
            print(f"GEPA Iteration {i}/{iterations}")
            print(f"  Parent: {parent.prompt_id} (mean={parent.mean_score:.4f}, worst={worst_doc})")

            # Reflect on the parent's worst-performing document
            worst_eval = parent.eval_results.get(worst_doc, {})
            worst_issues = worst_eval.get("eval_report", {}).get("issues", [])[:10]
            worst_feedback = worst_eval.get("qualitative_feedback", "")

            iter_log["type"] = "mutation"
            iter_log["parent_id"] = parent.prompt_id
            iter_log["parent_mean_score"] = parent.mean_score
            iter_log["reflection_target_doc"] = worst_doc
            iter_log["reflection_target_score"] = parent.scores.get(worst_doc, 0)
            iter_log["reflection_target_feedback"] = worst_feedback
            iter_log["reflection_target_top_issues"] = [
                {"path": x["path"], "kind": x["kind"], "category": x["category"],
                 "expected": str(x.get("expected", ""))[:100],
                 "got": str(x.get("got", ""))[:100],
                 "penalty": x["penalty"]}
                for x in worst_issues
            ]

            print(f"  Reflecting on {worst_doc} (score={parent.scores.get(worst_doc, 0):.4f})...")
            print(f"  Feedback: {worst_feedback[:200]}...")

            new_prompts = propose(parent.text, worst_eval, client)
            new_text = new_prompts["extraction_prompt"]

            # Log prompt diff
            diff_info = _prompt_diff_summary(parent.text, new_text)
            iter_log["prompt_diff"] = diff_info
            print(f"  Prompt changes: +{diff_info['lines_added']} / -{diff_info['lines_removed']} lines")

            candidate = pool.add(new_text, parent_id=parent.prompt_id, iteration=i)
            iter_log["prompt_id"] = candidate.prompt_id
            print(f"  New candidate: {candidate.prompt_id}")

        # ── Evaluate candidate on ALL documents ──
        doc_details: dict[str, dict] = {}
        for doc_id, pdf_path, gt_path in doc_jobs:
            print(f"  [{doc_id}] Extracting...")
            prediction = extract(pdf_path, candidate.text, client)

            # Save prediction
            pred_path = out_dir / f"{candidate.prompt_id}_{doc_id}_prediction.json"
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(prediction, f, indent=2, default=str)

            print(f"  [{doc_id}] Evaluating...")
            eval_result = evaluate(prediction, ground_truths[doc_id], client)
            eval_report = eval_result["eval_report"]
            score = eval_report.get("score", 0.0)
            subscores = eval_report.get("subscores", {})

            candidate.scores[doc_id] = score
            candidate.subscores[doc_id] = subscores
            candidate.eval_results[doc_id] = eval_result

            # Section alignment info
            sa = eval_report.get("section_alignment", {})

            print(f"  [{doc_id}] score={score:.4f} "
                  f"(S={subscores.get('structure', 0):.3f} "
                  f"N={subscores.get('numbers', 0):.3f} "
                  f"T={subscores.get('text', 0):.3f})")

            # Per-doc detail for the log
            top_issues = eval_report.get("issues", [])[:5]
            doc_details[doc_id] = {
                "score": score,
                "subscores": subscores,
                "section_alignment": {
                    "matched": sa.get("matched", []),
                    "missing": sa.get("missing_from_prediction", []),
                    "extra": sa.get("extra_in_prediction", []),
                },
                "qualitative_feedback": eval_result.get("qualitative_feedback", ""),
                "top_issues": [
                    {"path": x["path"], "kind": x["kind"], "category": x["category"],
                     "penalty": x["penalty"]}
                    for x in top_issues
                ],
                "num_sections_predicted": len(prediction.get("sections", [])),
            }

        # ── Save prompt ──
        prompt_path = prompts_dir / f"{candidate.prompt_id}_extraction.txt"
        prompt_path.write_text(candidate.text, encoding="utf-8")

        # ── Compute deltas vs parent ──
        if i > 1:
            deltas = _score_deltas(
                candidate.scores, candidate.subscores,
                parent.scores, parent.subscores,
            )
            iter_log["deltas"] = deltas

            mean_delta = candidate.mean_score - parent.mean_score
            improved_docs = [d for d, v in deltas.items() if v["status"] == "improved"]
            worsened_docs = [d for d, v in deltas.items() if v["status"] == "worsened"]
            unchanged_docs = [d for d, v in deltas.items() if v["status"] == "unchanged"]

            print(f"\n  vs parent {parent.prompt_id}:")
            print(f"    Mean Δ: {mean_delta:+.4f}")
            if improved_docs:
                print(f"    Improved: {improved_docs}")
                for d in improved_docs:
                    print(f"      {d}: {deltas[d]['parent_score']:.4f} → {deltas[d]['score']:.4f} "
                          f"(Δ{deltas[d]['delta']:+.4f})")
            if worsened_docs:
                print(f"    Worsened: {worsened_docs}")
                for d in worsened_docs:
                    print(f"      {d}: {deltas[d]['parent_score']:.4f} → {deltas[d]['score']:.4f} "
                          f"(Δ{deltas[d]['delta']:+.4f})")
            if unchanged_docs:
                print(f"    Unchanged: {unchanged_docs}")

        # ── Log ──
        front = pool.pareto_front
        iter_log["mean_score"] = candidate.mean_score
        iter_log["min_score"] = candidate.min_score
        iter_log["scores"] = dict(candidate.scores)
        iter_log["subscores"] = dict(candidate.subscores)
        iter_log["doc_details"] = doc_details
        iter_log["pareto_front"] = [c.prompt_id for c in front]
        iter_log["win_frequencies"] = pool.win_frequencies()
        iter_log["prompt_path"] = str(prompt_path)
        iteration_logs.append(iter_log)

        print(f"\n  Mean={candidate.mean_score:.4f}, Min={candidate.min_score:.4f}")
        print(f"  Pareto front: {[c.prompt_id for c in front]} ({len(front)} prompts)")

        row = {
            "iteration": i,
            "prompt_id": candidate.prompt_id,
            "parent_id": candidate.parent_id or "",
            "mean_score": f"{candidate.mean_score:.4f}",
            "min_score": f"{candidate.min_score:.4f}",
            "pareto_front_size": len(front),
            "pool_size": len(pool.candidates),
            "prompt_path": str(prompt_path),
        }
        for doc_id in doc_ids:
            row[f"{doc_id}_score"] = f"{candidate.scores.get(doc_id, 0):.4f}"
        rows.append(row)

        # Write CSV + log incrementally (survive crashes)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"iterations": iteration_logs}, f, indent=2, default=str)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("GEPA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    best = pool.best_prompt
    if best:
        print(f"Best prompt: {best.prompt_id} (mean={best.mean_score:.4f})")
        for doc_id in doc_ids:
            print(f"  {doc_id}: {best.scores.get(doc_id, 0):.4f}")

    front = pool.pareto_front
    print(f"\nPareto front ({len(front)} prompts):")
    for c in front:
        wins = pool.win_frequencies().get(c.prompt_id, 0)
        print(f"  {c.prompt_id}: mean={c.mean_score:.4f}, wins={wins}")

    # Save pool state
    pool_path = out_dir / f"gepa_pool_{timestamp}.json"
    with open(pool_path, "w", encoding="utf-8") as f:
        f.write(pool.to_json())
    print(f"\nPool saved: {pool_path}")
    print(f"CSV saved: {csv_path}")
    print(f"Detailed log: {log_path}")

    return out_dir
