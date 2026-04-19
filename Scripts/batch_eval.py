#!/usr/bin/env python3
"""
Batch evaluation runner: score predictions against ground truth for all sample documents.

Runs eval.py on each (prediction, ground_truth) pair and produces a summary table
with per-document scores, per-section alignment, and aggregate statistics.

This script is the evaluation backbone for the GEPA optimization loop — every
mutation attempt needs scores across all documents to update the Pareto frontier.

Usage:
  # Score all predictions in a directory against ground truth:
  python Scripts/batch_eval.py --pred-dir results/ --gt-dir Data/Samples/

  # Self-test: compare each GT against itself (should all score 1.0):
  python Scripts/batch_eval.py --self-test

  # Score a single prediction:
  python Scripts/batch_eval.py --pred results/201414_prediction.json --gt Data/Samples/201414.json

  # Output as CSV:
  python Scripts/batch_eval.py --self-test --csv results/batch_scores.csv

Public API:
  batch_evaluate(pred_gt_pairs, config=None) -> BatchReport
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow running from repo root or Scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Scripts.eval import evaluate_extraction


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class DocScore:
    """Evaluation result for a single document."""
    doc_id: str
    score: float
    structure: float
    numbers: float
    text: float
    issue_count: int
    matched_sections: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    extra_sections: list[str] = field(default_factory=list)
    gt_section_count: int = 0
    pred_section_count: int = 0


@dataclass
class BatchReport:
    """Aggregate results across all documents."""
    doc_scores: list[DocScore]
    mean_score: float
    mean_structure: float
    mean_numbers: float
    mean_text: float
    total_issues: int
    total_missing_sections: int
    total_extra_sections: int


# ── Core batch evaluation ──────────────────────────────────────────────

def evaluate_single(
    gt: dict[str, Any],
    pred: dict[str, Any],
    doc_id: str,
    config: dict[str, Any] | None = None,
) -> DocScore:
    """Run eval on one (gt, pred) pair and return a DocScore."""
    report = evaluate_extraction(gt, pred, config)
    sa = report.get("section_alignment", {})

    return DocScore(
        doc_id=doc_id,
        score=report["score"],
        structure=report["subscores"].get("structure", 0.0),
        numbers=report["subscores"].get("numbers", 0.0),
        text=report["subscores"].get("text", 0.0),
        issue_count=len(report.get("issues", [])),
        matched_sections=sa.get("matched", []),
        missing_sections=sa.get("missing_from_prediction", []),
        extra_sections=sa.get("extra_in_prediction", []),
        gt_section_count=sa.get("gt_section_count", 0),
        pred_section_count=sa.get("pred_section_count", 0),
    )


def batch_evaluate(
    pred_gt_pairs: list[tuple[dict, dict, str]],
    config: dict[str, Any] | None = None,
) -> BatchReport:
    """Evaluate multiple (prediction, ground_truth, doc_id) triples.

    Args:
        pred_gt_pairs: List of (prediction_dict, gt_dict, doc_id) tuples
        config: Optional eval config override

    Returns:
        BatchReport with per-doc scores and aggregates
    """
    doc_scores: list[DocScore] = []

    for pred, gt, doc_id in pred_gt_pairs:
        ds = evaluate_single(gt, pred, doc_id, config)
        doc_scores.append(ds)

    n = len(doc_scores) or 1
    return BatchReport(
        doc_scores=doc_scores,
        mean_score=sum(d.score for d in doc_scores) / n,
        mean_structure=sum(d.structure for d in doc_scores) / n,
        mean_numbers=sum(d.numbers for d in doc_scores) / n,
        mean_text=sum(d.text for d in doc_scores) / n,
        total_issues=sum(d.issue_count for d in doc_scores),
        total_missing_sections=sum(len(d.missing_sections) for d in doc_scores),
        total_extra_sections=sum(len(d.extra_sections) for d in doc_scores),
    )


# ── Display ────────────────────────────────────────────────────────────

def print_report(report: BatchReport) -> None:
    """Print a human-readable score matrix to stdout."""
    print()
    print("=" * 90)
    print(f"{'doc_id':>10}  {'score':>7}  {'struct':>7}  {'numbers':>7}  "
          f"{'text':>7}  {'issues':>6}  {'sections (matched/missing/extra)'}")
    print("-" * 90)

    for d in report.doc_scores:
        sec_info = (
            f"{len(d.matched_sections)}/"
            f"{len(d.missing_sections)}/"
            f"{len(d.extra_sections)}"
        )
        if d.missing_sections:
            sec_info += f"  missing: {','.join(d.missing_sections)}"
        if d.extra_sections:
            sec_info += f"  extra: {','.join(d.extra_sections)}"

        print(
            f"{d.doc_id:>10}  {d.score:>7.4f}  {d.structure:>7.4f}  "
            f"{d.numbers:>7.4f}  {d.text:>7.4f}  {d.issue_count:>6}  {sec_info}"
        )

    print("-" * 90)
    print(
        f"{'MEAN':>10}  {report.mean_score:>7.4f}  {report.mean_structure:>7.4f}  "
        f"{report.mean_numbers:>7.4f}  {report.mean_text:>7.4f}  "
        f"{report.total_issues:>6}  "
        f"missing={report.total_missing_sections} extra={report.total_extra_sections}"
    )
    print("=" * 90)
    print()


def write_csv(report: BatchReport, path: str) -> None:
    """Write scores to CSV for tracking across optimization iterations."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "doc_id", "score", "structure", "numbers", "text",
            "issues", "matched", "missing", "extra",
        ])
        for d in report.doc_scores:
            writer.writerow([
                d.doc_id, f"{d.score:.4f}", f"{d.structure:.4f}",
                f"{d.numbers:.4f}", f"{d.text:.4f}", d.issue_count,
                ";".join(d.matched_sections),
                ";".join(d.missing_sections),
                ";".join(d.extra_sections),
            ])
        writer.writerow([
            "MEAN", f"{report.mean_score:.4f}", f"{report.mean_structure:.4f}",
            f"{report.mean_numbers:.4f}", f"{report.mean_text:.4f}",
            report.total_issues, "", "", "",
        ])
    print(f"CSV written to {path}")


# ── File loading helpers ───────────────────────────────────────────────

def load_pairs_from_dirs(
    pred_dir: str, gt_dir: str,
) -> list[tuple[dict, dict, str]]:
    """Match prediction files to GT files by doc_id (filename stem)."""
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    pairs = []

    for gt_file in sorted(gt_path.glob("*.json")):
        doc_id = gt_file.stem
        # Look for prediction with various naming conventions
        pred_candidates = [
            pred_path / f"{doc_id}.json",
            pred_path / f"{doc_id}_prediction.json",
            pred_path / f"{doc_id}_pred.json",
        ]
        pred_file = next((p for p in pred_candidates if p.exists()), None)
        if pred_file:
            with open(gt_file) as f:
                gt = json.load(f)
            with open(pred_file) as f:
                pred = json.load(f)
            pairs.append((pred, gt, doc_id))

    return pairs


def load_self_test_pairs(gt_dir: str) -> list[tuple[dict, dict, str]]:
    """Load GT files and use them as both prediction and ground truth."""
    gt_path = Path(gt_dir)
    pairs = []
    for gt_file in sorted(gt_path.glob("*.json")):
        with open(gt_file) as f:
            gt = json.load(f)
        pairs.append((gt, gt, gt_file.stem))
    return pairs


# ── CLI ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch evaluate predictions against ground truth.",
    )
    ap.add_argument("--pred-dir", help="Directory containing prediction JSONs.")
    ap.add_argument("--gt-dir", default="Data/Samples/",
                    help="Directory containing ground truth JSONs (default: Data/Samples/).")
    ap.add_argument("--pred", help="Single prediction JSON file.")
    ap.add_argument("--gt", help="Single ground truth JSON file.")
    ap.add_argument("--self-test", action="store_true",
                    help="Compare each GT against itself (all scores should be 1.0).")
    ap.add_argument("--csv", help="Write results to CSV file.")
    ap.add_argument("--config", help="Optional eval config JSON.")
    args = ap.parse_args(argv)

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    if args.self_test:
        pairs = load_self_test_pairs(args.gt_dir)
        if not pairs:
            print(f"No JSON files found in {args.gt_dir}", file=sys.stderr)
            return 1
    elif args.pred and args.gt:
        with open(args.gt) as f:
            gt = json.load(f)
        with open(args.pred) as f:
            pred = json.load(f)
        doc_id = Path(args.gt).stem
        pairs = [(pred, gt, doc_id)]
    elif args.pred_dir:
        pairs = load_pairs_from_dirs(args.pred_dir, args.gt_dir)
        if not pairs:
            print(f"No matching prediction/GT pairs found.", file=sys.stderr)
            return 1
    else:
        ap.print_help()
        return 1

    report = batch_evaluate(pairs, config)
    print_report(report)

    if args.csv:
        write_csv(report, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
