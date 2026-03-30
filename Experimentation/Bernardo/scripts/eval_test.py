#!/usr/bin/env python3
"""
Stack AI Python node adapter for eval.py.

Use this script in the Stack AI flow where:
  - LLM Extractor output is available as:  llm_0
  - ground_truth input is available as:     in_1

The script parses these, calls evaluate_extraction() from Scripts/eval.py,
and returns the scored report (score, subscores, issues) for the LLM Evaluator
and Reflection nodes.

Stack AI: paste this into the Python node. Connect the LLM Extractor and
ground_truth nodes to this Python node. Stack AI will expose their outputs as:
  - llm_0   (LLM Extractor output — the extraction JSON, possibly with ```json fences)
  - in_1    (ground_truth — the reference JSON string or object)
If your flow shows different names in "Available Variables", change the
keys in _get_inputs_from_stack_ai() (e.g. to the names Stack AI displays).
For Stack AI you must either (1) paste the contents of Scripts/eval.py above
this code so evaluate_extraction is defined, or (2) ensure eval is on the path.

Local run (no Stack AI):
  python eval_test.py --gt path/to/201414.json --pred path/to/pred.json
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Optional

# ── Import evaluate_extraction ───────────────────────────────────────────
# When run from repo: add Scripts to path. When run in Stack AI: use eval if pasted or on path.
evaluate_extraction = None
try:
    from pathlib import Path
    _scripts_dir = Path(__file__).resolve().parent
    _repo_root = _scripts_dir.parent.parent  # repo root when run from Experimentation/Bernardo/scripts
    _eval_dir = _repo_root / "Scripts"
    if _eval_dir.exists():
        if str(_eval_dir) not in sys.path:
            sys.path.insert(0, str(_eval_dir))
        from eval import evaluate_extraction as _eval_fn
        evaluate_extraction = _eval_fn
except Exception:
    pass
if evaluate_extraction is None:
    try:
        import config
        if str(config.REPO_ROOT / "Scripts") not in sys.path:
            sys.path.insert(0, str(config.REPO_ROOT / "Scripts"))
        from eval import evaluate_extraction as _eval_fn
        evaluate_extraction = _eval_fn
    except Exception:
        pass


def _extract_json(text: Any) -> Dict[str, Any]:
    """Parse JSON from a string; strip markdown code fences if present. If already a dict, return as-is."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise TypeError(f"Expected str or dict, got {type(text)}")
    s = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    m = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


def _get_inputs_from_stack_ai() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Read prediction and ground truth from Stack AI-injected variables: llm_0 and in_1."""
    g = globals()
    llm_0 = g.get("llm_0")
    in_1 = g.get("in_1")
    if llm_0 is None:
        raise ValueError("llm_0 (LLM Extractor output) not set. In Stack AI, connect the LLM Extractor node to this Python node.")
    if in_1 is None:
        raise ValueError("in_1 (ground_truth) not set. In Stack AI, connect the ground_truth input node to this Python node.")
    prediction = _extract_json(llm_0)
    ground_truth = _extract_json(in_1)
    return ground_truth, prediction


def run_eval(ground_truth: Dict[str, Any], prediction: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run evaluate_extraction and return the report. Uses repo eval.py when available."""
    if evaluate_extraction is None:
        return {
            "score": 0.0,
            "subscores": {"structure": 0.0, "numbers": 0.0, "text": 0.0},
            "issues": [{"path": "", "kind": "error", "detail": "evaluate_extraction not available. Paste Scripts/eval.py into this flow or add Scripts to path."}],
            "counts_by_category": {},
            "counts_by_kind": {},
        }
    return evaluate_extraction(ground_truth, prediction, config)


# ── Main: CLI (local) vs Stack AI (llm_0, in_1) ─────────────────────────────

def _run_stack_ai_path() -> Dict[str, Any]:
    """Use variables llm_0 (Extractor output) and in_1 (ground_truth). Called when run in Stack AI."""
    try:
        _gt, _pred = _get_inputs_from_stack_ai()
        return run_eval(_gt, _pred, None)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return {
            "score": 0.0,
            "subscores": {"structure": 0.0, "numbers": 0.0, "text": 0.0},
            "issues": [{"path": "", "kind": "error", "detail": str(e)}],
            "counts_by_category": {},
            "counts_by_kind": {},
        }


# Stack AI: connected nodes are available as llm_0 and in_1. We set result for the next node.
# Local CLI: run with --gt and --pred to test without Stack AI.
if __name__ == "__main__" and "--gt" in sys.argv and "--pred" in sys.argv:
    import argparse
    ap = argparse.ArgumentParser(description="Run eval_test with files (for testing outside Stack AI).")
    ap.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    ap.add_argument("--pred", required=True, help="Path to prediction/extraction JSON.")
    ap.add_argument("--config", default=None, help="Optional eval config JSON path.")
    args = ap.parse_args()
    with open(args.gt) as f:
        ground_truth = json.load(f)
    with open(args.pred) as f:
        raw = f.read()
    prediction = _extract_json(raw)
    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    report = run_eval(ground_truth, prediction, cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    sys.exit(0 if report.get("score", 0) >= 0.25 else 1)

# Stack AI path: no CLI args, so use llm_0 and in_1 from connected nodes.
result = _run_stack_ai_path()
