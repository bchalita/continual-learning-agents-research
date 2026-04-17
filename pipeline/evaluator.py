from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anthropic

from . import config


def _load_eval() -> Any:
    """Import evaluate_extraction from Scripts/eval.py."""
    eval_dir = str(config.EVAL_SCRIPT.parent)
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from eval import evaluate_extraction  # type: ignore[import]
    return evaluate_extraction


def _llm_evaluate(
    prediction: dict,
    ground_truth: dict,
    eval_report: dict,
    client: anthropic.Anthropic,
) -> str:
    """Ask Claude for a brief qualitative diagnosis of extraction failures."""
    top_issues = eval_report.get("issues", [])[:10]
    issues_text = "\n".join(
        f"  [{i['category']}:{i['kind']}] {i['path']}: expected {i['expected']!r} got {i['got']!r} (penalty {i['penalty']})"
        for i in top_issues
    )

    score = eval_report.get("score", 0)
    subscores = eval_report.get("subscores", {})

    user_msg = (
        f"Extraction score: {score:.4f}\n"
        f"Subscores — structure: {subscores.get('structure', '?')}, "
        f"numbers: {subscores.get('numbers', '?')}, text: {subscores.get('text', '?')}\n\n"
        f"Top issues:\n{issues_text or '(none)'}\n\n"
        "In 2-3 sentences, diagnose the main failure patterns and what the extraction model is getting wrong."
    )

    response = client.messages.create(
        model=config.EXTRACTION_MODEL,
        max_tokens=512,
        system="You are an expert at diagnosing structured data extraction errors from documents. Be concise and specific.",
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text.strip()


def evaluate(
    prediction: dict,
    ground_truth: dict,
    client: anthropic.Anthropic,
) -> dict:
    """Run deterministic eval.py scorer + LLM qualitative evaluator.

    Returns:
        {
            "eval_report": {score, subscores, issues, ...},
            "qualitative_feedback": str,
        }
    """
    evaluate_extraction = _load_eval()
    eval_report = evaluate_extraction(ground_truth, prediction)
    qualitative_feedback = _llm_evaluate(prediction, ground_truth, eval_report, client)
    return {"eval_report": eval_report, "qualitative_feedback": qualitative_feedback}
