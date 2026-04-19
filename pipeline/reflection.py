from __future__ import annotations

import json
import re

import anthropic

from . import config


def _load_reflection_prompt() -> str:
    return (config.PROMPTS_DIR / "reflection.txt").read_text(encoding="utf-8").strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def propose(
    extraction_prompt: str,
    eval_result: dict,
    client: anthropic.Anthropic,
) -> dict:
    """GEPA-style reflection: propose an improved extraction prompt.

    Text-only task → routes through Stack AI (Opus 4.6) for stronger reasoning.
    Falls back to Anthropic API if Stack AI is unavailable.

    Returns:
        {"extraction_prompt": str}
    Falls back to the current prompt unchanged on any parse failure.
    """
    eval_report = eval_result["eval_report"]
    qualitative_feedback = eval_result.get("qualitative_feedback", "")

    score = eval_report.get("score", 0)
    subscores = eval_report.get("subscores", {})
    top_issues = eval_report.get("issues", [])[:20]

    issues_lines = "\n".join(
        f"  {i+1}. [{issue['category']}:{issue['kind']}] {issue['path']}: "
        f"expected {issue['expected']!r} got {issue['got']!r} (penalty {issue['penalty']})"
        for i, issue in enumerate(top_issues)
    )

    user_msg = (
        f"Current extraction prompt:\n---\n{extraction_prompt}\n---\n\n"
        f"Evaluation results:\n"
        f"- Overall score: {score:.4f}\n"
        f"- Subscores: structure={subscores.get('structure')}, "
        f"numbers={subscores.get('numbers')}, text={subscores.get('text')}\n\n"
        f"Top issues (worst first):\n{issues_lines or '(none)'}\n\n"
        f"LLM evaluator diagnosis:\n{qualitative_feedback}\n\n"
        'Propose an improved prompt. Return only JSON: {"extraction_prompt": "..."}'
    )

    reflection_system = _load_reflection_prompt()

    # Route through Stack AI (text-only, Opus 4.6)
    try:
        from . import stackai
        raw = _strip_fences(stackai.call(reflection_system, user_msg))
    except Exception:
        # Fallback: direct Anthropic API
        response = client.messages.create(
            model=config.REFLECTION_MODEL,
            max_tokens=4096,
            system=reflection_system,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = _strip_fences(response.content[0].text)

    try:
        result = json.loads(raw)
        if "extraction_prompt" in result:
            return {"extraction_prompt": result["extraction_prompt"].strip()}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return {"extraction_prompt": extraction_prompt}
