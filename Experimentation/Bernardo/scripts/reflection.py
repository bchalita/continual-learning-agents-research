"""
GEPA-style reflection: given current prompt, eval report, and optional Stack AI
evaluator feedback, ask an LLM to propose an improved system prompt for the
extraction agent.

Reflection can run:
  - Locally via OpenAI (OPENAI_API_KEY).
  - In Stack AI: deploy a "GEPA Reflection" workflow and call it via API (uses
    your Stack AI free tokens; no OpenAI key needed for this step).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Optional: use OpenAI for reflection. If not installed or no key, propose_prompt returns a stub message.
try:
    import openai
except ImportError:
    openai = None


SYSTEM_PROMPT_REFLECTION = """You are an expert at improving extraction prompts for document understanding systems (e.g. work order / repair order JSON extraction).

Current logic: The extraction agent runs with a system prompt and a document, and outputs structured JSON. That JSON is compared to ground truth by a deterministic evaluator (eval.py); the evaluator produces a score, subscores, and a list of issues. You receive (1) the current system prompt, (2) that evaluation report, and optionally (3) free-text feedback from an LLM grader. Your job is to propose a revised system prompt that fixes the main failures.

---
INPUT SCHEMA (exactly what you will receive in the user message)
---
The payload is plain text with the following sections in order:

1. **Current system prompt**
   - Header line: "Current system prompt:"
   - Then a line "---"
   - Then the full text of the extraction agent's current system prompt
   - Then a line "---"

2. **Evaluation report** (from the deterministic eval: ground truth vs extraction JSON)
   - Header: "Evaluation report:"
   - Line: "- Overall score: <number between 0 and 1>"
   - Line: "- Subscores: structure=<...>, numbers=<...>, text=<...>"
   - Header: "Top issues (path, kind, expected vs got):"
   - A numbered list of issues, each line: "  N. [<kind>] <path>: expected <value> got <value>"
   - <kind> is one of: missing, extra, type, value
   - <path> is a JSON path into the extraction (e.g. sections[0].header.unit_number, sections[1].content.job)
   - Issues are ordered by penalty (worst first). Use them to fix: wrong types (e.g. string vs int), wrong values, missing keys, extra keys, or structural mismatches. Subscores indicate which category (structure, numbers, text) is weakest.

3. **Stack AI evaluator feedback** (optional)
   - Header: "Stack AI evaluator feedback:"
   - Then free-text feedback from an LLM grader node, if present. Use it to complement the evaluation report.

4. **Instruction line**
   - "Propose a revised system prompt (output only the prompt text):"

---
OUTPUT FORMAT (you must follow this exactly)
---
- Output ONLY the revised system prompt text. Nothing else.
- Do NOT add any prefix (e.g. "Here is the revised prompt:", "Revised prompt below:").
- Do NOT wrap the prompt in markdown code blocks (no ```).
- Do NOT add commentary, explanations, or meta-notes before or after the prompt.
- The first character of your response must be the first character of the new system prompt (e.g. "You" or the first word of the instructions).
- The last character of your response must be the last character of the new system prompt.

Your task: Propose a revised system prompt that addresses the main failures in the evaluation report (missing keys, wrong types, wrong values, structural mismatches) while keeping the prompt clear and the output schema consistent with the original. Preserve the intended extraction schema (e.g. doc_id, sections with header/footer/content, field names); fix instructions so the model produces correct types and values. If the weakest subscore is "numbers", emphasize numeric fields and units (e.g. cents vs dollars); if "structure", emphasize required keys and nesting; if "text", emphasize string formatting and null vs empty."""


def _build_user_message(
    current_prompt: str,
    eval_report: Dict[str, Any],
    stack_ai_feedback: Optional[str] = None,
    max_issues: int = 30,
) -> str:
    """Build the user message for the reflection LLM."""
    score = eval_report.get("score", 0)
    subscores = eval_report.get("subscores", {})
    issues = eval_report.get("issues", [])[:max_issues]

    parts = [
        "Current system prompt:",
        "---",
        current_prompt,
        "---",
        "",
        "Evaluation report:",
        f"- Overall score: {score}",
        f"- Subscores: structure={subscores.get('structure')}, numbers={subscores.get('numbers')}, text={subscores.get('text')}",
        "",
        "Top issues (path, kind, expected vs got):",
    ]
    for i, issue in enumerate(issues, 1):
        path = issue.get("path", "?")
        kind = issue.get("kind", "?")
        expected = issue.get("expected")
        got = issue.get("got")
        parts.append(f"  {i}. [{kind}] {path}: expected {expected!r} got {got!r}")

    if stack_ai_feedback:
        parts.append("")
        parts.append("Stack AI evaluator feedback:")
        parts.append(stack_ai_feedback)

    parts.append("")
    parts.append("Propose a revised system prompt (output only the prompt text):")
    return "\n".join(parts)


def propose_prompt(
    current_prompt: str,
    eval_report: Dict[str, Any],
    stack_ai_feedback: Optional[str] = None,
    model: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
) -> str:
    """
    Call the reflection LLM to propose an improved system prompt.

    If OpenAI is not available or api_key is missing, returns a placeholder
    string so the pipeline can run without an API.
    """
    user_msg = _build_user_message(current_prompt, eval_report, stack_ai_feedback)

    if openai is None:
        return "[Reflection LLM not available: install openai and set API key. Stub: improve types (int vs string) and numeric units (e.g. cents vs dollars) in the schema.]"

    api_key = os.environ.get(api_key_env)
    if not api_key:
        return "[Set OPENAI_API_KEY to run reflection. Stub: align numeric types and units with ground truth; add missing keys for job-level totals and wsi_labor.]"

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_REFLECTION},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def build_reflection_user_message(
    current_prompt: str,
    eval_report: Dict[str, Any],
    stack_ai_feedback: Optional[str] = None,
    max_issues: int = 30,
) -> str:
    """
    Build the single user message sent to the reflection LLM.
    Use this when calling Stack AI: send the result as the workflow's text input.
    """
    return _build_user_message(
        current_prompt, eval_report, stack_ai_feedback, max_issues
    )


def propose_prompt_via_stack_ai(
    current_prompt: str,
    eval_report: Dict[str, Any],
    stack_ai_feedback: Optional[str] = None,
    org_id: str = "",
    flow_id: str = "",
    base_url: str = "https://api.stack-ai.com/inference/v0/run",
    api_key_env: str = "STACK_AI_API_KEY",
    input_key: str = "in-0",
) -> str:
    """
    Run GEPA-style reflection inside Stack AI (uses your free tokens).

    Requires a deployed Stack AI workflow "GEPA Reflection" with:
      - One text input (e.g. in-0): the reflection payload (current prompt + eval report + feedback).
      - System prompt: same as SYSTEM_PROMPT_REFLECTION in this file.
      - One LLM node → output = proposed prompt.

    Set org_id and flow_id in config, or pass them here. API key from env (api_key_env).
    """
    api_key = os.environ.get(api_key_env)
    if not api_key or not org_id or not flow_id:
        return (
            "[Stack AI reflection: set STACK_AI_API_KEY, and config STACK_AI_REFLECTION_ORG_ID + "
            "STACK_AI_REFLECTION_FLOW_ID (or pass org_id/flow_id). "
            "Create a 'GEPA Reflection' workflow in Stack AI with one text input and the reflection system prompt.]"
        )

    user_msg = _build_user_message(
        current_prompt, eval_report, stack_ai_feedback
    )
    url = f"{base_url.rstrip('/')}/{org_id}/{flow_id}"
    payload = {"user_id": "", input_key: user_msg}
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except (HTTPError, URLError) as e:
        return f"[Stack AI reflection request failed: {e}]"

    # Stack AI returns outputs as out-0, out-1, ... or under "outputs" / "output"
    out = (
        data.get("out-0")
        or data.get("output")
        or (data.get("outputs") or [{}])[0]
    )
    if isinstance(out, dict):
        out = out.get("output", out.get("text", str(out)))
    return (out or "").strip()
