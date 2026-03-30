"""
PASTE THIS INTO A STACK AI CODE NODE that runs BEFORE the Evaluator LLM.

PURPOSE: Build the user message for the Evaluator LLM from:
  - Extractor output (the extraction JSON from the LLM)
  - Ground truth (reference JSON from in-1)

The Evaluator's Instructions (system prompt) already define the grading rubric.
Its Prompt (user message) must contain the two JSONs to compare — not the extraction
system prompt. This script outputs that single string.

STACK AI WIRING:
  - Input 1: extraction_output  <- Connect Extractor LLM output (llm-0 or LLM Executor). Script uses .completion if it's an object.
  - Input 2: ground_truth       <- Connect ground truth input (in-1)
  - Output:  evaluator_user_message -> Connect to Evaluator LLM's Prompt field.

STACK AI SAFE: No import/from at top level (uses __import__ inside function), no type annotations, no raise.
"""

def _extract_completion_only(extraction_output):
    """Get only the completion text from Extractor output (object or stringified object)."""
    if extraction_output is None:
        return ""
    # Already a dict (e.g. from Stack AI as object)
    if hasattr(extraction_output, "get") and callable(extraction_output.get):
        out = extraction_output.get("completion") or extraction_output.get("message") or extraction_output.get("content")
        return out if out is not None else str(extraction_output)
    s = str(extraction_output).strip()
    if not s:
        return ""
    # String that might be JSON of full LLM output (Stack AI sometimes stringifies the node output)
    if (s.startswith("{") and ("completion" in s or '"completion"' in s)):
        try:
            json = __import__("json")
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                out = parsed.get("completion") or parsed.get("message") or parsed.get("content")
                if out is not None:
                    return str(out).strip()
        except Exception:
            pass
    return s


def build_evaluator_user_message(extraction_output, ground_truth):
    """Build the exact user message for the Evaluator LLM: extraction JSON + ground truth JSON."""
    extraction_output = _extract_completion_only(extraction_output)
    if ground_truth is None:
        ground_truth = ""
    ground_truth = str(ground_truth)
    return (
        "Compare the following two JSON outputs and grade the extraction quality.\n\n"
        "**Extraction output (from the model):**\n" + extraction_output.strip() + "\n\n"
        "**Ground truth (expected):**\n" + ground_truth.strip() + "\n\n"
        "Output your grade as: Score: <number 0.0-1.0>\nFeedback: <1-3 sentences>"
    )


evaluator_user_message = ""
try:
    g = globals()
    extraction = g.get("extraction_output") or g.get("extraction") or g.get("llm_0") or g.get("llm-0") or g.get("llm_1") or g.get("llm-1") or ""
    ground = g.get("ground_truth") or g.get("in_1") or g.get("in-1") or g.get("ground_truth_json") or ""
    evaluator_user_message = build_evaluator_user_message(extraction, ground)
except Exception as e:
    evaluator_user_message = "Error building evaluator input: " + str(e) + ". Connect Extractor output and ground_truth (in-1) to this node."

output = evaluator_user_message
