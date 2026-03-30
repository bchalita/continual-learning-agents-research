"""
PASTE THIS ENTIRE FILE INTO A STACK AI CODE NODE that runs BEFORE the Reflection LLM.

PURPOSE: Build the single user-message string for the Reflection LLM from:
  - system_prompt (current extraction prompt)
  - eval node output (full object: score, subscores, issues — no sub-entry references)
  - LLM Evaluator completion (explicit feedback from your Evaluator node)

Connect these three inputs to this Code node. Output: one string → connect to Reflection LLM's Prompt.

STACK AI WIRING:
  - Input 1: system_prompt (Variable: system_prompt)
  - Input 2: eval node's output (the whole result object — e.g. the output of your eval.py Python node)
  - Input 3: LLM Evaluator's completion (e.g. LLM Evaluator → completion)

  - Output: reflection_user_message → connect this to the Reflection LLM node's Prompt field.

PARAMETER VARIABLES (why "Parameter Variables" is empty / Output is None):
  In Stack AI, Parameter Variables are this node's INPUTS. They stay empty until you CONNECT
  upstream nodes to this Code node. You must connect three inputs and name them so the script
  can read them (the script uses globals() and expects these names):
    1. system_prompt   ← connect the input/variable that holds the current extraction prompt (e.g. in-0 or the same source as the LLM Extractor's system prompt)
    2. eval_result     ← connect the OUTPUT of your eval Python node (the whole result object: score, subscores, issues). If Stack AI names it "result", the script also checks for "result"
    3. llm_evaluator_completion ← connect the LLM Evaluator node's completion/output
  After connecting, run the flow again; the Code node will receive these as variables and
  set reflection_user_message. Then set this node's Output to the variable: reflection_user_message.

STACK AI SAFE: No import/from (uses __import__), no type annotations, no raise, no lambda.
"""

# Stack AI injects connected inputs as variables. Use the names YOU gave when connecting.
# If your eval node output is named "result", use result. If "eval_py", use eval_py. etc.
# LLM Evaluator completion: use the variable name for that connection (e.g. llm_evaluator_completion).

def build_reflection_user_message(current_prompt, eval_output, evaluator_completion=None):
    """Build the exact user message string for the Reflection LLM. No sub-entry refs: we accept full eval_output dict."""
    if eval_output is None:
        eval_output = {}
    if isinstance(eval_output, str):
        json = __import__("json")
        s = eval_output.strip()
        try:
            eval_output = json.loads(s)
        except Exception:
            try:
                ast = __import__("ast")
                eval_output = ast.literal_eval(s)
            except Exception:
                eval_output = {"score": 0, "subscores": {}, "issues": []}
    score = eval_output.get("score", 0)
    subscores = eval_output.get("subscores") or {}
    issues = eval_output.get("issues") or []
    issues = issues[:30]

    parts = [
        "Current system prompt:",
        "---",
        str(current_prompt or ""),
        "---",
        "",
        "Evaluation report:",
        f"- Overall score: {score}",
        f"- Subscores: structure={subscores.get('structure')}, numbers={subscores.get('numbers')}, text={subscores.get('text')}",
        "",
        "Top issues (path, kind, expected vs got):",
    ]
    for i, issue in enumerate(issues, 1):
        if isinstance(issue, dict):
            path = issue.get("path", "?")
            kind = issue.get("kind", "?")
            expected = issue.get("expected")
            got = issue.get("got")
            parts.append(f"  {i}. [{kind}] {path}: expected {expected!r} got {got!r}")
        else:
            parts.append(f"  {i}. {issue}")

    if evaluator_completion and str(evaluator_completion).strip():
        parts.append("")
        parts.append("Stack AI evaluator feedback:")
        parts.append(str(evaluator_completion).strip())

    parts.append("")
    parts.append("Propose a revised system prompt (output only the prompt text):")
    return "\n".join(parts)


# —— Stack AI: read injected inputs (Available Variables), map to our names —————
# The variables are NOT defined in this file; Stack AI injects them when you connect nodes.
# In "Available Variables" you see e.g. system_prompt:in_0, eval.py:python_0, LLM Evaluator:llm_2.
# The name before the colon is the label; the name after (in_0, python_0, llm_2) is often the
# variable name in the script. We read from all possible names below (no need for current_prompt = in_0
# at the top—we do the equivalent via g.get("in_0") etc. so it works whatever Stack AI calls them).

reflection_user_message = ""
try:
    g = globals()
    current_prompt = (g.get("system_prompt") or g.get("in_0") or g.get("in-0") or g.get("current_prompt") or "")
    eval_output = (g.get("eval_result") or g.get("result") or g.get("eval_py") or g.get("python_0") or g.get("python-0") or g.get("eval_output") or {})
    evaluator_completion = (g.get("llm_evaluator_completion") or g.get("evaluator_completion") or g.get("llm_evaluator") or g.get("llm_2") or g.get("llm-2") or None)

    if evaluator_completion is None and ("llm_2" in g or "llm-2" in g):
        llm_2 = g.get("llm_2") or g.get("llm-2")
        if hasattr(llm_2, "get") and callable(llm_2.get):
            evaluator_completion = llm_2.get("completion") or llm_2.get("message") or llm_2.get("content")
        else:
            evaluator_completion = llm_2
    if evaluator_completion is None:
        for key in ("LLM Evaluator_completion", "llm_evaluator_completion", "evaluator_completion", "evaluator_feedback"):
            if key in g and g[key] is not None:
                evaluator_completion = g[key]
                break

    reflection_user_message = build_reflection_user_message(current_prompt, eval_output, evaluator_completion)
except Exception as e:
    reflection_user_message = "Error building reflection input: " + str(e) + ". Check that system_prompt, eval result, and (optionally) evaluator completion are connected."

output = reflection_user_message
