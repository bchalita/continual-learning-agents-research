# Run the Full Flow End-to-End (Stack AI)

Use this as a copy-paste guide to run your Stack AI flow **once** and get the **reflection LLM’s output** (proposed new prompt), plus extraction and evaluation.

---

## Copy-paste commands (pre-filled)

Replace `YOUR_BEARER_TOKEN` with your Stack AI API key. Run from repo root.

**1. Single iteration with document URL (Dropbox 201414):**
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
python3 run_test_iteration.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --prompt-file baseline_prompt.txt \
  --ground-truth-file ../Samples/201414.json
```

**2. Single iteration with preloaded file (after uploading via Documents API):**
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
python3 run_test_iteration.py \
  --user-id 201414 \
  --prompt-file baseline_prompt.txt \
  --ground-truth-file ../Samples/201414.json
```

**3. Single iteration, URL as string (if flow ignores array):**
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
python3 run_test_iteration.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --doc-as-string \
  --prompt-file baseline_prompt.txt \
  --ground-truth-file ../Samples/201414.json
```

**4. Upload samples to Stack AI (need Private API key), then use command 2:**
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
export STACK_AI_PRIVATE_API_KEY="YOUR_PRIVATE_API_KEY"
python3 upload_samples_to_stack_ai.py
```

**5. Optimization loop (4 iterations, same document, prompt evolves):**
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
python3 run_optimization_loop.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --ground-truth-file ../Samples/201414.json \
  --prompt-file baseline_prompt.txt \
  --iterations 4 \
  --output-csv ../results/optimization_run.csv
```

**6. Verify document URL only (no API call):**
```bash
cd Experimentation/Bernardo/scripts
python3 run_test_iteration.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --verify-url-only
```

---

## Test the full-flow API (no fixed PDF)

**API endpoint:** After you republish the flow, Stack AI gives you a new API snippet. The scripts use `config.py` (org + flow ID) and build the same URL: `https://api.stackai.com/inference/v0/run/{org_id}/{flow_id}`. Set your **Bearer token** from that snippet as `STACK_AI_API_KEY` in the environment (do not commit the token).

**There is no fixed PDF.** Every API request sends a **new** payload: document URL, system prompt, and ground truth. So you can test with different PDFs, prompts, and ground truth on each run.

- **Document:** Pass a **public URL** in `doc-0` (e.g. Dropbox with `dl=1`, or S3 pre-signed URL from `generate_sample_urls_s3.py`). Stack AI downloads the PDF from that URL for that run.
- **Prompt:** Pass the extraction system prompt in `in-0`.
- **Ground truth:** Pass the reference JSON in `in-1`.

**From repo root**, with a public PDF URL and ground truth file:

```bash
cd Experimentation/Bernardo/scripts

export STACK_AI_API_KEY="your_bearer_token"

python3 run_test_iteration.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --prompt-file baseline_prompt.txt \
  --ground-truth-file ../Samples/201414.json
```

To use the **refined prompt** from a previous run, paste it into a file (e.g. `prompt_v2.txt`) and pass `--prompt-file prompt_v2.txt` or `--prompt "$(cat prompt_v2.txt)"`.

**Flow config in Stack AI:** Leave the document/Files node **empty** (no file uploaded in the editor). Expose `doc-0` in Export → API so the request body is used. Then each run can send a different `doc-0` URL.

**If the document is not read** (extraction returns `doc_id: null`, `sections: []`): The request is sending `doc-0` correctly; the issue is on Stack AI’s side (Files node not using the URL). See **README.md → “Diagnosis: Document is not being read from the URL”** for the checklist. As a quick test, try sending the URL as a single string instead of an array: add `--doc-as-string` to the command above (e.g. `python3 run_test_iteration.py --doc-url "..." --doc-as-string ...`).

**Optional:** If `Samples/sample_document_urls.json` exists (e.g. from Dropbox or `generate_sample_urls_s3.py`), you can omit `--doc-url` and the script will use the first URL in that file: `python3 run_test_iteration.py --ground-truth-file ../Samples/201414.json`

---

## Multi-iteration optimization loop (same file, prompt evolves)

To run **N iterations** on the same document, using the Reflection-proposed prompt as the next run’s input, and track **both** the deterministic eval score and the LLM evaluator score (plus marginal improvement) in a CSV:

**Before running: clear hardcoded inputs in Stack AI.**  
In your flow, remove any fixed values from the **Input** nodes (in-0, in-1) and from the **Files** node (doc-0). The API sends these per request; if you leave default text or a fixed URL in the flow, the API payload may be ignored and you’ll get the same run every time. Leave the input/node fields **empty** or clearly wired so the run body provides `in-0`, `doc-0`, and `in-1`.

**Complete command** (replace `YOUR_BEARER_TOKEN` with your Stack AI API key):

```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"

python3 run_optimization_loop.py \
  --doc-url "https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1" \
  --ground-truth-file ../Samples/201414.json \
  --prompt-file baseline_prompt.txt \
  --iterations 4 \
  --output-csv ../results/optimization_run.csv
```

- **Baseline:** `--prompt-file` (e.g. `baseline_prompt.txt`).
- **Each iteration:** Calls the full flow (extract → eval → evaluator → reflection), parses **eval score** (deterministic) and **LLM evaluator score**, computes **marginal improvement** vs previous iteration, saves a log and the proposed prompt to `results/prompts/iter_N_proposed.txt`, then uses that prompt as `in-0` for the next iteration.
- **Console:** Prints `Eval score`, `LLM evaluator score`, and `Marginal improvement: eval ±X | LLM evaluator ±Y` each iteration.
- **Log files:** Each `scripts/logs/api_run_<timestamp>.json` includes a `marginal_improvement` block with `eval_score` and `llm_evaluator_score` deltas when applicable.
- **Output CSV** columns: `iteration`, `eval_score`, `llm_evaluator_score`, `marginal_eval`, `marginal_llm`, `evaluator_feedback`, `proposed_prompt_path`, `log_path`. Full request/response logs go to `scripts/logs/` as usual.

If the flow returns no proposed prompt in a given iteration, the loop stops early.

---

## Where to run

- In Stack AI: open your flow (Extractor → eval.py → Evaluator → Reflection), then use the **Run** or **Test** / **Configuration** panel.
- Fill each input field with the values below. Then click **Run**.

---

## What to change (Evaluator LLM)

The **Evaluator LLM** (strict grader) must receive the **extraction JSON** and **ground truth JSON** in its **Prompt** (user message), not the extraction system prompt. If its Prompt is wired to `in-0` / system_prompt, it will say things like “I don’t see a document” and will not grade.

**Fix:**

1. **Disconnect** whatever is currently connected to the Evaluator LLM’s **Prompt** field (e.g. `in-0` or the extraction prompt).
2. **Add a Code node** before the Evaluator LLM (e.g. “Build Evaluator prompt”).
3. **Paste** the script from `Experimentation/Bernardo/scripts/build_evaluator_prompt_stack_ai.py` into that Code node.
4. **Connect two inputs** to this Code node:
   - **extraction_output** ← **Extractor LLM** output (e.g. LLM Executor / `llm-0`). The script will use the `completion` field if it’s an object.
   - **ground_truth** ← ground truth input (**in-1**).
5. **Set the Code node output** to `evaluator_user_message` (or `output`).
6. **Connect** that Code node’s output → **Evaluator LLM’s Prompt** field.

**Evaluator LLM Instructions** (system prompt): leave as is — the strict grader rubric (Score 0.0–1.0, Feedback, key fields) is correct.

Result: the Evaluator receives a single user message containing “Extraction output: …” and “Ground truth: …” and returns “Score: … Feedback: …”.

---

## 1. System prompt (`in-0` / system_prompt)

**Option A — Copy from file (recommended)**  
Open this file and copy its **entire** contents into the system prompt input:

```
Experimentation/Bernardo/scripts/baseline_prompt.txt
```

**Option B — Short excerpt (for quick tests)**  
Paste this into the system prompt field:

```
You are an expert at extracting structured data from automotive work order and repair order documents (e.g. BMW dealer service documents).
Your task: Read the document provided and extract all relevant information into a single JSON object that matches the following structure. Return only valid JSON—no markdown, no code fences, no explanation before or after.

Top-level structure:
- "doc_id": string (document or repair order identifier if visible; otherwise null)
- "sections": array of section objects (see below)

Each section object must have:
- "section_id": string (e.g. "ASI-201414", "BWO-201414"; combine prefix and doc_id if available)
- "prefix": string (section type: ASI, BWO, CSI, JSI, WSI, ISI, or as shown in the document)
- "page_count": number or null
- "header": object (see header fields below)
- "footer": object (see footer fields below; use {} if empty)
- "content": object (see content fields below)

Rules:
- Use the exact key names above. Preserve nesting (header, footer, content, job, op, labor, log, etc.).
- If a value is not in the document or not readable, use null (or 0/0.0 for numeric fields where that makes sense).
- Return only valid JSON, no markdown or explanation.
```

For a full run (better extraction quality), use **Option A** and paste the whole `baseline_prompt.txt`.

---

## Reflection LLM (what it receives as input)

The **Reflection LLM** node should get:

| Input | Source | Description |
|-------|--------|-------------|
| **Current system prompt** | Same as the extraction system prompt you used for this run (e.g. from `in-0` or `baseline_prompt.txt`). |
| **Evaluation report** | Output of the **eval Python node** (eval.py / eval_stack_ai_paste.py): overall score, subscores (structure, numbers, text), and top issues (path, kind, expected vs got). This is the deterministic comparison of extraction JSON vs ground truth. |
| **Stack AI evaluator feedback** | Output of your **LLM Evaluator** node: free-text feedback on the extraction. You have this node in the flow — it must be included in the Reflection LLM’s user message. |

**Do not reference eval sub-entries in the Prompt box.** Stack AI does not support things like `eval.result.score` or nested refs. Use a Code node instead (below).

**What goes where (Instructions vs Prompt)**

- **Instructions** (sometimes labeled as system prompt):  Put **only** the instruction text below. Do **not** put node names or “take input from X” in Instructions. The prompt describes *how* the model should behave and what format it will receive; it does not define *where* the data comes from.
- **Prompt** (the other box): Connect the **output** of the Code node described below. Do not paste template text with `{{eval.result.score}}`-style refs — they error. Only one variable: the Code node’s output string.

---

## Reflection LLM — Code node (required; includes LLM Evaluator)

Add one **Code** node between your eval node + LLM Evaluator and the Reflection LLM. It builds the full user message (current prompt + eval report + **LLM Evaluator feedback**) so you never reference eval sub-entries.

1. **Create a Code node** in the flow (e.g. “Build reflection prompt”).
2. **Paste the script** from `Experimentation/Bernardo/scripts/build_reflection_prompt_stack_ai.py` into that Code node.
3. **Connect three inputs** to this Code node (name them so the script can read them):
   - **system_prompt** ← Variable `system_prompt` (same extraction prompt you use for the run).
   - **eval_result** (or **result**) ← Output of your **eval Python node** (the whole object: score, subscores, issues). Do not try to wire “eval.result.score” — wire the node’s single output.
   - **llm_evaluator_completion** ← **LLM Evaluator** node’s completion/output. Your flow has an LLM Evaluator; wire it here so its feedback is included in the Reflection prompt.
4. **Code node output:** the script sets `reflection_user_message`. Expose that as the Code node’s output (e.g. name the output `reflection_user_message` or `output`).
5. **Connect** that Code node output. If the Run Python Code modal shows **Parameter Variables** empty and **Output: None**, the Code node has no inputs yet: connect (1) system prompt source to `system_prompt`, (2) eval Python node output to `eval_result` or `result`, (3) LLM Evaluator completion to `llm_evaluator_completion`. Set the Code node Output to `reflection_user_message`. Then connect that output to the Reflection LLM's Prompt field (→ **Reflection LLM** node’s **Prompt** field.

Result: Reflection LLM receives Instructions = the long system prompt; Prompt = this single string (current prompt + eval report + LLM Evaluator feedback + “Propose a revised…”).

---

## Reflection LLM — Instructions vs Prompt

- **Instructions** box: Paste the **full** reflection system prompt (the block below). This is the model’s role and rules. Do not put it in Prompt.
- **Prompt** box: This is the **user message** (current system prompt + evaluation report + “Propose a revised…”). Connect a Code node that builds that payload, or use variables that reference the extraction prompt input and the eval node’s `result`.

---

## Reflection LLM — system prompt (paste into **Instructions**)

**Copy everything between the lines below** and paste it into the **Reflection LLM node’s system prompt / prompt box** in Stack AI. Put this in **Instructions** only; **Prompt** is for the user message from the flow.

```
You are an expert at improving extraction prompts for document understanding systems (e.g. work order / repair order JSON extraction).

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

Your task: Propose a revised system prompt that addresses the main failures in the evaluation report (missing keys, wrong types, wrong values, structural mismatches) while keeping the prompt clear and the output schema consistent with the original. Preserve the intended extraction schema (e.g. doc_id, sections with header/footer/content, field names); fix instructions so the model produces correct types and values. If the weakest subscore is "numbers", emphasize numeric fields and units (e.g. cents vs dollars); if "structure", emphasize required keys and nesting; if "text", emphasize string formatting and null vs empty.
```

---

## 2. Document URL (`doc-0` / files_node)

Paste this URL into the document / file input (must be the field that accepts URLs, e.g. `doc-0`). Use `dl=1` for direct download.

```
https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1
```

If your flow expects a **list** of URLs, use: `["https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1"]` (or the same single URL in your platform’s list format).

---

## 3. Ground truth (`in-1` / ground_truth)

Copy the **entire** contents of one of these JSON files into the ground truth input:

- `Experimentation/Bernardo/Samples/201414.json`  
- or `Data/Samples/201414.json`

Open the file in your editor, select all, copy, then paste into the ground_truth field. It is a large JSON (~16k characters); the flow needs the full object for evaluation and reflection.

---

## 4. User message (if your flow has a “User Message” input)

You can leave it empty or use a single instruction, for example:

```
Extract all sections and fields from this document. Return only valid JSON.
```

---

## What you should see in the output

After you click **Run**, expect something like the following (exact keys depend on how your flow’s Output node is set up):

| Output | Description |
|--------|-------------|
| **Extraction result** | JSON from the LLM Extractor (`doc_id`, `sections`, …). If the document didn’t reach the model, you may see `{"doc_id": null, "sections": []}`. |
| **eval.py result** | If the Python node output is exposed: score, subscores, and/or a list of issues (e.g. missing/wrong fields). |
| **Evaluator feedback** | If you have an LLM Evaluator node and its output is exposed: text feedback on the extraction (e.g. “RO number wrong; VIN missing”). |
| **Reflection LLM output** | **This is the main one.** Plain text: the **proposed new system prompt** that the reflection model suggests to improve extraction. Copy this and use it as `in-0` / system_prompt in the **next** run to close the loop. |
| **run_id** | Stack AI run identifier (useful for debugging or support). |
| **citations / metadata** | Optional; depends on your flow. |

So: you should get the **reflection LLM’s output** as a text block (the new prompt). Anything else (extraction, eval, evaluator text) is extra context; the loop is: use reflection output → paste as new system prompt → run again.

---

## If extraction is empty

If the extraction is `doc_id: null, sections: []`:

- The Evaluator and Reflection may still run (e.g. on “no document seen”) and you’ll still get a reflection output, but it won’t be grounded in real extraction errors.
- Fix the flow first: ensure the **Files** node is wired so the LLM receives the parsed document, **Enable parsing** and **Text in images (OCR)** are ON, and the document URL is the one above with `dl=1`. See `README.md` → “Troubleshooting: Empty extraction.”

---

## Stack AI Python node restrictions (errors to avoid)

Stack AI’s Code/Python nodes run in a restricted environment. These AST/expressions can trigger **“Unsafe expression … detected”** (or similar). The scripts `eval_stack_ai_paste.py` and `build_reflection_prompt_stack_ai.py` are written to avoid them.

| Blocked / risky | Meaning | What we do instead |
|-----------------|--------|---------------------|
| **Import** | `import x` | Use `x = __import__("x")` so there are no `Import` AST nodes. |
| **ImportFrom** | `from x import y` | Same: load module with `__import__("x")`, then use `x.y`. |
| **AnnAssign** | Annotated assignment, e.g. `config: Dict[str, Any] = {}` | Use plain assignment: `config = {}`. No type annotation on the variable. |
| **Function/param/return annotations** | `def f(x: int) -> str:` | Use `def f(x):` and no `-> ...` so there are no annotation nodes. |
| **Lambda** | `key=lambda x: x["a"]` | Use a named function and pass it: `def _key(it): return it["a"]` then `key=_key`. |
| **@dataclass / class with annotations** | Class attributes with types (e.g. `category: str`) are AnnAssign in the class body. | Use a plain function that returns a dict instead of a dataclass. |
| **Raise** | `raise ValueError("...")` | Return error info instead: e.g. return `(None, "error message")` and have the caller check and build an error result dict. |

If you see a **new** “Unsafe expression ‘X’ detected” message, remove or rewrite the construct that corresponds to that AST node name (e.g. avoid decorators if “Decorator” is blocked, avoid `exec`/`eval` if “Call” to those is blocked).

---

## Final output: include evaluations when calling the API

When you pull the flow result from the API, you typically get the **Reflection LLM output** (the proposed new system prompt). For debugging, logging, and the next iteration you should **also** expose and save the **evaluation result** from this run.

**Recommendation:** In Stack AI, configure your flow’s **Output** (or the response shape) so the final payload includes at least:

| Field | Source | Use |
|-------|--------|-----|
| **proposed_prompt** (or **reflection_output**) | Reflection LLM → completion | New system prompt for the next run. |
| **eval_result** | Eval Python node → output | Score, subscores, issues, counts. Save for metrics and debugging. |
| (Optional) **extraction** | LLM Extractor output | Raw extraction JSON from this run. |
| (Optional) **evaluator_feedback** | LLM Evaluator → completion | Free-text feedback from the evaluator. |

**Making the API point to them clearly:** In the Output node, define **one output variable per source** and give each a **clear name**. Map as follows so the API response has explicit keys:

| Output variable name | Connect from |
|----------------------|--------------|
| **extraction** | Extractor LLM (first LLM) → completion / output |
| **eval_result** | Eval Python node → output |
| **evaluator_feedback** | LLM Evaluator → completion / output |
| **proposed_prompt** | Reflection LLM → completion / output |

Then the API response will look like: `{ "extraction": "...", "eval_result": { "score": 0.37, ... }, "evaluator_feedback": "...", "proposed_prompt": "..." }`. If you only connect 3 of the 4, only those 3 keys will appear. For the full loop (extraction + both evaluations + proposed prompt), connect all four and use these names.

That way each API response gives you: (1) the prompt to use next, (2) the eval metrics and issue list to log or plot, and (3) optionally the extraction and evaluator text. Wire the eval node’s output (and optionally the extractor and evaluator outputs) into the same Output node or into the flow’s final response definition so they appear in the JSON you receive.

---

## Quick checklist

- [ ] System prompt: full `baseline_prompt.txt` (or the excerpt above).
- [ ] Document: Dropbox URL with `dl=1` in the document/files input.
- [ ] Ground truth: full contents of `201414.json` in the ground_truth input.
- [ ] Run the flow and locate the **Reflection LLM output** in the response.
- [ ] (Optional) Copy that reflection output and paste it as the **system prompt** for the next run to test the loop.
