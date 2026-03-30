# Stack AI — Copy-paste inputs for each field

Use this file when running your flow in Stack AI. Match the **Field** to the input in the Run/Test panel and paste the value (or copy from the file path given).

---

## Field: `in-0` (system prompt / extraction instructions)

**Paste the block below** into the system prompt field (or copy the whole file `baseline_prompt.txt`).

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

Header fields (use null for missing or unreadable values):
unit_number, customer_name, customer_address, customer_city_state_zip, customer_email, home_phone, contact_phone, bus_phone, cell_phone, advisor_id, advisor_name, color, year, make, model, submodel, vin, miles_in, miles_out, tag, open_date, deal_type, promise_time, close_date, rate, pay_type, delivery_date, stock_number, engine, time_in, date_in, time_out, date_out, customer_number, ro_number

Footer fields (all numbers; use 0 or 0.0 if absent):
labor_amount, parts_amount, gas_oil_lube, sublet_amount, misc_charges, total_charges, less_insurance, sales_tax, please_pay

Content: Include whatever you can extract. Important parts:
- "job": array of objects with at least: line_letter, text, cause (if present), op (array with code, text, labor, parts), fc, part_number, parts_cost, parts_sale, claim_type, labor_cost, labor_sale, log (array of tech/date/time/miles/text)
- "bar", "epa", "dealer_notes": strings or null
- "labor": array (date, start, finish, duration, type, tech, line_letter, charge) when present
- "acct_split": array (acct, sale, cost) when present
- "total_cost", "total_sale", "total_comp", "total_labor", "total_parts", "total_other", "total_lub", "subtotal", "total": numbers or null as appropriate

Rules:
- Use the exact key names above. Preserve nesting (header, footer, content, job, op, labor, log, etc.).
- If a value is not in the document or not readable, use null (or 0/0.0 for numeric fields where that makes sense).
- Keep types consistent: numbers as numbers, dates as strings (YYYY-MM-DD), times as strings (HH:MM).
- Extract only what is clearly present; do not invent values.
```

---

## Field: `doc-0` (document URL)

**Paste this URL** (one document; use `dl=1` for direct download):

```
https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1
```

If your flow expects an array of URLs, use: `["https://www.dropbox.com/scl/fi/1c8pbojce80arcawvkv0g/201414.pdf?rlkey=xo8tiamzvk9x3rx0fyq0si8sf&st=e5m8639y&dl=1"]`

---

## Field: `in-1` (ground truth JSON)

**Copy the entire contents** of this file (no edits):

- **Path:** `Data/Samples/201414.json`  
  (or from repo root: `Experimentation/Bernardo/` → use `Data/Samples/201414.json` relative to repo root)

Paste the full JSON as a single string into `in-1`. It is the reference extraction for document 201414; the eval node compares the LLM output to this.

---

## You do **not** paste these — the flow fills them

**`llm-2`** and **`python-0`** (or **`python-1`** if your eval node is named that) are **outputs** of other nodes. You never type them into the Run panel; once you fill **in-0**, **doc-0**, and **in-1** and click Run, the flow:

1. Runs the extractor → gets LLM output.
2. Runs the **eval Python node** → produces **python-0** (score, subscores, issues).
3. Runs the **LLM Evaluator** → produces **llm-2** (e.g. free-text feedback or the extraction snippet it was given).
4. Passes **in-0**, **python-0**, and **llm-2** into the Reflection Code node automatically.

So the “huge input” you had (with `python-0`, `in-0`, `llm-2`) is exactly what the **Reflection** script receives when the flow runs. You only supply **in-0**, **doc-0**, and **in-1**; the rest is wired by the flow.

**Reference (what those look like when the flow runs):**

- **python-0** — Eval node output: an object (or string repr) with `score`, `subscores`, `issues`, `counts_by_category`, `counts_by_kind` (from `eval_stack_ai_paste.py`).
- **llm-2** — LLM Evaluator output: usually a string, e.g. the model’s comment on the extraction or the raw extraction (e.g. `{"doc_id": null, "sections": []}` if extraction was empty).

No copy-paste needed for **llm-2** or **python-0**; they are present in the flow run and validated for the reflection script.

---

## Debug: run the Reflection Code node alone

To run **only** the Reflection Code node (e.g. to debug the script), the node needs three inputs. Paste these into the node’s **Parameter Variables** (or the Run panel inputs that map to this node).

### Parameter `in-0` (same as system prompt)

Use the **full system prompt** from the block under “Field: in-0” above (or from `baseline_prompt.txt`). Paste that entire text into the field that feeds the Reflection node as `in-0`.

### Parameter `python-0` (eval result)

Paste this **minimal eval result** (valid JSON). The Reflection script will parse it and build the message.

```json
{"score": 0.232, "subscores": {"structure": 0.36, "numbers": 0.1, "text": 0.2}, "issues": [{"path": "sections[0].footer.gas_oil_lube", "kind": "missing", "expected": 0.0, "got": null, "detail": "missing key"}, {"path": "sections[0].header.ro_number", "kind": "type", "expected": 201414, "got": "201414", "detail": "int → str"}]}
```

If Stack AI expects a **string** and your run used a Python repr before, you can paste the same JSON as a single line (above) or use this one-liner:

```
{"score": 0.232, "subscores": {"structure": 0.36, "numbers": 0.1, "text": 0.2}, "issues": [{"path": "sections[0].footer.gas_oil_lube", "kind": "missing", "expected": 0.0, "got": null, "detail": "missing key"}]}
```

### Parameter `llm-2` (LLM Evaluator output)

Paste this (e.g. empty extraction feedback):

```
{"doc_id": null, "sections": []}
```

Or any short string, e.g. `Extraction was empty.` — the Reflection script just appends it as “Stack AI evaluator feedback”.

After pasting **in-0**, **python-0**, and **llm-2**, run the Reflection Code node. The output should be the built reflection user message (current prompt + score + issues + evaluator feedback + “Propose a revised system prompt…”).

---

## Quick checklist

| Stack AI field | What to paste |
|----------------|----------------|
| **in-0**       | Full system prompt (block above, or `baseline_prompt.txt`) |
| **doc-0**      | Document URL with `dl=1` (block above) |
| **in-1**       | Entire contents of `Data/Samples/201414.json` |
| **llm-2**      | Do not paste — filled by LLM Evaluator node |
| **python-0**   | Do not paste — filled by eval Python node |

---

## Optional: `user_id`

Leave empty or set to e.g. `bchalita@mit.edu` if your flow uses it.
