# Stack AI + Eval + GEPA Reflection Pipeline

This folder holds the code that supports the **Stack AI workflow** and prompt optimization loop.

## Flow (high level)

1. **Stack AI** runs the extraction workflow (document + system prompt → LLM → extraction_result).
2. You export **extracted results** and (optionally) **evaluator feedback** from Stack AI (Batch Run or Evaluator view → CSV/download).
3. **This script** loads that export, runs **eval.py** (programmatic comparison vs ground truth) on each row, and aggregates scores + issues.
4. A **GEPA-style reflection** step (LLM) takes: current prompt, eval report, and feedback text → proposes a **new system prompt**.
5. You feed the new prompt back into Stack AI (next Batch run or manual run) and repeat.

So: **Stack AI** does extraction (and optional LLM grading); **our code** does deterministic eval + reflection and produces the next prompt.

## PDFs in Stack AI: Knowledge Base vs Files

- **Files node:** One document per run; you upload or reference a PDF for each execution. Best for our use case: one work order in → one extraction out, with the **system prompt** as the only thing we optimize.
- **Knowledge Base:** RAG over many uploaded documents; you query it (e.g. “find similar work orders”). Use a KB if you want to retrieve reference docs or similar cases; for the current extraction pipeline we use the **Files** node only.

### Files node configuration (for API / URL input)

So the document comes from the API request (e.g. `doc-0` with a public URL):

1. **Expose as input: ON** — So the node receives the file/URL from the request. If OFF, the flow uses only the file you uploaded in the editor and ignores `doc-0`.
2. **No default/static file** — When using the API with URLs, do not leave a file uploaded in the node; leave it empty so the request body (`doc-0`) is used.
3. **Enable parsing: ON** — Required for PDFs so Stack AI extracts text and sends it to the LLM.
4. **Text in images (OCR): ON** (if PDFs are image-based) — For scanned work orders so text in images is extracted. OCR increases cost and latency.
5. **Chunking** — Use defaults unless you have very long documents.
6. **Export → API** — Confirm the Files node is exposed as the input key you send (e.g. `doc-0`).

### Diagnosis: “Document is not being read from the URL”

If the **request** (e.g. in `scripts/logs/api_run_*.json`) clearly sends `doc-0` with your PDF URL (e.g. Dropbox with `dl=1`) and the **response** is still `doc_id: null`, `sections: []`, then:

- **The extraction LLM is not receiving the document.**  
- In practice this almost always means **the document input is not being read from the URL** on Stack AI’s side (wrong input key, format, or URL not fetched/parsed).

**What to check in Stack AI:**

1. **Export → API**  
   In your flow, open **Export → API** and note the **exact input name** for the Files/Document node. It might be `doc-0`, `file-0`, `url-0`, or something else. Our scripts use `doc-0` by default; if the flow expects another key, set it via config or `--doc-input-key`.

2. **Files node “Expose as input”**  
   The Files node must be **exposed as an input** so the run request can provide the file/URL. If it isn’t, the flow will only use a file you uploaded in the editor and will ignore the URL in the request.

3. **In-app Run and URLs**  
   Many Stack AI flows only let you **upload a file** in the in-app Run panel, not paste a URL. If your flow has no URL field for the document input, you can’t test “same URL in-app” directly. In that case rely on (2) and (4) below, and if you have upload capability, use (5) to confirm the flow works when the document comes from the bucket.

4. **Confirm document format expected by the flow (API only)**  
   When calling via API, some flows expect `doc-0` as a **single URL string**; others expect an **array of URLs**. Our default is array. To send a single string, use the pre-filled command in **RUN_END_TO_END.md → Copy-paste commands (3)**.

5. **Isolate: use upload + `user_id` instead of URL**  
   If you have the **Private** API key, upload the same PDF via the Documents API (`upload_samples_to_stack_ai.py`), then run with `--user-id 201414` (no `--doc-url`). If extraction **works** with `user_id`, the rest of the flow (LLM wiring, parsing) is fine and the issue is specifically **URL-based document input** (key, format, or fetch/parse). You can also run in-app after uploading: in the Run panel, the document may appear from the bucket when you set the same `user_id`, so you can confirm extraction in-app without pasting a URL.

### Troubleshooting: Empty extraction (`doc_id: null`, `sections: []`)

If the API returns successfully but the extraction is empty, the **LLM is not receiving the document**. Fix this in the Stack AI flow:

1. **Wire the Files node into the LLM**  
   The **output** of the Files (document) node must be connected to the **LLM node** as the document/user input. If only the system prompt (`in-0`) is connected, the model has no document to read.

2. **Files node settings**  
   - **Enable parsing: ON** (required for PDFs).  
   - **Text in images (OCR): ON** if the PDF is scanned/image-based (e.g. work orders).

3. **No static file in the node**  
   When using the API with `doc-0`, leave the Files node **empty** in the editor (no file uploaded). Otherwise the flow may ignore the URL from the request.

4. **Test inside Stack AI**  
   If your flow’s Run panel lets you paste a URL for the document, use the same URL (with `dl=1` for Dropbox) and run. If the panel only supports **file upload**, upload the same PDF and run with the same `user_id` you use for the API (or run via API with `--user-id` after uploading). If extraction is still empty, the issue is flow wiring or Files/LLM settings; if it works with upload, the issue is URL-based input (key/format or fetch).

5. **URL**  
   Use a **direct-download** link (e.g. Dropbox with `dl=1`). The script normalizes Dropbox URLs automatically; the URL printed at Step 3 is what is sent.  
   **Check that the URL actually returns a PDF:** run with your `--doc-url`; the script will try to fetch the URL and report `URL check: OK` or `FAIL`. You can also run **only** the URL check (no API call) with:  
   `python3 run_test_iteration.py --doc-url "https://...?dl=1" --verify-url-only`  
   If you see `URL check: FAIL` or `WARN`, the link may be expired, require login, or return HTML instead of the file — Stack AI will then have no document to parse.

6. **`file_urls` has 1 item but `documents.docs` and `raw_files.files` are empty**  
   The flow is receiving the URL but not producing parsed content. Check: (a) **Enable parsing** and **Text in images (OCR)** on the Files node so the URL is fetched and turned into text/chunks. (b) If the LLM is connected to the “documents” (or similar) output, that output must be populated by the Files node after it downloads from the URL; if download or parsing fails (e.g. Dropbox redirect, timeout), docs stay empty. Try a different host (e.g. S3 pre-signed URL or a direct PDF link) to see if the issue is URL-specific. (c) In Stack AI docs/support, look for “Files node” + “URL” or “public URL” to confirm URL-based document input is supported and how to get `documents.docs` filled.

**If wiring and settings look correct but extraction is still empty:** Run the test script with `--debug` to see the exact payload and full API response:  
Use the pre-filled command in **RUN_END_TO_END.md → Copy-paste commands (1)** and add `--debug` at the end.
Compare the **input keys** in the payload with the names shown in Stack AI’s Export → API (e.g. `in-0`, `doc-0`). Then run the **same** URL and prompt in Stack AI’s in-app **Run** (Configuration tab). If it’s empty in-app too, the problem is likely URL fetch/parsing (e.g. Dropbox or host); if it works in-app, the issue is the API payload keys or format.

## Where are my PDFs? Do I have “public URLs” in Stack AI?

- **Your PDFs in `Samples/` (e.g. 201414.pdf)** are only on your machine. Stack AI does not see them until you either upload them or give Stack AI a URL that points to them.
- **Stack AI does not give you “public URLs” for your files.** When you upload via the Documents API, files go into a **user bucket** (tied to org + flow + user_id). You do **not** get back a URL to paste into `doc-0`. Instead, you run the flow with that **same user_id** and the flow uses the file from the bucket. So: **upload → then use `--user-id 201414`** (no URL needed).
- **If you want to use `doc-0` with a list of URLs**, you must **create** those URLs yourself (e.g. upload to S3 and use `generate_sample_urls_s3.py`, or host the PDFs somewhere else that returns a public URL). Those URLs are not “in” Stack AI — they point to your own hosting.

**Summary:** Either (1) **upload to Stack AI** and use `user_id` when running (no public URLs), or (2) **host PDFs elsewhere**, get public URLs, and pass them in `doc-0`.

## Extraction flow API: document = public URL or user_id

When you call the **extraction flow** via API, you provide the document either as a **public URL** in `doc-0` (Stack AI downloads it) or by **user_id** (after uploading via Documents API; the flow uses the file from your bucket). You cannot send raw PDF bytes in the run request.

- **In the run request:** The body must include the document input with a **public URL** string. The exact key depends on your flow (e.g. `doc-0`, `url-0`, or whatever the Files/Document node is mapped to in the Export → API view). Example:
  ```json
  {
    "user_id": "run-1",
    "doc-0": "https://your-bucket.s3.amazonaws.com/samples/201414.pdf",
    "in-0": "Your system prompt text here..."
  }
  ```
- **If you don’t have public URLs:** You have to host the PDFs somewhere that returns a direct, publicly readable URL. Options:
  1. **Cloud storage (S3, GCS, R2, etc.):** Upload the PDF, set the object to public read (or use a signed URL if your flow supports it), and use the object URL as the document input key.
  2. **Stack AI’s guide:** In the Export / API section of your extraction flow, Stack AI may show a “guide” for document nodes (e.g. upload via their Documents API first, then reference the file in the run). Follow that in-app guide if you prefer not to host files elsewhere.
  3. **Quick testing:** Use a temporary public URL (e.g. put the PDF in a repo and use GitHub raw, or run a local HTTP server and expose it with ngrok) and pass that URL in the run.

Once you have a public URL for each PDF, use it as the value for the document input in every extraction run. For batch runs, your CSV (or script) should have one column with the document URL per row (and other columns for system prompt, ground truth, etc.).

### Commands: upload files and get references

**Option A — Upload to Stack AI (no URLs; use `user_id`):**

Requires the **Private** API key (Stack AI Settings → API Keys). If your org (e.g. MIT Sloan) manages Stack AI and you don’t have access to API keys, use **Option B** below instead.

```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
export STACK_AI_PRIVATE_API_KEY="YOUR_PRIVATE_API_KEY"
python3 upload_samples_to_stack_ai.py
```

Then run a single iteration with the preloaded file (see **RUN_END_TO_END.md → Copy-paste commands (2)**):
```bash
cd Experimentation/Bernardo/scripts
export STACK_AI_API_KEY="YOUR_BEARER_TOKEN"
python3 run_test_iteration.py \
  --user-id 201414 \
  --prompt-file baseline_prompt.txt \
  --ground-truth-file ../Samples/201414.json
```

- **References file:** `Samples/sample_document_user_ids.json` (doc_id → user_id).
- **How to reference:** `--user-id 201414`, `--user-id 678856`, etc. (doc_id = user_id).
- **See all refs:** `python3 list_sample_refs.py`

**Option B — Public URLs (no private key needed; good for org-managed accounts):**

If you can’t get the private key (e.g. org admin only), host the PDFs somewhere that gives **public URLs** and pass those to the extraction flow. You only need the **public** key (Run API) for inference.

- **S3:** If you have AWS access, use `generate_sample_urls_s3.py` (see below); then use `--doc-url <url>` from `Samples/sample_document_urls.json`.
- **Other:** Any host that returns a direct PDF URL (e.g. Dropbox/Drive share link, internal server, ngrok in front of a local folder). Use that URL as `--doc-url` when calling the extraction API.

**Option C — Upload to S3 and get public URLs (scripted):**

```bash
cd Experimentation/Bernardo/scripts
export S3_SAMPLES_BUCKET="YOUR_S3_BUCKET_NAME"
pip install boto3
python3 generate_sample_urls_s3.py
```

- **References file:** `Samples/sample_document_urls.json` (doc_id → url).
- **How to reference:** `--doc-url "<url>"` (copy URL from the JSON for the doc_id you want).
- **See all refs:** `python list_sample_refs.py`

Both reference files live in `Samples/` and are easy to reference in scripts; run `list_sample_refs.py` to print the mapping anytime.

### Making sample PDFs available (Stack AI’s guide)

You can use either approach so the extraction flow has access to the PDFs in `Samples/`:

1. **Upload to Stack AI (no public URL needed)**  
   Upload all sample PDFs into Stack AI’s Documents API (user bucket). When you run the extraction flow, pass the same `user_id` and the flow will use the uploaded file.
   - Set `STACK_AI_EXTRACTION_FLOW_ID` in `config.py` (your extraction flow ID from Export → API).
   - Set `STACK_AI_PRIVATE_API_KEY` in the environment (Stack AI Settings → API Keys → **Private** key; the public key returns 401 for uploads).
   - Run: `python3 upload_samples_to_stack_ai.py`
   - This creates `Samples/sample_document_user_ids.json` with `doc_id` → `user_id`. When calling the extraction flow, use `--user-id 201414` (for `201414.pdf`) instead of `--doc-url`.

2. **S3 pre-signed URLs (Stack AI’s documented option)**  
   Upload PDFs to an S3 bucket and generate pre-signed URLs (valid up to 12 hours). Pass the URL in the run body as the document input.
   - Create an S3 bucket and set `S3_SAMPLES_BUCKET` (and AWS credentials).
   - Install boto3: `pip install boto3`
   - Run: `python generate_sample_urls_s3.py`
   - This creates `Samples/sample_document_urls.json` with `doc_id` → `url`. Use `--doc-url <url>` when calling the extraction flow (or put the URL in your batch CSV).

## Layout

- `run_pipeline.py` – Entry point: load Stack AI export, run eval, run reflection, output new prompt.
- `run_extraction_api.py` – Call the **extraction flow** via API: pass `--doc-url <public URL>` or `--user-id <id>` (after uploading with `upload_samples_to_stack_ai.py`) and `--prompt`.
- `run_test_iteration.py` – **One-shot test**: baseline prompt + one document (201414) → Stack AI → printed result. Use to confirm the workflow works. Requires `--user-id 201414` (after upload) or `--doc-url <url>`. Optional: `--save-output` to write extraction to `Samples/201414_extracted_test.json` for eval.
- `baseline_prompt.txt` – Default system prompt for extraction (used by `run_test_iteration.py`).
- `upload_samples_to_stack_ai.py` – Upload all sample PDFs to Stack AI Documents API; then run the flow with `user_id` = doc_id (e.g. `201414`).
- `generate_sample_urls_s3.py` – Upload sample PDFs to S3 and write pre-signed URLs to `Samples/sample_document_urls.json` (Stack AI guide).
- `list_sample_refs.py` – Print doc_id → user_id or doc_id → URL from the reference JSON in `Samples/`. and write pre-signed URLs to `Samples/sample_document_urls.json` (Stack AI’s guide).
- `config.py` – Paths (Samples, repo Scripts), model/API settings for reflection and extraction flow.
- `reflection.py` – GEPA-style reflection prompt and LLM call (stub or real API).

## GEPA reflection inside Stack AI (free tokens)

You can run the **reflection LLM step inside Stack AI** so it uses your Stack AI free token allowance instead of OpenAI.

1. **Create a new project** (or use the same org): e.g. "GEPA Reflection".
2. **Build a minimal workflow:**
   - **Input node** (name doesn’t matter; it will be `in-0` in the API): one **long text** input. This will receive the full reflection payload (current prompt + eval report + optional feedback).
   - **LLM node:**
     - **System prompt (Instructions):** paste the exact text from `reflection.py` → `SYSTEM_PROMPT_REFLECTION` (see below).
     - **User prompt:** reference the input node (e.g. `\` and select the input, or `{{in-0}}`), so the LLM gets the payload as the user message.
   - **Output node:** connected to the LLM’s reply (the proposed new prompt).
3. **Publish** the workflow and get the **flow ID** (and org ID) from the API / deploy settings.
4. **config.py** already has org ID and flow ID set for the reflection workflow. Set the env var **`STACK_AI_API_KEY`** to your Stack AI Bearer token (from the workflow’s API / deploy). Do not commit the token.
5. Run the pipeline with **`--reflection-backend stack_ai`**. The script will POST the reflection payload to your Stack AI workflow and use the returned text as the proposed prompt.

**System prompt for the Stack AI Reflection workflow**

Use the **exact** text from `reflection.py` → `SYSTEM_PROMPT_REFLECTION`. It does two things:

1. **Input schema** – Describes the payload the workflow receives (so the model knows what it’s reading):
   - Section 1: "Current system prompt:" then `---`, then the prompt text, then `---`.
   - Section 2: "Evaluation report:" with Overall score, Subscores (structure/numbers/text), and "Top issues" as a numbered list of `[kind] path: expected X got Y` (kind = missing | extra | type | value).
   - Section 3 (optional): "Stack AI evaluator feedback:" and free text.
   - Section 4: The line "Propose a revised system prompt (output only the prompt text):".

2. **Output format** – Strict instructions so the reply is usable as the next prompt:
   - Output **only** the revised system prompt. No prefix ("Here is..."), no markdown code fences, no explanation before or after.
   - First character of the response = first character of the new prompt; last character = last character of the new prompt.

Copy the full constant from `reflection.py` into the LLM node’s **Instructions** so the reflection model knows the payload structure and how to format its answer.

---

## Samples and results

- **Samples/** (in this Experimentation/Bernardo folder) is a copy of `Data/Samples`: ground-truth JSONs and PDFs. Use it for local runs and to save **eval reports** and **extraction outputs** from Stack AI.
- Save Stack AI Batch/Evaluator exports (CSV or JSON) under `Samples/` or a dedicated `results/` subfolder so the script can read them.
