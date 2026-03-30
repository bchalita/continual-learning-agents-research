#!/usr/bin/env python3
"""
Run a single test iteration of the extraction workflow: baseline prompt + one document → Stack AI API → result.

Use this to verify the end-to-end flow works before batch runs or GEPA loops. Messages printed to the
console tell you what step is running and whether the call succeeded or failed.

Default document: 201414 (use --user-id 201414 after upload_samples_to_stack_ai.py, or --doc-url when you have a public URL).
Default prompt: scripts/baseline_prompt.txt.
Optional: pass ground truth (Samples/201414.json) so the flow can use it if configured.

Usage:
  # After uploading samples to Stack AI (user_id = doc_id):
  python3 run_test_iteration.py --user-id 201414

  # With a public document URL:
  python3 run_test_iteration.py --doc-url "https://example.com/201414.pdf"

  # Save extraction result to a file for later eval:
  python3 run_test_iteration.py --user-id 201414 --save-output
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config
from run_extraction_api import run_extraction, _normalize_doc_url

# Default baseline prompt (short fallback if file missing)
FALLBACK_PROMPT = (
    "You are an expert at extracting structured data from automotive work order and repair order documents. "
    "Read the document and extract all relevant information into a single JSON object with top-level keys: doc_id, sections. "
    "Each section: section_id, prefix, page_count, header, footer, content. Return only valid JSON, no markdown or explanation."
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run one test iteration: baseline prompt + document → Stack AI extraction → print result."
    )
    ap.add_argument(
        "--doc-url",
        default=None,
        help="Public URL of the PDF (Stack AI downloads it).",
    )
    ap.add_argument(
        "--user-id",
        default=None,
        help="User ID for pre-uploaded document (e.g. 201414 after upload_samples_to_stack_ai.py).",
    )
    ap.add_argument(
        "--prompt",
        default=None,
        help="Inline system prompt text (if set, overrides --prompt-file).",
    )
    ap.add_argument(
        "--prompt-file",
        type=Path,
        default=_here / "baseline_prompt.txt",
        help="Path to baseline system prompt text file (default: scripts/baseline_prompt.txt). Used only when --prompt is not set.",
    )
    ap.add_argument(
        "--ground-truth-file",
        type=Path,
        default=config.SAMPLES_DIR / "201414.json",
        help="Path to ground truth JSON for in-1 (default: Samples/201414.json). Omitted if file not found.",
    )
    ap.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Do not send ground truth (in-1) even if file exists.",
    )
    ap.add_argument(
        "--save-output",
        action="store_true",
        help="Save extraction result (out-0) to Samples/201414_extracted_test.json.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print full API response (not just extraction output).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print request payload and full response to verify API keys and Stack AI output structure.",
    )
    ap.add_argument(
        "--verify-url-only",
        action="store_true",
        help="Only verify that the document URL returns a PDF (no API call). Use to confirm Dropbox/dl=1 works.",
    )
    ap.add_argument(
        "--doc-as-string",
        action="store_true",
        help="Send doc-0 as a single URL string instead of [url]. Try this if the flow ignores the document when sent as an array.",
    )
    args = ap.parse_args()

    if not args.doc_url and not args.user_id:
        # Optional: try sample_document_urls.json for a default doc URL (e.g. 201414)
        urls_path = config.SAMPLES_DIR / "sample_document_urls.json"
        if urls_path.exists():
            try:
                urls_data = json.loads(urls_path.read_text(encoding="utf-8"))
                samples = urls_data.get("samples") or []
                if samples and samples[0].get("url"):
                    args.doc_url = samples[0]["url"]
                    print(f"[Doc URL] Using URL from {urls_path} (doc_id: {samples[0].get('doc_id', '?')})")
            except Exception:
                pass
        if not args.doc_url and not args.user_id:
            print("ERROR: Provide either --doc-url <public URL> or --user-id <id> (e.g. 201414).", file=sys.stderr)
            print("  If you have sample_document_urls.json in Samples/, run from scripts/ and omit --doc-url to use the first URL.", file=sys.stderr)
            print("  If you uploaded samples: python3 run_test_iteration.py --user-id 201414", file=sys.stderr)
            print("  If you have a public URL:  python3 run_test_iteration.py --doc-url 'https://...'", file=sys.stderr)
            sys.exit(1)

    # Step 1: Load baseline prompt (inline --prompt takes precedence over --prompt-file)
    if args.prompt is not None and args.prompt.strip():
        prompt = args.prompt.strip()
        print(f"[Step 1] Using inline prompt ({len(prompt)} chars)")
    else:
        prompt_path = args.prompt_file.resolve()
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
            print(f"[Step 1] Loaded baseline prompt from {prompt_path} ({len(prompt)} chars)")
        else:
            prompt = FALLBACK_PROMPT
            print(f"[Step 1] Prompt file not found, using built-in fallback ({len(prompt)} chars)")

    # Step 2: Ground truth (optional)
    ground_truth = None
    if not args.no_ground_truth and args.ground_truth_file.exists():
        ground_truth = args.ground_truth_file.read_text(encoding="utf-8")
        print(f"[Step 2] Loaded ground truth from {args.ground_truth_file} (will send as in-1)")
    else:
        print("[Step 2] No ground truth sent (optional)")

    # Step 2.5 (optional): Verify document URL returns a PDF (Stack AI will fetch from this URL)
    if args.doc_url:
        u = _normalize_doc_url(args.doc_url) if "dropbox" in args.doc_url.lower() else args.doc_url
        print(f"[Step 3] Document URL: {u[:90]}{'...' if len(u) > 90 else ''}")
        try:
            import urllib.request
            # GET with Range so we don't download the whole file (Dropbox often doesn't support HEAD)
            req = urllib.request.Request(u)
            req.add_header("User-Agent", "StackAI-DocCheck/1.0")
            req.add_header("Range", "bytes=0-0")
            with urllib.request.urlopen(req, timeout=15) as r:
                ct = r.headers.get("Content-Type", "")
                status = getattr(r, "status", 200)
                if status in (200, 206) and ("pdf" in ct.lower() or "octet-stream" in ct.lower() or "binary" in ct.lower()):
                    print(f"[Step 3] URL check: OK (HTTP {status}, {ct})")
                else:
                    print(f"[Step 3] URL check: WARN — expected PDF, got HTTP {status} Content-Type: {ct}")
        except Exception as e:
            print(f"[Step 3] URL check: FAIL — {e}")
            print(">>> If the URL is wrong or Dropbox returns a page instead of the file, Stack AI will not see the document.")
            if not args.verify_url_only:
                print(">>> Fix: use Dropbox link with dl=1, or try --verify-url-only to test the URL without calling the API.")
        if args.verify_url_only:
            print("Done (verify-url-only). No API call.")
            return

    print("[Step 3] Calling Stack AI extraction API...")
    try:
        response, payload_sent = run_extraction(
            prompt,
            doc_url=args.doc_url,
            user_id=args.user_id,
            ground_truth=ground_truth,
            doc_as_list=not args.doc_as_string,
        )
    except Exception as e:
        print(f"[Step 3] FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    # Save full API request + response to a log file for evaluation
    logs_dir = _here / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = logs_dir / f"api_run_{ts}.json"
    log_data = {
        "timestamp_utc": ts,
        "request": payload_sent,
        "response": response,
    }
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"[Log] Full API request/response saved to: {log_path}")
    except Exception as e:
        print(f"[Log] Could not write log file: {e}", file=sys.stderr)

    if args.debug:
        print("\n[DEBUG] Request payload (input keys sent to Stack AI):")
        for k in sorted(payload_sent.keys()):
            v = payload_sent[k]
            if k in ("in-0", "in-1") and isinstance(v, str) and len(v) > 200:
                print(f"  {k}: <string, {len(v)} chars>")
            else:
                print(f"  {k}: {v}")
        print("\n[DEBUG] Full API response:")
        print(json.dumps(response, indent=2, default=str))

    # Step 4: Interpret response
    if "error" in response:
        print(f"[Step 4] API returned an error: {response['error']}", file=sys.stderr)
        if args.verbose:
            print(json.dumps(response, indent=2))
        sys.exit(1)

    # Stack AI can return out-0, outputs.llm-0.completion, or nested structure. Prefer actual LLM extraction over prompt text.
    def is_likely_extraction_string(s):
        """True only if string looks like extraction JSON (starts with { or ```json {), not instruction text."""
        if not isinstance(s, str) or len(s.strip()) < 50:
            return False
        s = s.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[-1].strip()
            if s.startswith("json"):
                s = s[4:].strip()
        if not s.startswith("{"):
            return False
        return "doc_id" in s[:800] or '"sections"' in s[:800]

    def find_extraction(obj, depth=0):
        """Recursively find a string or dict that looks like LLM extraction (JSON with doc_id/sections)."""
        if depth > 10:
            return None
        if isinstance(obj, str):
            return obj if is_likely_extraction_string(obj) else None
        if isinstance(obj, dict):
            # Prefer dict that is clearly extraction (has doc_id/sections), not a wrapper
            if "doc_id" in obj or "sections" in obj:
                return obj
            # LLM node output: { "completion": "...", ... }
            if "completion" in obj and obj.get("completion") and is_likely_extraction_string(str(obj["completion"])):
                return obj["completion"]
            for k in ("out-0", "output", "text", "result", "content", "message", "completion"):
                if k in obj and obj[k]:
                    cand = obj[k]
                    if isinstance(cand, str) and is_likely_extraction_string(cand):
                        return cand
                    if isinstance(cand, dict) and ("doc_id" in cand or "sections" in cand):
                        return cand
            for v in obj.values():
                found = find_extraction(v, depth + 1)
                if found is not None:
                    return found
            return None
        if isinstance(obj, list):
            for item in reversed(obj):
                found = find_extraction(item, depth + 1)
                if found is not None:
                    return found
            return None
        return None

    # Prefer Extractor LLM completion (outputs["llm-0"] or similar), then out-0, then recursive search.
    # Reject strings that are the system prompt (instruction text), not extraction JSON.
    def is_prompt_not_extraction(s):
        if not isinstance(s, str) or not s.strip():
            return False
        t = s.strip()
        return t.startswith("You are an expert") and not t.startswith("{")

    outputs = response.get("outputs") or {}
    extraction = None
    for node_key in ("llm-0", "llm_0", "LLM Executor", "extraction"):
        node_out = outputs.get(node_key) if isinstance(outputs, dict) else None
        if node_out and isinstance(node_out, dict) and node_out.get("completion"):
            s = node_out["completion"]
            if is_prompt_not_extraction(s):
                continue
            extraction = s
            break
    if extraction is None:
        extraction = (
            response.get("out-0")
            or response.get("extraction_i")
            or response.get("output")
            or find_extraction(outputs)
            or find_extraction(response)
        )
    if extraction is not None and is_prompt_not_extraction(str(extraction)):
        extraction = None
    if extraction is not None:
        print("[Step 4] Response received successfully.")
        print("--- Extraction output ---")
        if isinstance(extraction, str):
            print(extraction[:2000] + ("..." if len(extraction) > 2000 else ""))
        else:
            print(json.dumps(extraction, indent=2)[:2000] + ("..." if len(json.dumps(extraction)) > 2000 else ""))

        # Detect empty extraction and point to flow configuration
        def _is_empty_extraction(ext):
            if isinstance(ext, dict):
                return ext.get("doc_id") is None and (not ext.get("sections") or ext.get("sections") == [])
            if isinstance(ext, str):
                return '"doc_id": null' in ext and '"sections": []' in ext
            return False
        if _is_empty_extraction(extraction):
            print("")
            print(">>> Empty extraction (doc_id: null, sections: []). The LLM is not seeing the document.")
            print(">>> Fix in Stack AI: see scripts/README.md section 'Troubleshooting: Empty extraction'.")

        # Full-flow output: eval, evaluator feedback, proposed prompt (if Output node exposes them)
        outputs = response.get("outputs") or {}
        for key, label in (
            ("eval_result", "Eval result (deterministic)"),
            ("evaluator_feedback", "Evaluator feedback (LLM)"),
            ("proposed_prompt", "Proposed prompt (Reflection)"),
        ):
            val = response.get(key) or (outputs.get(key) if isinstance(outputs, dict) else None)
            if val is not None:
                print("")
                print(f"--- {label} ---")
                if isinstance(val, str):
                    print(val[:3000] + ("..." if len(val) > 3000 else ""))
                elif isinstance(val, dict):
                    print(json.dumps(val, indent=2, default=str)[:3000] + ("..." if len(json.dumps(val, default=str)) > 3000 else ""))
                else:
                    print(val)
    else:
        print("[Step 4] Response received; extraction not found in out-0/outputs. Full keys:", list(response.keys()))
        print(">>> If the API returned the system prompt instead of JSON, wire the Output node to the Extractor LLM (llm-0) completion, not in-0.")
        # Show structure of outputs so you can see where extraction lives (or add --verbose for full response)
        outs = response.get("outputs")
        if outs is not None:
            try:
                preview = json.dumps(outs, indent=2, default=str)[:1200]
                print("[Step 4] outputs structure (preview):", preview + ("..." if len(str(outs)) > 1200 else ""))
            except Exception:
                print("[Step 4] outputs type:", type(outs).__name__, "repr:", repr(outs)[:600])
        extraction = response

    if args.verbose:
        print("--- Full API response ---")
        print(json.dumps(response, indent=2, default=str))

    # Optional: save to file for eval
    if args.save_output and extraction is not None:
        out_path = config.SAMPLES_DIR / "201414_extracted_test.json"
        if isinstance(extraction, str):
            try:
                # Try to parse as JSON so we save pretty-printed
                data = json.loads(extraction)
                out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except json.JSONDecodeError:
                out_path.write_text(extraction, encoding="utf-8")
        else:
            out_path.write_text(json.dumps(extraction, indent=2), encoding="utf-8")
        print(f"[Save] Wrote extraction to {out_path} (run eval.py against 201414.json to compare)")

    print("Done. Workflow test completed.")


if __name__ == "__main__":
    main()
