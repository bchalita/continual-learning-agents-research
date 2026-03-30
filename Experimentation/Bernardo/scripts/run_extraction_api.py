#!/usr/bin/env python3
"""
Call the Stack AI extraction flow via API.

Provide the document either as:
  - A public URL (--doc-url): Stack AI downloads the PDF from that URL.
  - A user_id (--user-id): After uploading via upload_samples_to_stack_ai.py, pass the same user_id and the flow uses the uploaded file.

Set in config.py: STACK_AI_EXTRACTION_ORG_ID, STACK_AI_EXTRACTION_FLOW_ID.
Set STACK_AI_API_KEY in the environment.

Usage:
  python run_extraction_api.py --doc-url "https://..." --prompt "Your system prompt..."
  python run_extraction_api.py --user-id 201414 --prompt "Your system prompt..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

# Ensure script dir on path for config
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config


def _normalize_doc_url(url: str) -> str:
    """Use direct-download link for Dropbox (dl=1); leave other URLs unchanged."""
    parsed = urlparse(url)
    if "dropbox.com" not in parsed.netloc:
        return url
    qs = parse_qs(parsed.query)
    qs["dl"] = ["1"]
    new_query = urlencode(qs, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def run_extraction(
    system_prompt: str,
    *,
    doc_url: Optional[str] = None,
    user_id: Optional[str] = None,
    ground_truth: Optional[str] = None,
    ground_truth_input_key: str = "in-1",
    org_id: Optional[str] = None,
    flow_id: Optional[str] = None,
    doc_input_key: Optional[str] = None,
    prompt_input_key: str = "in-0",
    doc_as_list: bool = True,
    api_key_env: str = "STACK_AI_API_KEY",
    base_url: Optional[str] = None,
) -> Tuple[dict, dict]:
    """POST to Stack AI extraction flow. Pass doc_url (public URL; sent as [url] if doc_as_list) or user_id (if document was uploaded via Documents API). Optionally pass ground_truth (in-1) for evaluation. Returns (response, payload_used) for debugging."""
    org_id = org_id or config.STACK_AI_EXTRACTION_ORG_ID
    flow_id = flow_id or config.STACK_AI_EXTRACTION_FLOW_ID
    doc_input_key = doc_input_key or config.STACK_AI_EXTRACTION_DOC_INPUT_KEY
    base_url = (base_url or config.STACK_AI_BASE_URL).rstrip("/")
    api_key = os.environ.get(api_key_env or config.STACK_AI_API_KEY_ENV or "STACK_AI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} in the environment (Stack AI Bearer token).")
    if not org_id or not flow_id:
        raise RuntimeError(
            "Set STACK_AI_EXTRACTION_ORG_ID and STACK_AI_EXTRACTION_FLOW_ID in config.py."
        )
    if not doc_url and not user_id:
        raise RuntimeError("Pass either doc_url (public URL) or user_id (after uploading via upload_samples_to_stack_ai.py).")

    url = f"{base_url}/{org_id}/{flow_id}"
    payload = {
        "user_id": user_id or "",
        prompt_input_key: system_prompt,
    }
    if doc_url:
        doc_url = _normalize_doc_url(doc_url)
        # Some Stack AI flows (e.g. document node) expect doc-0 as a list of URLs.
        payload[doc_input_key] = [doc_url] if doc_as_list else doc_url
    if ground_truth is not None:
        payload[ground_truth_input_key] = ground_truth
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return (json.loads(resp.read().decode()), payload)
    except Exception as e:
        return ({"error": str(e), "payload": payload}, payload)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Call Stack AI extraction flow with a document (URL or pre-uploaded via user_id) and system prompt."
    )
    ap.add_argument("--doc-url", default=None, help="Public URL of the PDF (Stack AI will download it).")
    ap.add_argument("--user-id", default=None, help="User ID used when uploading the document (use after upload_samples_to_stack_ai.py).")
    ap.add_argument("--prompt", required=True, help="System prompt / instructions for the extraction LLM.")
    ap.add_argument("--prompt-input-key", default="in-0", help="Input key for prompt in the flow (default: in-0).")
    ap.add_argument("--doc-input-key", default=None, help="Input key for document (default from config: doc-0).")
    ap.add_argument("--doc-as-string", action="store_true", help="Send doc-0 as a single URL (default is list [url] for Stack AI document node).")
    ap.add_argument("--ground-truth", default=None, help="Ground truth JSON string for in-1 (e.g. for Stack AI evaluator).")
    ap.add_argument("--ground-truth-file", type=Path, default=None, help="Path to ground truth JSON file; contents sent as in-1.")
    args = ap.parse_args()
    doc_as_list = not args.doc_as_string

    if not args.doc_url and not args.user_id:
        ap.error("Provide either --doc-url or --user-id.")

    ground_truth = args.ground_truth
    if args.ground_truth_file and args.ground_truth_file.exists():
        ground_truth = args.ground_truth_file.read_text(encoding="utf-8")
    if args.ground_truth_file and ground_truth is None:
        ap.error(f"File not found: {args.ground_truth_file}")

    out, _ = run_extraction(
        system_prompt=args.prompt,
        doc_url=args.doc_url,
        user_id=args.user_id,
        ground_truth=ground_truth,
        prompt_input_key=args.prompt_input_key,
        doc_input_key=args.doc_input_key,
        doc_as_list=doc_as_list,
    )
    print(json.dumps(out, indent=2))
    if "error" in out:
        sys.exit(1)


if __name__ == "__main__":
    main()
