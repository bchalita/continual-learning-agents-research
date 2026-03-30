#!/usr/bin/env python3
"""
Upload all sample PDFs to Stack AI's Documents API (user bucket) for the extraction flow.

Per Stack AI: you can pass documents via (1) public/signed URLs in the run body, or
(2) upload first via Documents API, then run the flow with the same user_id so the
flow uses the uploaded file.

This script does (2): uploads each PDF so you can run the extraction flow with
user_id = doc_id (e.g. "201414") and no document URL needed.

Requires:
  - config.py: STACK_AI_EXTRACTION_ORG_ID, STACK_AI_EXTRACTION_FLOW_ID set.
  - Env: STACK_AI_PRIVATE_API_KEY (Stack AI *private* API key). The Documents API rejects the *public* key (401).
    In Stack AI: Settings → API Keys → use the Private key. If unset, script tries STACK_AI_API_KEY.

Output:
  - sample_document_user_ids.json in Samples/ with doc_id -> user_id (and filename).
  - Use that user_id in the run request when calling the extraction flow.

Usage:
  python upload_samples_to_stack_ai.py
  python upload_samples_to_stack_ai.py --samples-dir ../Samples
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

# Ensure script dir on path for config
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config
from typing import List, Tuple


def _multipart_form_data(field_name: str, file_path: Path) -> Tuple[bytes, str]:
    """Build multipart/form-data body for one file. Returns (body, boundary)."""
    boundary = "----FormBoundary" + os.urandom(12).hex()
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    filename = file_path.name
    mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    lines = [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"'.encode(),
        f"Content-Type: {mime}".encode(),
        b"",
        file_bytes,
        f"--{boundary}--".encode(),
    ]
    body = b"\r\n".join(lines)
    return body, boundary


def upload_document(
    file_path: Path,
    *,
    org_id: str,
    flow_id: str,
    node_id: str,
    user_id: str,
    api_key: str,
) -> dict:
    """Upload one file to Stack AI documents endpoint. Returns response as dict or error."""
    url = f"https://api.stack-ai.com/documents/{org_id}/{flow_id}/{node_id}/{user_id}"
    body, boundary = _multipart_form_data("file", file_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }
    try:
        from urllib.request import Request, urlopen

        req = Request(url, data=body, headers=headers, method="POST")
        with urlopen(req, timeout=60) as resp:
            return {"ok": True, "status": resp.status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload sample PDFs to Stack AI Documents API for the extraction flow."
    )
    ap.add_argument(
        "--samples-dir",
        type=Path,
        default=config.SAMPLES_DIR,
        help="Directory containing sample PDFs",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=["overview.pdf"],
        help="Filename(s) to skip (default: overview.pdf)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list PDFs that would be uploaded",
    )
    args = ap.parse_args()

    org_id = config.STACK_AI_EXTRACTION_ORG_ID
    flow_id = config.STACK_AI_EXTRACTION_FLOW_ID
    node_id = getattr(config, "STACK_AI_EXTRACTION_NODE_ID", "doc-0")
    # Documents API requires the *private* API key; public key returns 401.
    docs_key_env = getattr(config, "STACK_AI_DOCUMENTS_API_KEY_ENV", "STACK_AI_PRIVATE_API_KEY")
    api_key = os.environ.get(docs_key_env) or os.environ.get(config.STACK_AI_API_KEY_ENV or "STACK_AI_API_KEY")

    if not flow_id:
        print("Set STACK_AI_EXTRACTION_FLOW_ID in config.py (your extraction flow ID from Export → API).", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("Set STACK_AI_PRIVATE_API_KEY in the environment (Stack AI Settings → API Keys → Private key).", file=sys.stderr)
        print("The Documents API returns 401 with the public key; use the Private key for uploads.", file=sys.stderr)
        sys.exit(1)

    samples_dir = args.samples_dir.resolve()
    if not samples_dir.exists():
        print(f"Samples dir not found: {samples_dir}", file=sys.stderr)
        sys.exit(1)

    skip_set = set(args.skip)
    pdfs = sorted(f for f in samples_dir.glob("*.pdf") if f.name not in skip_set)
    if not pdfs:
        print(f"No PDFs found in {samples_dir} (skipping {skip_set}).", file=sys.stderr)
        sys.exit(0)

    # doc_id = filename stem (e.g. 201414)
    mapping: List[dict] = []
    for path in pdfs:
        doc_id = path.stem
        mapping.append({"doc_id": doc_id, "user_id": doc_id, "filename": path.name})

    if args.dry_run:
        print("Would upload (dry-run):")
        for m in mapping:
            print(f"  {m['filename']} -> user_id = {m['user_id']}")
        return

    print(f"Uploading {len(pdfs)} PDFs to Stack AI (org={org_id}, flow={flow_id}, node={node_id})...")
    failed: List[str] = []
    for path in pdfs:
        doc_id = path.stem
        result = upload_document(
            path,
            org_id=org_id,
            flow_id=flow_id,
            node_id=node_id,
            user_id=doc_id,
            api_key=api_key,
        )
        if result.get("ok"):
            print(f"  OK {path.name} -> user_id={doc_id}")
        else:
            err = result.get("error", result)
            print(f"  FAIL {path.name}: {err}", file=sys.stderr)
            if "401" in str(err) and failed == []:
                print("  Hint: 401 Unauthorized usually means the Documents API needs your *Private* API key (Settings → API Keys), not the public key.", file=sys.stderr)
            failed.append(path.name)

    out_path = samples_dir / "sample_document_user_ids.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Use user_id when calling the extraction flow; the document is already in the user bucket.",
                "org_id": org_id,
                "flow_id": flow_id,
                "node_id": node_id,
                "samples": mapping,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")

    if failed:
        print(f"Failed: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
