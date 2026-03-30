#!/usr/bin/env python3
"""
Print sample document references (user_id or URL) from our environment files.

After upload_samples_to_stack_ai.py → reads Samples/sample_document_user_ids.json.
After generate_sample_urls_s3.py   → reads Samples/sample_document_urls.json.

Shows doc_id and how to reference it in run_test_iteration.py or run_extraction_api.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
import config

SAMPLES_DIR = config.SAMPLES_DIR
USER_IDS_FILE = SAMPLES_DIR / "sample_document_user_ids.json"
URLS_FILE = SAMPLES_DIR / "sample_document_urls.json"


def main() -> None:
    if USER_IDS_FILE.exists():
        data = json.loads(USER_IDS_FILE.read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        print("References (uploaded to Stack AI — use --user-id):")
        print(f"  File: {USER_IDS_FILE}")
        for s in samples:
            print(f"  {s['doc_id']:12} → --user-id {s['user_id']}")
        print("\nExample: python3 run_test_iteration.py --user-id 201414")
        return

    if URLS_FILE.exists():
        data = json.loads(URLS_FILE.read_text(encoding="utf-8"))
        samples = data.get("samples", [])
        print("References (S3 URLs — use --doc-url):")
        print(f"  File: {URLS_FILE}")
        for s in samples:
            url = s.get("url", "")[:60] + "..." if len(s.get("url", "")) > 60 else s.get("url", "")
            print(f"  {s['doc_id']:12} → {url}")
        print("\nExample: python3 run_test_iteration.py --doc-url \"<url from file>\"")
        return

    print("No reference file found. Run one of:", file=sys.stderr)
    print("  python upload_samples_to_stack_ai.py   # → sample_document_user_ids.json", file=sys.stderr)
    print("  python generate_sample_urls_s3.py     # → sample_document_urls.json (needs S3)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
