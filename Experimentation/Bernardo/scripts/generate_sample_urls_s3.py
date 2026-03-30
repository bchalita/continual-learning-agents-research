#!/usr/bin/env python3
"""
Generate public/signed URLs for sample PDFs using S3 (Stack AI's guide).

Per Stack AI docs: "To load documents from S3, you can create a pre-signed URL…
Pass the signed URL in the inputs of the flow." Pre-signed URLs are valid up to 12 hours.

Requires:
  - boto3: pip install boto3
  - AWS credentials (env or ~/.aws/credentials): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  - Env: S3_SAMPLES_BUCKET (bucket name), optionally AWS_REGION

Output:
  - Samples/sample_document_urls.json with doc_id -> signed URL (and expiry note).

Usage:
  export S3_SAMPLES_BUCKET=your-bucket-name
  python generate_sample_urls_s3.py
  python generate_sample_urls_s3.py --expiry-hours 12 --samples-dir ../Samples
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure script dir on path for config
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import config
from typing import List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception


def upload_and_signed_url(
    file_path: Path,
    bucket: str,
    key: str,
    *,
    expiry_hours: int = 12,
    region: Optional[str] = None,
) -> str:
    """Upload file to S3 and return a pre-signed GET URL."""
    client = boto3.client("s3", region_name=region)
    with open(file_path, "rb") as f:
        client.upload_fileobj(f, bucket, key, ExtraArgs={"ContentType": "application/pdf"})
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_hours * 3600,
    )
    return url


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload sample PDFs to S3 and generate pre-signed URLs for Stack AI."
    )
    ap.add_argument(
        "--samples-dir",
        type=Path,
        default=config.SAMPLES_DIR,
        help="Directory containing sample PDFs",
    )
    ap.add_argument(
        "--bucket",
        default=os.environ.get("S3_SAMPLES_BUCKET"),
        help="S3 bucket name (or set S3_SAMPLES_BUCKET)",
    )
    ap.add_argument(
        "--expiry-hours",
        type=int,
        default=12,
        help="Pre-signed URL validity in hours (default 12, max 168 for sigv4)",
    )
    ap.add_argument(
        "--skip",
        nargs="*",
        default=["overview.pdf"],
        help="Filename(s) to skip",
    )
    ap.add_argument(
        "--prefix",
        default="samples",
        help="S3 key prefix (default: samples)",
    )
    args = ap.parse_args()

    if not boto3:
        print("Install boto3: pip install boto3", file=sys.stderr)
        sys.exit(1)
    if not args.bucket:
        print("Set S3_SAMPLES_BUCKET or pass --bucket.", file=sys.stderr)
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

    region = os.environ.get("AWS_REGION")
    mapping: List[dict] = []
    for path in pdfs:
        doc_id = path.stem
        key = f"{args.prefix.rstrip('/')}/{path.name}"
        try:
            url = upload_and_signed_url(
                path,
                args.bucket,
                key,
                expiry_hours=args.expiry_hours,
                region=region,
            )
            mapping.append({"doc_id": doc_id, "filename": path.name, "url": url})
            print(f"  {path.name} -> signed URL ({args.expiry_hours}h)")
        except ClientError as e:
            print(f"  FAIL {path.name}: {e}", file=sys.stderr)
            sys.exit(1)

    out_path = samples_dir / "sample_document_urls.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "note": "Pre-signed URLs expire after the specified hours. Re-run this script to refresh.",
                "expiry_hours": args.expiry_hours,
                "bucket": args.bucket,
                "samples": mapping,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
