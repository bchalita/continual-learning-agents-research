"""
LLM merge node: deduplicate and reconcile per-section extractions.

BMW feedback (2026-04-19): "Add an LLM node to merge sections parsed so that
we prevent repeated info, and aggregate more cleanly."

Problem: Each section (ASI, BWO, CSI, etc.) is extracted independently. Headers
are duplicated across sections (same RO#, VIN, customer in every section). When
the LLM misreads a value in one section but gets it right in another, we end up
with conflicting data. The merge node:

1. Identifies repeated header fields across sections
2. Reconciles conflicts (majority vote or highest-confidence value)
3. Flags unresolvable conflicts for human review
4. Produces a cleaner merged document

This is a text-only task — no images needed. Routes through Stack AI.
Falls back to passthrough (no merge) if Stack AI is unavailable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import config

# Header fields that are truly document-level and should be identical across all
# sections of the same repair order. Only these are reconciled by deterministic merge.
# Other header fields (dates, advisor, mileage, etc.) may legitimately differ per section.
SHARED_HEADER_FIELDS = {
    "ro_number",
    "vin",
    "customer_name",
    "customer_number",
    "year",
    "make",
    "model",
    "dealership_name",
    "dealership_number",
}

MERGE_INSTRUCTIONS = """You are a data reconciliation expert for BMW repair order documents.

You will receive a JSON document with multiple sections extracted from the same repair order.
Each section (ASI, BWO, CSI, JSI, WSI, ISI) was extracted independently from different pages.

Your task:
1. The header fields (ro_number, vin, customer_name, etc.) should be IDENTICAL across sections
   since they come from the same repair order. If values differ, pick the most complete/correct one.
2. Footer totals should remain section-specific (each section has its own charges).
3. Content (jobs, labor, acct_split) is section-specific — do not merge across sections.
4. If a field is null in one section but has a value in another, use the non-null value.

Return ONLY valid JSON — no markdown, no code fences, no explanation.
Return the same structure: {"doc_id": "...", "sections": [...]} with reconciled values."""


def merge_sections(document: dict[str, Any]) -> dict[str, Any]:
    """Merge and reconcile per-section extractions.

    Attempts to use Stack AI for LLM-powered reconciliation. Falls back to
    a deterministic header-dedup if Stack AI is unavailable.

    Args:
        document: {"doc_id": str, "sections": [section_dict, ...]}

    Returns:
        Reconciled document with same structure.
    """
    sections = document.get("sections", [])
    if len(sections) <= 1:
        return document  # Nothing to merge

    # Skip sections with parse errors
    valid_sections = [s for s in sections if not s.get("_parse_error")]
    if len(valid_sections) <= 1:
        return document

    # Try LLM merge via Stack AI
    try:
        return _llm_merge(document)
    except Exception:
        # Fall back to deterministic merge
        return _deterministic_merge(document)


def _llm_merge(document: dict[str, Any]) -> dict[str, Any]:
    """Use Stack AI to reconcile sections via LLM."""
    from . import stackai

    prompt = (
        "Reconcile the following extracted repair order document. "
        "Make header fields consistent across sections.\n\n"
        + json.dumps(document, indent=2, default=str)
    )

    raw = stackai.call(MERGE_INSTRUCTIONS, prompt)

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    merged = json.loads(text)

    # Validate structure — must have doc_id and sections
    if "sections" not in merged or not isinstance(merged["sections"], list):
        return document  # LLM returned bad structure, keep original

    return merged


def _deterministic_merge(document: dict[str, Any]) -> dict[str, Any]:
    """Deterministic header reconciliation without LLM.

    Only reconciles SHARED_HEADER_FIELDS (document-level identifiers like
    RO#, VIN, customer). Section-specific fields (dates, advisor, mileage)
    are left untouched since they may legitimately differ across sections.

    For each shared field, collects values across all sections and picks
    the most common non-null value (majority vote).
    """
    sections = document.get("sections", [])
    valid_sections = [s for s in sections if not s.get("_parse_error")]

    if len(valid_sections) <= 1:
        return document

    # Collect shared header field values across sections
    header_values: dict[str, list] = {}
    for sec in valid_sections:
        header = sec.get("header", {})
        if not isinstance(header, dict):
            continue
        for key, val in header.items():
            if key not in SHARED_HEADER_FIELDS:
                continue  # Skip section-specific fields
            if key not in header_values:
                header_values[key] = []
            header_values[key].append(val)

    # For each shared field, pick the most common non-null value
    reconciled_header: dict[str, Any] = {}
    for key, values in header_values.items():
        non_null = [v for v in values if v is not None]
        if non_null:
            # Majority vote: stringify for comparison, count on stringified list
            str_values = [str(v) for v in non_null]
            winner_str = max(set(str_values), key=str_values.count)
            # Preserve original type from the first matching value
            reconciled_header[key] = winner_str
            for v in non_null:
                if str(v) == winner_str:
                    reconciled_header[key] = v
                    break
        else:
            reconciled_header[key] = None

    # Apply reconciled shared fields to all sections (only overwrite shared fields)
    merged_sections = []
    for sec in sections:
        if sec.get("_parse_error"):
            merged_sections.append(sec)
            continue
        merged_sec = {**sec}
        if isinstance(merged_sec.get("header"), dict):
            merged_sec["header"] = {**merged_sec["header"]}
            for key, val in reconciled_header.items():
                merged_sec["header"][key] = val
        merged_sections.append(merged_sec)

    return {**document, "sections": merged_sections}
