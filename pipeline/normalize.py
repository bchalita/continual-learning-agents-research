"""Post-extraction type normalization.

Fixes common type mismatches between model output and ground truth schema:
- Account codes (acct) as int instead of string
- IDs/counts as int (advisor_id, tech, miles, stock_number, etc.)
- Money/hours as float
- Year normalization (2-digit string "19" vs 4-digit int 2019 → int per schema)

This is a deterministic transform applied AFTER extraction, BEFORE evaluation.
It lets the GEPA loop focus on structural/content accuracy rather than type coercion.
"""

from __future__ import annotations

import re
from typing import Any


# Fields that MUST be int (per ground truth schema)
INT_FIELDS = {
    # Header
    "unit_number", "advisor_id", "miles_in", "miles_out",
    "stock_number", "customer_number", "ro_number", "year",
    # Content
    "tech", "miles", "qty", "est_advisor",
    # acct_split
    "acct",
}

# Fields that MUST be float
FLOAT_FIELDS = {
    # Header
    "rate",
    # Footer
    "labor_amount", "parts_amount", "gas_oil_lube", "sublet_amount",
    "misc_charges", "total_charges", "less_insurance", "sales_tax", "please_pay",
    # Content - money/hours
    "hours", "a_hrs", "s_hrs", "cost", "sale", "comp", "list", "net", "total",
    "duration", "charge",
    "parts_cost", "parts_sale", "labor_cost", "labor_sale",
    "total_cost", "total_sale", "total_comp", "total_labor", "total_parts",
    "total_other", "total_lub", "subtotal", "est_amount",
}


def _try_int(value: Any) -> Any:
    """Attempt to coerce value to int. Return original if not possible."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        # Only convert if it's a whole number
        if value == int(value):
            return int(value)
        return value
    if isinstance(value, str):
        # Strip whitespace and try conversion
        s = value.strip()
        if not s:
            return None
        # Handle comma-separated numbers like "1,237"
        s_clean = s.replace(",", "")
        try:
            return int(s_clean)
        except ValueError:
            # Try float then int (e.g., "1237.0")
            try:
                f = float(s_clean)
                if f == int(f):
                    return int(f)
            except ValueError:
                pass
    return value


def _try_float(value: Any) -> Any:
    """Attempt to coerce value to float. Return original if not possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Handle currency symbols and commas
        s_clean = re.sub(r"[$,]", "", s)
        try:
            return float(s_clean)
        except ValueError:
            pass
    return value


def normalize_value(key: str, value: Any) -> Any:
    """Normalize a single field value based on its key name."""
    if value is None:
        return None
    if key in INT_FIELDS:
        return _try_int(value)
    if key in FLOAT_FIELDS:
        return _try_float(value)
    return value


def normalize_dict(d: dict) -> dict:
    """Recursively normalize all fields in a dict."""
    if not isinstance(d, dict):
        return d

    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = normalize_dict(value)
        elif isinstance(value, list):
            result[key] = normalize_list(key, value)
        else:
            result[key] = normalize_value(key, value)
    return result


def normalize_list(parent_key: str, lst: list) -> list:
    """Normalize a list of items."""
    if not isinstance(lst, list):
        return lst

    result = []
    for item in lst:
        if isinstance(item, dict):
            result.append(normalize_dict(item))
        elif isinstance(item, list):
            result.append(normalize_list(parent_key, item))
        else:
            result.append(normalize_value(parent_key, item))
    return result


def normalize_document(document: dict) -> dict:
    """Normalize all fields in a full extraction result.

    Args:
        document: {"doc_id": str, "sections": [...]} or just a sections list

    Returns:
        Same structure with type-normalized values.
    """
    if "sections" in document:
        normalized_sections = []
        for section in document["sections"]:
            normalized_sections.append(normalize_dict(section))
        return {**document, "sections": normalized_sections}

    # If it's a flat section dict
    return normalize_dict(document)
