#!/usr/bin/env python3
"""
JSON evaluation: compare ground-truth synthetic JSON to PDF extraction output.

Produces a scored report with per-field issue diagnostics suitable for
feeding back into extraction prompt tuning.

Dependencies:
  pip install deepdiff

Usage:
  python eval.py --gt ground_truth.json --pred prediction.json
  python eval.py --gt gt.json --pred pred.json --out report.json
  python eval.py --gt gt.json --pred pred.json --config eval_config.json

Public API:
  evaluate_extraction(ground_truth, prediction, config=None) -> dict

Changelog:
  2026-04-19 (Bernardo Chalita) — Section-aware alignment & missing-section penalties
    PROBLEM: The original eval compared sections by positional index (sections[0]
    vs sections[0], etc.). This caused two critical bugs:
      1. If the prediction reordered sections (e.g., BWO first instead of ASI),
         every field in both sections would show as a "value" mismatch even though
         the extraction was correct — just in a different order.
      2. If the prediction had FEWER sections than ground truth, the missing
         sections were recorded as a single "missing list item" issue with a
         trivial 0.08 penalty — regardless of how many fields that section
         contained (often 40-60 fields). This meant a prediction that missed
         entire sections could still score >0.90.
    FIX: Before running DeepDiff, sections are now matched by their "prefix"
    field (ASI, BWO, CSI, etc.). Matched sections are compared field-by-field.
    Missing sections are replaced with {} so DeepDiff naturally discovers every
    missing field and assigns per-field penalties. Extra predicted sections are
    similarly caught. A new "section_alignment" block in the report shows exactly
    which sections were matched, missing, or extra.
    IMPACT: Scores will drop for predictions that miss sections or reorder them
    incorrectly. This is intentional — the previous scores were inflated.
    CONFIG: Section alignment is enabled by default. To restore legacy behavior,
    set config.section_alignment.enabled = false.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any

from deepdiff import DeepDiff


# ── Config ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "normalization": {
        "strip_strings": True,
        "collapse_whitespace": True,
        "empty_is_null": True,
        "missing_is_null": True,
    },
    "diff": {
        "ignore_numeric_type_changes": True,
        "significant_digits": 2,
        "verbose_level": 2,
        "exclude_paths": [],
        "exclude_path_regexes": [],
    },
    # Section alignment config — added 2026-04-19 (Bernardo Chalita)
    # Matches GT and prediction sections by "prefix" field before comparison.
    # Without this, DeepDiff compares sections by positional index, which
    # causes incorrect scoring when sections are reordered or missing.
    "section_alignment": {
        "enabled": True,
        # The field used to match GT sections to prediction sections.
        # BMW repair orders use "prefix" (ASI, BWO, CSI, JSI, WSI, ISI).
        "match_key": "prefix",
    },
    "rubric": {
        "category_weights": {
            "structure": 0.45,
            "numbers": 0.40,
            "text": 0.15,
        },
        "penalties": {
            "structure": {"missing": 0.08, "extra": 0.04, "type": 0.06, "value": 0.03},
            "numbers":   {"missing": 0.08, "extra": 0.02, "type": 0.06, "value": 0.06},
            "text":      {"missing": 0.05, "extra": 0.02, "type": 0.03, "value": 0.04},
        },
        "category_caps": {
            "structure": 0.90,
            "numbers": 0.90,
            "text": 0.80,
        },
        "max_issues": 200,
        "severity_thresholds": {
            "high": 0.07,
            "med": 0.04,
            "low": 0.0,
        },
    },
}


# ── Issue ───────────────────────────────────────────────────────────────

@dataclass
class Issue:
    category: str   # structure | numbers | text
    kind: str       # missing | extra | type | value
    path: str
    expected: Any = None
    got: Any = None
    detail: str = ""
    penalty: float = 0.0
    severity: str = "low"


# ── Utilities ───────────────────────────────────────────────────────────

def _humanize_path(path: str) -> str:
    """root['sections'][0]['content'] → sections[0].content"""
    if not path.startswith("root"):
        return path
    p = path[4:]
    p = re.sub(r"\['([^']+)'\]", r".\1", p)
    return p.lstrip(".")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _categorize(path: str, expected: Any, got: Any, kind: str) -> str:
    """Assign an issue to a scoring category based on the values involved."""
    if kind in ("missing", "extra"):
        if _is_number(expected) or _is_number(got):
            return "numbers"
        if isinstance(expected, str) or isinstance(got, str):
            return "text"
        return "structure"
    if _is_number(expected) or _is_number(got):
        return "numbers"
    if isinstance(expected, str) or isinstance(got, str):
        return "text"
    return "structure"


def _normalize(obj: Any, cfg: dict[str, Any]) -> Any:
    """Normalize whitespace; optionally treat empty strings as null."""
    strip = cfg.get("strip_strings", True)
    collapse = cfg.get("collapse_whitespace", True)
    empty_is_null = cfg.get("empty_is_null", True)

    def walk(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str):
            y = x.strip() if strip else x
            if collapse:
                y = re.sub(r"\s+", " ", y).strip()
            if empty_is_null and y == "":
                return None
            return y
        return x

    return walk(obj)


# ── Tree-view diff → Issues ────────────────────────────────────────────

def _diff_to_issues(
    dd: DeepDiff,
    rubric: dict[str, Any],
    diff_cfg: dict[str, Any] | None = None,
    missing_is_null: bool = True,
    _depth: int = 0,
) -> list[Issue]:
    """Walk DeepDiff tree-view result and emit one Issue per leaf difference.

    Container-level value changes (both sides are dict/list) are recursively
    decomposed into leaf-level issues via a sub-DeepDiff.
    """
    MAX_DECOMPOSE_DEPTH = 4
    penalties_cfg = rubric["penalties"]
    thresholds = rubric["severity_thresholds"]
    issues: list[Issue] = []

    def _penalty(cat: str, kind: str) -> float:
        cat_tbl = penalties_cfg.get(cat, penalties_cfg.get("text", {}))
        return float(cat_tbl.get(kind, 0.03))

    def _severity(p: float) -> str:
        if p >= thresholds.get("high", 0.07):
            return "high"
        if p >= thresholds.get("med", 0.04):
            return "med"
        return "low"

    def emit(kind: str, path: str, exp: Any, got: Any, detail: str = ""):
        cat = _categorize(path, exp, got, kind)
        pen = _penalty(cat, kind)
        issues.append(Issue(cat, kind, path, exp, got, detail, pen, _severity(pen)))

    for item in dd.get("values_changed", []):
        path = _humanize_path(item.path())
        if (
            isinstance(item.t1, (dict, list))
            and isinstance(item.t2, (dict, list))
            and _depth < MAX_DECOMPOSE_DEPTH
        ):
            sub_dd = DeepDiff(
                item.t1, item.t2,
                view="tree",
                ignore_numeric_type_changes=(
                    diff_cfg.get("ignore_numeric_type_changes", True) if diff_cfg else True
                ),
                significant_digits=diff_cfg.get("significant_digits", 2) if diff_cfg else 2,
                verbose_level=2,
            )
            sub_issues = _diff_to_issues(
                sub_dd, rubric, diff_cfg, missing_is_null, _depth + 1,
            )
            for si in sub_issues:
                si.path = f"{path}.{si.path}" if si.path else path
            if sub_issues:
                issues.extend(sub_issues)
                continue
        emit("value", path, item.t1, item.t2)

    for item in dd.get("type_changes", []):
        detail = f"{type(item.t1).__name__} \u2192 {type(item.t2).__name__}"
        emit("type", _humanize_path(item.path()), item.t1, item.t2, detail)

    for item in dd.get("dictionary_item_removed", []):
        if missing_is_null and item.t1 is None:
            continue
        emit("missing", _humanize_path(item.path()), item.t1, None, "missing key")

    for item in dd.get("dictionary_item_added", []):
        if missing_is_null and item.t2 is None:
            continue
        emit("extra", _humanize_path(item.path()), None, item.t2, "extra key")

    for item in dd.get("iterable_item_removed", []):
        emit("missing", _humanize_path(item.path()), item.t1, None, "missing list item")

    for item in dd.get("iterable_item_added", []):
        emit("extra", _humanize_path(item.path()), None, item.t2, "extra list item")

    return issues


# ── Scoring ─────────────────────────────────────────────────────────────

def _score(issues: list[Issue], rubric: dict[str, Any]) -> dict[str, Any]:
    """Compute weighted category subscores and an aggregate score."""
    caps = rubric["category_caps"]
    weights = rubric["category_weights"]

    raw: dict[str, float] = {}
    for it in issues:
        raw[it.category] = raw.get(it.category, 0.0) + it.penalty

    capped = {cat: min(p, caps.get(cat, 0.9)) for cat, p in raw.items()}

    subscores = {cat: 1.0 for cat in weights}
    for cat, p in capped.items():
        subscores[cat] = max(0.0, 1.0 - p)

    total_w = sum(weights.values())
    score = sum(weights.get(c, 0) * subscores.get(c, 1.0) for c in weights) / total_w

    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "subscores": subscores,
        "category_penalties": capped,
    }


# ── Exclusion filter ───────────────────────────────────────────────────

def _apply_exclusions(issues: list[Issue], diff_cfg: dict[str, Any]) -> list[Issue]:
    regexes = [re.compile(r) for r in diff_cfg.get("exclude_path_regexes", []) if r]
    if not regexes:
        return issues
    return [it for it in issues if not any(r.search(it.path) for r in regexes)]


# ── Section alignment (added 2026-04-19, Bernardo Chalita) ────────────
#
# WHY THIS EXISTS:
#   BMW repair orders contain multiple sections (ASI, BWO, CSI, JSI, etc.),
#   each identified by a "prefix" field. The original eval compared sections
#   by their array index — sections[0] vs sections[0] — which caused:
#
#   1. ORDER SENSITIVITY: If the prediction returned [BWO, ASI] instead of
#      [ASI, BWO], every field was flagged as wrong even though the data
#      was correct, just reordered.
#
#   2. MISSING SECTION BLINDNESS: If GT had 4 sections and prediction had 1,
#      only 1 section got compared (index 0). The other 3 were recorded as
#      a single "missing list item" with a trivial 0.08 penalty each,
#      regardless of how many fields (40-60) each section contained.
#      Result: a prediction missing 75% of sections could score >0.90.
#
# HOW IT WORKS:
#   Before running DeepDiff, we align sections by prefix:
#   - GT sections [ASI, BWO, CSI, JSI] + Pred sections [BWO]
#   - Matched: BWO ↔ BWO (compared field-by-field)
#   - Missing from pred: ASI, CSI, JSI → replaced with {} in pred copy
#     so DeepDiff discovers every field in those GT sections as "missing"
#   - Extra in pred (not in GT): replaced with {} in GT copy
#     so DeepDiff discovers every field as "extra"
#
# IMPACT ON SCORES:
#   A prediction missing one section (with ~40 fields) now incurs ~40
#   individual missing-field penalties across structure/numbers/text
#   categories. This is intentional — the previous behavior of charging
#   0.08 for an entire missing section was a scoring bug.
#
# DISABLE:
#   Set config["section_alignment"]["enabled"] = False for legacy behavior.

def _leaf_fields(obj: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Recursively extract all non-null leaf (path, value) pairs from a nested structure.

    Used to enumerate every field in a GT section that's missing from the prediction,
    so each gets its own penalty instead of one flat penalty for the whole section.
    """
    if obj is None:
        return []
    if isinstance(obj, dict):
        results = []
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            results.extend(_leaf_fields(v, path))
        return results
    if isinstance(obj, list):
        results = []
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]"
            results.extend(_leaf_fields(v, path))
        return results
    # Leaf value (str, int, float, bool)
    return [(prefix, obj)]


def _issues_for_missing_section(
    gt_section: dict,
    section_index: int,
    rubric: dict[str, Any],
    missing_is_null: bool,
) -> list[Issue]:
    """Generate per-field "missing" issues for a GT section absent from prediction.

    Instead of one flat penalty for the whole section, we walk every leaf field
    in the GT section and emit a "missing" issue for each non-null value. This way
    a section with 40 fields incurs 40 penalties across structure/numbers/text,
    which accurately reflects the information loss.

    Additionally, emits a structure penalty for the section itself being absent,
    since a missing section is fundamentally a structural failure regardless of
    what data types it contained.
    """
    penalties_cfg = rubric["penalties"]
    thresholds = rubric["severity_thresholds"]
    issues: list[Issue] = []

    prefix_code = gt_section.get("prefix", "?")
    base_path = f"sections[{section_index}]"

    # Structural penalty for the entire section being missing.
    # This ensures "structure" subscore is properly penalized when sections
    # don't match, not just numbers/text.
    struct_pen = float(penalties_cfg.get("structure", {}).get("missing", 0.08))
    struct_sev = (
        "high" if struct_pen >= thresholds.get("high", 0.07)
        else "med" if struct_pen >= thresholds.get("med", 0.04)
        else "low"
    )
    issues.append(Issue(
        "structure", "missing", base_path, prefix_code, None,
        f"entire {prefix_code} section absent from prediction",
        struct_pen, struct_sev,
    ))

    for field_path, value in _leaf_fields(gt_section):
        if missing_is_null and value is None:
            continue
        full_path = f"{base_path}.{field_path}"
        cat = _categorize(full_path, value, None, "missing")
        pen = float(penalties_cfg.get(cat, {}).get("missing", 0.03))
        sev = (
            "high" if pen >= thresholds.get("high", 0.07)
            else "med" if pen >= thresholds.get("med", 0.04)
            else "low"
        )
        detail = f"missing (entire {prefix_code} section absent from prediction)"
        issues.append(Issue(cat, "missing", full_path, value, None, detail, pen, sev))

    return issues


def _issues_for_extra_section(
    pred_section: dict,
    section_index: int,
    rubric: dict[str, Any],
    missing_is_null: bool,
) -> list[Issue]:
    """Generate per-field "extra" issues for a predicted section not in GT."""
    penalties_cfg = rubric["penalties"]
    thresholds = rubric["severity_thresholds"]
    issues: list[Issue] = []

    prefix_code = pred_section.get("prefix", "?")
    base_path = f"sections[{section_index}]"

    # Structural penalty for the extra section existing
    struct_pen = float(penalties_cfg.get("structure", {}).get("extra", 0.04))
    struct_sev = (
        "high" if struct_pen >= thresholds.get("high", 0.07)
        else "med" if struct_pen >= thresholds.get("med", 0.04)
        else "low"
    )
    issues.append(Issue(
        "structure", "extra", base_path, None, prefix_code,
        f"predicted {prefix_code} section not in ground truth",
        struct_pen, struct_sev,
    ))

    for field_path, value in _leaf_fields(pred_section):
        if missing_is_null and value is None:
            continue
        full_path = f"{base_path}.{field_path}"
        cat = _categorize(full_path, None, value, "extra")
        pen = float(penalties_cfg.get(cat, {}).get("extra", 0.03))
        sev = (
            "high" if pen >= thresholds.get("high", 0.07)
            else "med" if pen >= thresholds.get("med", 0.04)
            else "low"
        )
        detail = f"extra (predicted {prefix_code} section not in ground truth)"
        issues.append(Issue(cat, "extra", full_path, value, None, detail, pen, sev))

    return issues


def _align_and_compare_sections(
    gt_sections: list[dict],
    pred_sections: list[dict],
    match_key: str,
    rubric: dict[str, Any],
    diff_cfg: dict[str, Any],
    missing_is_null: bool,
) -> tuple[list[Issue], dict[str, Any]]:
    """Match sections by prefix, compare matched pairs, and penalize missing/extra.

    This replaces the naive positional comparison with prefix-based alignment.
    Instead of relying on DeepDiff for missing sections (which collapses a
    multi-field section into a single issue), we manually enumerate every leaf
    field and create individual penalties.

    Returns:
        issues: All issues from section comparison (matched + missing + extra)
        alignment_info: Summary of what was matched/missing/extra
    """
    # ── Index prediction sections by prefix ──
    pred_by_key: dict[str, dict] = {}
    pred_unmatched: list[dict] = []
    for s in pred_sections:
        key = s.get(match_key)
        if key and key not in pred_by_key:
            pred_by_key[key] = s
        elif key:
            pred_unmatched.append(s)  # duplicate prefix → extra
        else:
            pred_unmatched.append(s)  # no prefix → extra

    # ── Match and compare ──
    all_issues: list[Issue] = []
    matched_prefixes: list[str] = []
    missing_prefixes: list[str] = []

    for idx, gt_s in enumerate(gt_sections):
        key = gt_s.get(match_key, "")
        if key in pred_by_key:
            # MATCHED: compare this GT section vs its matching pred section
            # using DeepDiff for accurate field-by-field comparison.
            pred_s = pred_by_key.pop(key)
            matched_prefixes.append(key)

            dd = DeepDiff(
                gt_s, pred_s,
                view="tree",
                ignore_numeric_type_changes=diff_cfg.get("ignore_numeric_type_changes", True),
                significant_digits=diff_cfg.get("significant_digits", 2),
                verbose_level=diff_cfg.get("verbose_level", 2),
            )
            section_issues = _diff_to_issues(dd, rubric, diff_cfg, missing_is_null)
            # Prefix paths with sections[idx] for proper reporting
            for si in section_issues:
                si.path = f"sections[{idx}].{si.path}" if si.path else f"sections[{idx}]"
            all_issues.extend(section_issues)
        else:
            # MISSING: GT section has no match in prediction.
            # Enumerate every leaf field and create a "missing" issue for each.
            missing_prefixes.append(key)
            all_issues.extend(
                _issues_for_missing_section(gt_s, idx, rubric, missing_is_null)
            )

    # ── Extra sections in prediction (not in GT) ──
    extra_sections = list(pred_by_key.values()) + pred_unmatched
    extra_prefixes = [s.get(match_key, "<no prefix>") for s in extra_sections]
    for i, extra_s in enumerate(extra_sections):
        extra_idx = len(gt_sections) + i
        all_issues.extend(
            _issues_for_extra_section(extra_s, extra_idx, rubric, missing_is_null)
        )

    alignment_info = {
        "matched": matched_prefixes,
        "missing_from_prediction": missing_prefixes,
        "extra_in_prediction": extra_prefixes,
        "gt_section_count": len(gt_sections),
        "pred_section_count": len(pred_sections),
    }

    return all_issues, alignment_info


# ── Public API ──────────────────────────────────────────────────────────

def evaluate_extraction(
    ground_truth: dict[str, Any],
    prediction: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare extracted JSON to ground truth.

    Returns a report with:
      score      – aggregate 0‒1 quality score
      subscores  – per-category (structure, numbers, text)
      issues     – ranked list of individual discrepancies
      section_alignment – (new) details of how sections were matched
    """
    cfg = deepcopy(DEFAULT_CONFIG)
    if config:
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    rubric = cfg["rubric"]
    diff_cfg = cfg["diff"]

    norm_cfg = cfg["normalization"]
    missing_is_null = norm_cfg.get("missing_is_null", True)

    gt = _normalize(ground_truth, norm_cfg)
    pred = _normalize(prediction, norm_cfg)

    # ── Section alignment (2026-04-19, Bernardo Chalita) ─────────────────
    # Match sections by prefix before running DeepDiff to avoid positional
    # comparison bugs. See docstring at top of file for full explanation.
    #
    # Strategy: remove "sections" from both GT and pred, handle section
    # comparison separately via _align_and_compare_sections(), then run
    # DeepDiff on the remaining top-level fields (doc_id, etc.) only.
    sa_cfg = cfg.get("section_alignment", {})
    alignment_info = None
    section_issues: list[Issue] = []

    if sa_cfg.get("enabled", True):
        gt_sections = gt.get("sections", [])
        pred_sections = pred.get("sections", [])

        if isinstance(gt_sections, list) and isinstance(pred_sections, list):
            match_key = sa_cfg.get("match_key", "prefix")
            section_issues, alignment_info = _align_and_compare_sections(
                gt_sections, pred_sections, match_key,
                rubric, diff_cfg, missing_is_null,
            )
            # Remove sections from the dicts so the main DeepDiff only
            # compares top-level fields (doc_id, etc.), avoiding double-counting.
            gt = {k: v for k, v in gt.items() if k != "sections"}
            pred = {k: v for k, v in pred.items() if k != "sections"}

        elif isinstance(gt_sections, list) and not isinstance(pred_sections, list):
            # Prediction has no sections array (e.g., flat JSON output).
            # Treat every GT section as missing.
            for idx, gt_s in enumerate(gt_sections):
                section_issues.extend(
                    _issues_for_missing_section(gt_s, idx, rubric, missing_is_null)
                )
            alignment_info = {
                "matched": [],
                "missing_from_prediction": [
                    s.get("prefix", "?") for s in gt_sections
                ],
                "extra_in_prediction": [],
                "gt_section_count": len(gt_sections),
                "pred_section_count": 0,
                "note": "prediction has no 'sections' list",
            }
            # Remove sections from GT to avoid double-counting in DeepDiff.
            # Keep pred as-is so DeepDiff catches the missing "sections" key.
            gt = {k: v for k, v in gt.items() if k != "sections"}
    # ── End section alignment ───────────────────────────────────────────

    # Run DeepDiff on remaining fields (or full objects if alignment disabled)
    dd = DeepDiff(
        gt,
        pred,
        view="tree",
        ignore_numeric_type_changes=diff_cfg["ignore_numeric_type_changes"],
        significant_digits=diff_cfg["significant_digits"],
        verbose_level=diff_cfg["verbose_level"],
        exclude_paths=diff_cfg.get("exclude_paths", []),
    )

    issues = _diff_to_issues(dd, rubric, diff_cfg, missing_is_null)

    # Combine section issues with top-level issues
    issues = section_issues + issues
    issues = _apply_exclusions(issues, diff_cfg)

    scoring = _score(issues, rubric)

    issues.sort(key=lambda x: (-x.penalty, x.category, x.path))
    max_issues = rubric.get("max_issues", 200)

    by_cat: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for it in issues:
        by_cat[it.category] = by_cat.get(it.category, 0) + 1
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1

    result = {
        **scoring,
        "counts_by_category": by_cat,
        "counts_by_kind": by_kind,
        "issues": [asdict(i) for i in issues[:max_issues]],
    }

    # Include section alignment info so the reflection LLM (and humans)
    # can see exactly which sections were matched, missing, or extra.
    if alignment_info is not None:
        result["section_alignment"] = alignment_info

    return result


# ── CLI ─────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate extracted JSON vs ground truth with scored feedback.",
    )
    ap.add_argument("--gt", required=True, help="Path to ground-truth JSON.")
    ap.add_argument("--pred", required=True, help="Path to predicted/extracted JSON.")
    ap.add_argument("--config", default=None, help="Optional config JSON to merge into defaults.")
    ap.add_argument("--out", default=None, help="Write report to path (else stdout).")
    args = ap.parse_args(argv)

    with open(args.gt) as f:
        gt = json.load(f)
    with open(args.pred) as f:
        pred = json.load(f)

    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    report = evaluate_extraction(gt, pred, cfg)

    out = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)

    if report["score"] < 0.25:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
