#!/usr/bin/env python3
"""
Smoke tests for eval.py and batch_eval.py.

Verifies the section alignment fix and batch runner work correctly
without needing any API calls or pipeline execution.

Usage:
  python Scripts/smoke_test.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Scripts.eval import evaluate_extraction
from Scripts.batch_eval import batch_evaluate, print_report


SAMPLES_DIR = Path(__file__).resolve().parent.parent / "Data" / "Samples"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def load_gt(doc_id: str) -> dict:
    with open(SAMPLES_DIR / f"{doc_id}.json") as f:
        return json.load(f)


def test_self_comparison():
    """Every GT compared against itself should score 1.0."""
    print("Test: self-comparison (GT vs GT) ...")
    for doc_id in ["201414", "344098", "629903", "678856", "809570", "944962"]:
        gt = load_gt(doc_id)
        report = evaluate_extraction(gt, gt)
        score = report["score"]
        status = PASS if score == 1.0 else FAIL
        print(f"  {doc_id}: {score:.4f} {status}")
        assert score == 1.0, f"{doc_id} self-comparison should be 1.0, got {score}"


def test_reordered_sections():
    """Reversed section order should still score 1.0 (alignment fix)."""
    print("Test: reordered sections ...")
    for doc_id in ["201414", "344098", "944962"]:
        gt = load_gt(doc_id)
        pred = copy.deepcopy(gt)
        pred["sections"] = list(reversed(gt["sections"]))
        report = evaluate_extraction(gt, pred)
        score = report["score"]
        status = PASS if score == 1.0 else FAIL
        print(f"  {doc_id} (reversed): {score:.4f} {status}")
        assert score == 1.0, f"Reversed sections should score 1.0, got {score}"


def test_missing_section_penalty():
    """Missing one section should score significantly below 1.0."""
    print("Test: missing section penalty ...")
    gt = load_gt("344098")
    pred = copy.deepcopy(gt)
    dropped = pred["sections"].pop()  # drop last section
    report = evaluate_extraction(gt, pred)
    score = report["score"]
    sa = report.get("section_alignment", {})
    status = PASS if score < 0.8 else FAIL
    print(f"  344098 (dropped {dropped.get('prefix','?')}): {score:.4f} {status}")
    print(f"    alignment: matched={sa.get('matched')}, missing={sa.get('missing_from_prediction')}")
    assert score < 0.8, f"Missing section should penalize heavily, got {score}"


def test_flat_json_penalty():
    """A flat JSON with no proper sections should score poorly."""
    print("Test: flat JSON (no sections structure) ...")
    gt = load_gt("344098")
    flat_pred = {
        "invoice_type": "PRE-INVOICE",
        "invoice_number": "344098",
        "sections": [
            {"invoice_number": "344098", "customer_number": "996865"}
        ]
    }
    report = evaluate_extraction(gt, flat_pred)
    score = report["score"]
    sa = report.get("section_alignment", {})
    status = PASS if score < 0.7 else FAIL
    print(f"  flat prediction: {score:.4f} {status}")
    print(f"    alignment: missing={sa.get('missing_from_prediction')}, extra={sa.get('extra_in_prediction')}")
    assert score < 0.7, f"Flat JSON should score below 0.7, got {score}"


def test_legacy_mode():
    """Legacy mode (alignment disabled) should show the old buggy behavior."""
    print("Test: legacy mode (alignment disabled) ...")
    gt = load_gt("344098")
    pred = copy.deepcopy(gt)
    pred["sections"] = list(reversed(gt["sections"]))
    legacy_cfg = {"section_alignment": {"enabled": False}}

    report_new = evaluate_extraction(gt, pred)
    report_old = evaluate_extraction(gt, pred, config=legacy_cfg)

    new_score = report_new["score"]
    old_score = report_old["score"]
    # New should be 1.0 (aligned), old should be < 1.0 (positional mismatch)
    status = PASS if (new_score == 1.0 and old_score < 1.0) else FAIL
    print(f"  new={new_score:.4f}, legacy={old_score:.4f} {status}")
    assert new_score == 1.0
    assert old_score < 1.0


def test_extra_section_penalty():
    """Extra predicted sections not in GT should be penalized."""
    print("Test: extra section penalty ...")
    gt = load_gt("344098")
    pred = copy.deepcopy(gt)
    pred["sections"].append({
        "prefix": "FAKE",
        "section_id": "FAKE-344098",
        "header": {"ro_number": 344098, "customer_name": "Fake Person"},
    })
    report = evaluate_extraction(gt, pred)
    score = report["score"]
    sa = report.get("section_alignment", {})
    status = PASS if score < 1.0 and "FAKE" in sa.get("extra_in_prediction", []) else FAIL
    print(f"  with extra FAKE section: {score:.4f} {status}")
    print(f"    extra sections detected: {sa.get('extra_in_prediction')}")
    assert score < 1.0


def test_batch_runner():
    """Batch runner should produce correct aggregate results."""
    print("Test: batch runner ...")
    pairs = []
    for doc_id in ["201414", "344098", "629903"]:
        gt = load_gt(doc_id)
        pairs.append((gt, gt, doc_id))

    report = batch_evaluate(pairs)
    status = PASS if report.mean_score == 1.0 else FAIL
    print(f"  batch self-test (3 docs): mean={report.mean_score:.4f} {status}")
    assert report.mean_score == 1.0
    assert report.total_missing_sections == 0


def test_section_alignment_info():
    """Section alignment info should be present and accurate."""
    print("Test: section alignment info ...")
    gt = load_gt("201414")
    report = evaluate_extraction(gt, gt)
    sa = report.get("section_alignment", {})
    expected_prefixes = ["ASI", "BWO", "CSI", "JSI", "WSI"]
    status = PASS if sa.get("matched") == expected_prefixes else FAIL
    print(f"  201414 matched: {sa.get('matched')} {status}")
    assert sa.get("matched") == expected_prefixes
    assert sa.get("missing_from_prediction") == []
    assert sa.get("extra_in_prediction") == []


def main():
    print("=" * 60)
    print("SMOKE TESTS — eval.py + batch_eval.py")
    print("=" * 60)
    print()

    tests = [
        test_self_comparison,
        test_reordered_sections,
        test_missing_section_penalty,
        test_flat_json_penalty,
        test_legacy_mode,
        test_extra_section_penalty,
        test_batch_runner,
        test_section_alignment_info,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  {FAIL}: {e}")
            failed += 1
        except Exception as e:
            print(f"  {FAIL}: unexpected error: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
