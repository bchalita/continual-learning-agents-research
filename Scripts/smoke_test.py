#!/usr/bin/env python3
"""
Smoke tests for eval.py, batch_eval.py, and merge.py.

Verifies the section alignment fix, batch runner, and merge node work
correctly without needing any API calls or pipeline execution.

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
from pipeline.merge import merge_sections, _deterministic_merge
from pipeline.gepa import PromptPool, PromptCandidate


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


def test_merge_single_section():
    """Merge with <=1 section should be a no-op."""
    print("Test: merge single section (no-op) ...")
    doc = {"doc_id": "test", "sections": [{"prefix": "ASI", "header": {"ro_number": 123}}]}
    result = merge_sections(doc)
    status = PASS if result == doc else FAIL
    print(f"  single section passthrough: {status}")
    assert result == doc, "Single section should pass through unchanged"


def test_merge_consistent_headers():
    """Merge with consistent headers should keep them unchanged."""
    print("Test: merge consistent headers ...")
    doc = {
        "doc_id": "test",
        "sections": [
            {"prefix": "ASI", "header": {"ro_number": 344098, "vin": "ABC123", "customer_name": "John"}},
            {"prefix": "BWO", "header": {"ro_number": 344098, "vin": "ABC123", "customer_name": "John"}},
            {"prefix": "CSI", "header": {"ro_number": 344098, "vin": "ABC123", "customer_name": "John"}},
        ],
    }
    result = _deterministic_merge(doc)
    for sec in result["sections"]:
        h = sec["header"]
        status = PASS if h["ro_number"] == 344098 and h["vin"] == "ABC123" else FAIL
        print(f"  {sec['prefix']}: ro={h['ro_number']}, vin={h['vin']} {status}")
        assert h["ro_number"] == 344098
        assert h["vin"] == "ABC123"


def test_merge_conflicting_headers():
    """Merge should pick majority value when headers conflict."""
    print("Test: merge conflicting headers (majority vote) ...")
    doc = {
        "doc_id": "test",
        "sections": [
            {"prefix": "ASI", "header": {"ro_number": 344098, "vin": "ABC123"}},
            {"prefix": "BWO", "header": {"ro_number": 344098, "vin": "ABC123"}},
            {"prefix": "CSI", "header": {"ro_number": 344099, "vin": "ABC123"}},  # misread
        ],
    }
    result = _deterministic_merge(doc)
    # Majority vote: 344098 appears twice, 344099 once → 344098 wins
    for sec in result["sections"]:
        h = sec["header"]
        status = PASS if h["ro_number"] == 344098 else FAIL
        print(f"  {sec['prefix']}: ro={h['ro_number']} {status}")
        assert h["ro_number"] == 344098, f"Expected 344098, got {h['ro_number']}"


def test_merge_null_fill():
    """Merge should fill null values from other sections."""
    print("Test: merge null fill ...")
    doc = {
        "doc_id": "test",
        "sections": [
            {"prefix": "ASI", "header": {"ro_number": 344098, "customer_name": None}},
            {"prefix": "BWO", "header": {"ro_number": 344098, "customer_name": "John Doe"}},
        ],
    }
    result = _deterministic_merge(doc)
    for sec in result["sections"]:
        h = sec["header"]
        status = PASS if h["customer_name"] == "John Doe" else FAIL
        print(f"  {sec['prefix']}: customer={h['customer_name']} {status}")
        assert h["customer_name"] == "John Doe", f"Expected 'John Doe', got {h['customer_name']}"


def test_merge_preserves_content():
    """Merge should NOT touch section-specific content, only headers."""
    print("Test: merge preserves section-specific content ...")
    doc = {
        "doc_id": "test",
        "sections": [
            {
                "prefix": "ASI",
                "header": {"ro_number": 344098},
                "content": {"job": [{"desc": "oil change"}]},
                "footer": {"total_charges": 150.00},
            },
            {
                "prefix": "BWO",
                "header": {"ro_number": 344098},
                "content": {"job": [{"desc": "tire rotation"}]},
                "footer": {"total_charges": 200.00},
            },
        ],
    }
    result = _deterministic_merge(doc)
    asi = result["sections"][0]
    bwo = result["sections"][1]
    ok1 = asi["content"]["job"][0]["desc"] == "oil change"
    ok2 = bwo["content"]["job"][0]["desc"] == "tire rotation"
    ok3 = asi["footer"]["total_charges"] == 150.00
    ok4 = bwo["footer"]["total_charges"] == 200.00
    status = PASS if all([ok1, ok2, ok3, ok4]) else FAIL
    print(f"  ASI content preserved: {ok1}, BWO content preserved: {ok2} {status}")
    assert all([ok1, ok2, ok3, ok4]), "Section-specific content should not be modified"


def test_merge_skips_parse_errors():
    """Sections with _parse_error should be kept but not used for reconciliation."""
    print("Test: merge skips parse error sections ...")
    doc = {
        "doc_id": "test",
        "sections": [
            {"prefix": "ASI", "header": {"ro_number": 344098}},
            {"_parse_error": True, "raw": "garbage"},
            {"prefix": "CSI", "header": {"ro_number": 344098}},
        ],
    }
    result = _deterministic_merge(doc)
    # Error section should still be present
    error_secs = [s for s in result["sections"] if s.get("_parse_error")]
    status = PASS if len(error_secs) == 1 else FAIL
    print(f"  error sections preserved: {len(error_secs)} {status}")
    assert len(error_secs) == 1


def test_merge_on_real_gt():
    """Deterministic merge on real GT data should not change scores."""
    print("Test: merge on real GT (should be idempotent) ...")
    for doc_id in ["201414", "344098", "944962"]:
        gt = load_gt(doc_id)
        merged = _deterministic_merge(gt)
        report = evaluate_extraction(gt, merged)
        score = report["score"]
        status = PASS if score == 1.0 else FAIL
        print(f"  {doc_id}: merged score={score:.4f} {status}")
        assert score == 1.0, f"Merge should not degrade GT, got {score}"


def test_gepa_pareto_dominance():
    """Pareto dominance: A dominates B iff A >= B on all docs and A > B on at least one."""
    print("Test: GEPA Pareto dominance ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.8, "doc3": 0.7}

    b = pool.add("prompt B", iteration=2)
    b.scores = {"doc1": 0.85, "doc2": 0.75, "doc3": 0.65}

    # A dominates B (better on all docs)
    ok1 = a.dominates(b)
    ok2 = not b.dominates(a)
    status = PASS if ok1 and ok2 else FAIL
    print(f"  A dominates B: {ok1}, B dominates A: {not ok2} {status}")
    assert ok1, "A should dominate B"
    assert ok2, "B should not dominate A"


def test_gepa_pareto_non_dominated():
    """Two prompts that each win on different docs → both on Pareto front."""
    print("Test: GEPA Pareto non-dominated ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.7}

    b = pool.add("prompt B", iteration=2)
    b.scores = {"doc1": 0.7, "doc2": 0.9}

    # Neither dominates the other → both on front
    front = pool.pareto_front
    front_ids = {c.prompt_id for c in front}
    ok = front_ids == {a.prompt_id, b.prompt_id}
    status = PASS if ok else FAIL
    print(f"  Front: {front_ids} {status}")
    assert ok, f"Both should be on front, got {front_ids}"


def test_gepa_win_frequency():
    """Win frequency: count how many docs each prompt wins on."""
    print("Test: GEPA win frequency ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.8, "doc3": 0.6}

    b = pool.add("prompt B", iteration=2)
    b.scores = {"doc1": 0.7, "doc2": 0.85, "doc3": 0.9}

    wins = pool.win_frequencies()
    # A wins doc1, B wins doc2 and doc3
    ok1 = wins[a.prompt_id] == 1
    ok2 = wins[b.prompt_id] == 2
    status = PASS if ok1 and ok2 else FAIL
    print(f"  A wins={wins[a.prompt_id]}, B wins={wins[b.prompt_id]} {status}")
    assert ok1, f"A should win 1 doc, got {wins[a.prompt_id]}"
    assert ok2, f"B should win 2 docs, got {wins[b.prompt_id]}"


def test_gepa_worst_doc():
    """Worst doc: the document where a prompt scores lowest."""
    print("Test: GEPA worst doc identification ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.5, "doc3": 0.7}

    worst = a.worst_doc()
    ok = worst == "doc2"
    status = PASS if ok else FAIL
    print(f"  Worst doc: {worst} {status}")
    assert ok, f"Worst doc should be doc2, got {worst}"


def test_gepa_select_parent():
    """Select parent should return a prompt from the Pareto front."""
    print("Test: GEPA parent selection ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.8}

    b = pool.add("prompt B", iteration=2)
    b.scores = {"doc1": 0.7, "doc2": 0.9}

    front_ids = {c.prompt_id for c in pool.pareto_front}
    # Select 10 times — all should be from the front
    selections = {pool.select_parent().prompt_id for _ in range(20)}
    ok = selections.issubset(front_ids)
    status = PASS if ok else FAIL
    print(f"  All selections from front: {ok} {status}")
    assert ok, f"Selections {selections} should be subset of front {front_ids}"


def test_gepa_pool_serialization():
    """Pool should serialize to JSON and contain summary stats."""
    print("Test: GEPA pool serialization ...")
    pool = PromptPool()

    a = pool.add("prompt A", iteration=1)
    a.scores = {"doc1": 0.9, "doc2": 0.8}

    pool.add("prompt B (unscored)", iteration=2)

    data = json.loads(pool.to_json())
    ok1 = len(data["candidates"]) == 2
    ok2 = data["summary"]["pool_size"] == 2
    ok3 = data["summary"]["scored"] == 1
    ok4 = data["summary"]["best_prompt_id"] == a.prompt_id
    status = PASS if all([ok1, ok2, ok3, ok4]) else FAIL
    print(f"  candidates={len(data['candidates'])}, scored={data['summary']['scored']} {status}")
    assert all([ok1, ok2, ok3, ok4])


def main():
    print("=" * 60)
    print("SMOKE TESTS — eval.py + batch_eval.py + merge.py + gepa.py")
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
        test_merge_single_section,
        test_merge_consistent_headers,
        test_merge_conflicting_headers,
        test_merge_null_fill,
        test_merge_preserves_content,
        test_merge_skips_parse_errors,
        test_merge_on_real_gt,
        test_gepa_pareto_dominance,
        test_gepa_pareto_non_dominated,
        test_gepa_win_frequency,
        test_gepa_worst_doc,
        test_gepa_select_parent,
        test_gepa_pool_serialization,
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
