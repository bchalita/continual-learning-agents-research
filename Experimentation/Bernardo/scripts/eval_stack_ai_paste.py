#!/usr/bin/env python3
"""
PASTE THIS ENTIRE FILE INTO THE STACK AI PYTHON NODE.

Inputs (connect these nodes to the Python node):
  - llm_0   = LLM Extractor output (extraction JSON, possibly with ```json fences)
  - in_1    = ground_truth (reference JSON)

Output:  result  = report dict with score, subscores, issues (for Evaluator / Reflection).

Uses deepdiff if available; otherwise a no-dependency recursive comparison (stdlib only).
"""

# Stack AI may block: Import, ImportFrom, AnnAssign, Lambda, Raise, type annotations.
# Use __import__(); no annotations; no dataclass; no lambda; no raise (return error instead).
json = __import__("json")
re = __import__("re")
copy = __import__("copy")

try:
    deepdiff = __import__("deepdiff")
    DeepDiff = deepdiff.DeepDiff
    _has_deepdiff = True
except Exception:
    DeepDiff = None
    _has_deepdiff = False


# ── Stack AI: parse inputs llm_0 and in_1 ─────────────────────────────────

def _extract_json(text):
    """Parse JSON from string; strip ```json ... ``` if present. Returns (data, None) or (None, error_msg). No raise."""
    if isinstance(text, dict):
        return (text, None)
    if not isinstance(text, str):
        return (None, "Expected str or dict for extraction input")
    s = text.strip()
    m = re.search(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        return (json.loads(s), None)
    except Exception as e:
        return (None, str(e))


def _get_stack_ai_inputs():
    """Read prediction (llm_0) and ground truth (in_1). Returns (gt, pred, None) or (None, None, error_msg). No raise."""
    g = globals()
    llm_0 = g.get("llm_0") or g.get("llm-0")
    in_1 = g.get("in_1") or g.get("in-1")
    if llm_0 is None:
        return (None, None, "llm_0 not set. Connect the LLM Extractor node to this Python node.")
    if in_1 is None:
        return (None, None, "in_1 not set. Connect the ground_truth input node to this Python node.")
    # If Stack AI passes the full Extractor output object, use the completion string.
    if isinstance(llm_0, dict) and "completion" in llm_0:
        llm_0 = llm_0["completion"]
    ground_truth, err1 = _extract_json(in_1)
    if err1 is not None:
        return (None, None, err1)
    prediction, err2 = _extract_json(llm_0)
    if err2 is not None:
        return (None, None, err2)
    return (ground_truth, prediction, None)


# ── Full evaluation logic (from eval.py) ──────────────────────────────────

DEFAULT_CONFIG = {
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
    "rubric": {
        "category_weights": {"structure": 0.45, "numbers": 0.40, "text": 0.15},
        "penalties": {
            "structure": {"missing": 0.08, "extra": 0.04, "type": 0.06, "value": 0.03},
            "numbers": {"missing": 0.08, "extra": 0.02, "type": 0.06, "value": 0.06},
            "text": {"missing": 0.05, "extra": 0.02, "type": 0.03, "value": 0.04},
        },
        "category_caps": {"structure": 0.90, "numbers": 0.90, "text": 0.80},
        "max_issues": 200,
        "severity_thresholds": {"high": 0.07, "med": 0.04, "low": 0.0},
    },
}


def _issue_dict(category, kind, path, expected=None, got=None, detail="", penalty=0.0, severity="low"):
    return {"category": category, "kind": kind, "path": path, "expected": expected, "got": got, "detail": detail, "penalty": penalty, "severity": severity}


def _humanize_path(path):
    if not path.startswith("root"):
        return path
    p = path[4:]
    p = re.sub(r"\['([^']+)'\]", r".\1", p)
    return p.lstrip(".")


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _categorize(path, expected, got, kind):
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


def _normalize(obj, cfg):
    strip = cfg.get("strip_strings", True)
    collapse = cfg.get("collapse_whitespace", True)
    empty_is_null = cfg.get("empty_is_null", True)

    def walk(x):
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


def _diff_to_issues(dd, rubric, diff_cfg=None, missing_is_null=True, _depth=0):
    MAX_DECOMPOSE_DEPTH = 4
    penalties_cfg = rubric["penalties"]
    thresholds = rubric["severity_thresholds"]
    issues = []

    def _penalty(cat, kind):
        cat_tbl = penalties_cfg.get(cat, penalties_cfg.get("text", {}))
        return float(cat_tbl.get(kind, 0.03))

    def _severity(p):
        if p >= thresholds.get("high", 0.07):
            return "high"
        if p >= thresholds.get("med", 0.04):
            return "med"
        return "low"

    def emit(kind, path, exp, got, detail=""):
        cat = _categorize(path, exp, got, kind)
        pen = _penalty(cat, kind)
        issues.append(_issue_dict(cat, kind, path, exp, got, detail, pen, _severity(pen)))

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
                ignore_numeric_type_changes=diff_cfg.get("ignore_numeric_type_changes", True) if diff_cfg else True,
                significant_digits=diff_cfg.get("significant_digits", 2) if diff_cfg else 2,
                verbose_level=2,
            )
            sub_issues = _diff_to_issues(sub_dd, rubric, diff_cfg, missing_is_null, _depth + 1)
            for si in sub_issues:
                si["path"] = (path + "." + si["path"]) if si["path"] else path
            if sub_issues:
                issues.extend(sub_issues)
                continue
        emit("value", path, item.t1, item.t2)

    for item in dd.get("type_changes", []):
        detail = "{} \u2192 {}".format(type(item.t1).__name__, type(item.t2).__name__)
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


def _score(issues, rubric):
    caps = rubric["category_caps"]
    weights = rubric["category_weights"]
    raw = {}
    for it in issues:
        raw[it["category"]] = raw.get(it["category"], 0.0) + it["penalty"]
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


def _apply_exclusions(issues, diff_cfg):
    regexes = [re.compile(r) for r in diff_cfg.get("exclude_path_regexes", []) if r]
    if not regexes:
        return issues
    return [it for it in issues if not any(r.search(it["path"]) for r in regexes)]


# ── No-dependency fallback: recursive compare (no deepdiff) ─────────────────

def _path_join(base, key):
    if base == "root" or base == "":
        return "root[" + repr(key) + "]"
    if isinstance(key, int):
        return base + "[" + str(key) + "]"
    return base + "[" + repr(key) + "]"


def _simple_compare(gt, pred, path, issues, rubric, missing_is_null, _depth, max_depth=20):
    if _depth > max_depth:
        return
    penalties_cfg = rubric["penalties"]
    thresholds = rubric["severity_thresholds"]

    def emit(kind, p, exp, got, detail=""):
        p_flat = _humanize_path(p)
        cat = _categorize(p_flat, exp, got, kind)
        cat_tbl = penalties_cfg.get(cat, penalties_cfg.get("text", {}))
        pen = float(cat_tbl.get(kind, 0.03))
        sev = "high" if pen >= thresholds.get("high", 0.07) else ("med" if pen >= thresholds.get("med", 0.04) else "low")
        issues.append(_issue_dict(cat, kind, p_flat, exp, got, detail, pen, sev))

    if isinstance(gt, dict) and isinstance(pred, dict):
        all_keys = set(gt.keys()) | set(pred.keys())
        for k in all_keys:
            p = _path_join(path, k)
            if k not in pred:
                if missing_is_null and gt.get(k) is None:
                    continue
                emit("missing", p, gt.get(k), None, "missing key")
            elif k not in gt:
                if missing_is_null and pred.get(k) is None:
                    continue
                emit("extra", p, None, pred.get(k), "extra key")
            else:
                _simple_compare(gt[k], pred[k], p, issues, rubric, missing_is_null, _depth + 1, max_depth)
        return
    if isinstance(gt, list) and isinstance(pred, list):
        for i in range(max(len(gt), len(pred))):
            p = _path_join(path, i)
            if i >= len(gt):
                emit("extra", p, None, pred[i] if i < len(pred) else None, "extra list item")
            elif i >= len(pred):
                emit("missing", p, gt[i], None, "missing list item")
            else:
                _simple_compare(gt[i], pred[i], p, issues, rubric, missing_is_null, _depth + 1, max_depth)
        return
    if type(gt) is not type(pred):
        detail = "{} \u2192 {}".format(type(gt).__name__, type(pred).__name__)
        emit("type", path, gt, pred, detail)
        return
    if gt != pred:
        if _is_number(gt) or _is_number(pred):
            if gt != pred:
                emit("value", path, gt, pred)
        else:
            emit("value", path, gt, pred)


def evaluate_extraction_fallback(ground_truth, prediction, config=None):
    """Same output as evaluate_extraction but uses recursive compare (no deepdiff)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
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
    issues = []
    _simple_compare(gt, pred, "root", issues, rubric, missing_is_null, 0)
    issues = _apply_exclusions(issues, diff_cfg)
    scoring = _score(issues, rubric)
    def _issue_sort_key(it):
        return (-it["penalty"], it["category"], it["path"])
    issues.sort(key=_issue_sort_key)
    max_issues = rubric.get("max_issues", 200)
    by_cat = {}
    by_kind = {}
    for it in issues:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    return {
        **scoring,
        "counts_by_category": by_cat,
        "counts_by_kind": by_kind,
        "issues": issues[:max_issues],
    }


def evaluate_extraction(ground_truth, prediction, config=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
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

    dd = DeepDiff(
        gt, pred,
        view="tree",
        ignore_numeric_type_changes=diff_cfg["ignore_numeric_type_changes"],
        significant_digits=diff_cfg["significant_digits"],
        verbose_level=diff_cfg["verbose_level"],
        exclude_paths=diff_cfg.get("exclude_paths", []),
    )
    issues = _diff_to_issues(dd, rubric, diff_cfg, missing_is_null)
    issues = _apply_exclusions(issues, diff_cfg)
    scoring = _score(issues, rubric)
    def _issue_sort_key(it):
        return (-it["penalty"], it["category"], it["path"])
    issues.sort(key=_issue_sort_key)
    max_issues = rubric.get("max_issues", 200)
    by_cat = {}
    by_kind = {}
    for it in issues:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    return {
        **scoring,
        "counts_by_category": by_cat,
        "counts_by_kind": by_kind,
        "issues": issues[:max_issues],
    }


# ── Run in Stack AI: get llm_0, in_1 → evaluate → result ──────────────────
# In Stack AI, set this node's OUTPUT to the variable name you use below (e.g. "result" or "output").

result = None
try:
    _gt, _pred, _err = _get_stack_ai_inputs()
    if _err is not None:
        result = {
            "score": 0.0,
            "subscores": {"structure": 0.0, "numbers": 0.0, "text": 0.0},
            "issues": [{"path": "", "kind": "error", "detail": _err}],
            "counts_by_category": {},
            "counts_by_kind": {},
        }
    else:
        if _has_deepdiff:
            result = evaluate_extraction(_gt, _pred, None)
        else:
            result = evaluate_extraction_fallback(_gt, _pred, None)
except Exception as e:
    result = {
        "score": 0.0,
        "subscores": {"structure": 0.0, "numbers": 0.0, "text": 0.0},
        "issues": [{"path": "", "kind": "error", "detail": str(e)}],
        "counts_by_category": {},
        "counts_by_kind": {},
    }

if result is None:
    result = {"score": 0.0, "subscores": {"structure": 0.0, "numbers": 0.0, "text": 0.0}, "issues": [{"path": "", "kind": "error", "detail": "No result produced"}], "counts_by_category": {}, "counts_by_kind": {}}

output = result
