# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BMW Group × MIT GenAI Lab research project. Goal: build a closed-loop system that improves BMW repair-order interpretation agents **without retraining any model** — by automatically evolving the system prompt using evaluation feedback (GEPA-style: Evaluate → Modify Prompt → Re-test → Select Improvement).

All code runs locally via direct API calls. No model weight updates — prompt-only optimization.

## Key Commands

### Evaluation (authoritative scorer — BMW will use this)
```bash
# Compare predicted JSON vs ground truth
python Scripts/eval.py --gt Data/Samples/201414.json --pred <prediction.json>
python Scripts/eval.py --gt gt.json --pred pred.json --out report.json
```

Requires: `pip install deepdiff`

## Architecture

### Pipeline Flow
```
PDF (repair order)
  → [FIXED] Document Encoding (vision model or OCR — never changes)
  → [TRAINABLE] LLM + System Prompt → structured JSON
  → [FIXED] eval.py (score 0–1 + per-field issues = mu_f feedback)
  → [OPTIMIZER] GEPA-style reflection LLM (current prompt + trace + score + issues → new prompt)
  → repeat
```

The **only thing that changes** between baseline and optimized runs is the system prompt. The delta (before vs after) is the proof of concept.

### Repository Layout
- `Data/Samples/` — 6 BMW repair-order document pairs: `{id}.pdf` (image-based, multi-section) + `{id}.json` (ground-truth structured JSON)
- `Scripts/eval.py` — authoritative JSON scorer (BMW-provided spec). Public API: `evaluate_extraction(gt, pred, config=None) -> dict`. Returns `{score, subscores, issues}`.
- `CONTEXT.md` — detailed research context, data schema, paper summaries, and open tasks

### Eval Scoring Rubric
| Category  | Weight | Penalty sources |
|-----------|--------|----------------|
| structure | 45%    | missing/extra keys, wrong types |
| numbers   | 40%    | numeric field accuracy (RO#, VIN, amounts) |
| text      | 15%    | free-text fields |

Returns: `{score: 0–1, subscores: {structure, numbers, text}, issues: [{category, kind, path, expected, got, penalty, severity}]}`

### Ground Truth JSON Schema
```json
{
  "doc_id": "201414",
  "sections": [{
    "section_id": "ASI-201414",
    "prefix": "ASI",          // ASI, BWO, CSI, JSI, WSI, ISI
    "page_count": 1,
    "header": { "ro_number", "vin", "customer_name", "vehicle", "dates", ... },
    "footer": { "labor_amount", "parts_amount", "total_charges", ... },
    "content": { "job": [...], "labor": [...], "acct_split": [...], ... }
  }]
}
```

### API Keys (set as environment variables — never commit)
- `OPENAI_API_KEY` — for extraction and reflection via OpenAI

### Key Papers
- **GEPA** (ICLR 2026) — primary approach. Reflective prompt evolution via natural-language feedback on execution traces. 35× fewer rollouts than RL.
- **Trace** (NeurIPS 2024, Microsoft) — PyTorch-like computation graph for agents; prompt as trainable node.
- **DSPy + MIPROv2** — Bayesian prompt optimization; comparison approach.
