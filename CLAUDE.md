# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BMW Group × MIT GenAI Lab research project. Goal: build a closed-loop system that improves BMW repair-order interpretation agents **without retraining any model** — by automatically evolving the system prompt using evaluation feedback (GEPA-style: Evaluate → Modify Prompt → Re-test → Select Improvement).

All code runs locally via direct Anthropic API calls. No model weight updates — prompt-only optimization.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Interactive App

```bash
streamlit run app.py
```

Opens a browser UI with real-time step visualization: PDF thumbnails → structure JSON → per-section extraction → eval metrics → prompt diffs → score chart.

## Key Commands (CLI)

```bash
# Single document — extract + eval (no optimization)
python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json

# Single document — optimization loop (N iterations)
python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json --iterations 4

# All 6 sample documents — eval only
python run.py --all

# All 6 sample documents — optimization loop
python run.py --all --iterations 3

# Evaluation script directly (compare two JSON files)
python Scripts/eval.py --gt Data/Samples/201414.json --pred results/201414_prediction.json
```

Requires: `pip install deepdiff` (included in requirements.txt)

## Pipeline Architecture

```
PDF (repair order)
  → pdf_utils.py: PDF → page images (PyMuPDF, 150 DPI)
  → tools.py Tool 1: analyze_structure → section boundaries
  → tools.py Tool 2: parse_section (per section, OPTIMIZABLE prompt) → section JSON
  → orchestrator.py: merge sections → full document JSON
  → evaluator.py: eval.py (deterministic) + LLM evaluator → score + feedback
  → reflection.py: GEPA-style reflection → proposed new prompts
  → loop.py: iterate N times, track scores in CSV
```

The **only thing that changes** between iterations is the extraction prompt (Tool 2 system prompt), proposed by the reflection LLM after each evaluation.

## Repository Layout

- `Data/Samples/` — 6 BMW repair-order document pairs: `{id}.pdf` + `{id}.json` (ground truth)
- `Scripts/eval.py` — authoritative JSON scorer (BMW-provided spec). API: `evaluate_extraction(gt, pred, config=None) -> dict`
- `pipeline/` — core modules (config, pdf_utils, tools, orchestrator, evaluator, reflection, loop)
- `prompts/` — system prompt files. `extraction.txt` is optimized each iteration; `structure.txt` and `reflection.txt` are fixed
- `results/` — auto-created; gitignored. Contains per-iteration predictions, eval reports, CSV, and saved prompts
- `run.py` — CLI entry point
- `CONTEXT.md` — detailed research context, data schema, paper summaries

## Eval Scoring Rubric

| Category  | Weight | What it covers |
|-----------|--------|----------------|
| structure | 45%    | missing/extra keys, wrong types |
| numbers   | 40%    | numeric field accuracy (RO#, VIN, amounts) |
| text      | 15%    | free-text fields |

Returns: `{score: 0–1, subscores: {structure, numbers, text}, issues: [{category, kind, path, expected, got, penalty, severity}]}`

## Ground Truth JSON Schema

```json
{
  "doc_id": "201414",
  "sections": [{
    "section_id": "ASI-201414",
    "prefix": "ASI",          // ASI, BWO, CSI, JSI, WSI, ISI
    "page_count": 1,
    "header": { "ro_number", "vin", "customer_name", "vehicle fields", "dates", ... },
    "footer": { "labor_amount", "parts_amount", "total_charges", ... },
    "content": { "job": [...], "labor": [...], "acct_split": [...], ... }
  }]
}
```

## Models

- `claude-haiku-4-5-20251001` — Tool 1 (structure) + Tool 2 (extraction) + LLM evaluator — cost-efficient for many iterations
- `claude-sonnet-4-6` — Reflection — stronger reasoning for prompt improvement

## API Key

Set `ANTHROPIC_API_KEY` in `.env`. Never commit it.

## Key Papers

- **GEPA** (ICLR 2026) — primary approach. Reflective prompt evolution via natural-language feedback on execution traces. 35× fewer rollouts than RL.
- **Trace** (NeurIPS 2024, Microsoft) — PyTorch-like computation graph for agents; prompt as trainable node.
- **DSPy + MIPROv2** — Bayesian prompt optimization; comparison approach.
