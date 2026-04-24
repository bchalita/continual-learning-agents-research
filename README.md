# Continual Learning Agents — BMW × MIT GenAI Lab

Closed-loop system that improves AI agents **without retraining models**. The system prompt is treated as an optimizable policy: evaluation feedback drives automatic prompt improvement each iteration (GEPA-style).

**Loop:** Extract → Evaluate → Reflect → Repeat

---

## Quick Start — Run the Demo Locally

```bash
# 1. Clone the repo (this branch)
git clone -b bernardo/improved-pipeline https://github.com/bchalita/continual-learning-agents-research.git
cd continual-learning-agents-research

# 2. Install dependencies (including Streamlit)
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open in your browser (usually `http://localhost:8501`). Paste your **Anthropic API key** directly into the sidebar — no `.env` file needed. You can get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

Pick a document and number of iterations in the sidebar, then click **▶ Run Pipeline**. The app shows every step in real time: PDF thumbnails → structure analysis → per-section extraction → eval scores → prompt diff → score chart.

> **Note:** If `streamlit` is not on your PATH after installing, run `python -m streamlit run app.py` instead.

---

## CLI Usage

```bash
# Single document — extract + evaluate (no optimization)
python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json

# Single document — run the optimization loop (N iterations)
python run.py --doc Data/Samples/201414.pdf --gt Data/Samples/201414.json --iterations 4

# All 6 sample documents — eval only
python run.py --all

# All 6 sample documents — optimization loop
python run.py --all --iterations 3
```

Results are saved to `results/` (gitignored): prediction JSONs, eval reports, per-iteration prompts, and a CSV score log.

---

## How It Works

```
PDF
 ├─► Tool 1 (Claude Haiku) — structure analysis → section boundaries
 └─► Tool 2 (Claude Haiku) — JSON extraction per section  ◄── extraction_prompt
          │
          ▼
     eval.py (deterministic) + LLM Evaluator (Claude Haiku)
          │
          ▼
     Reflection (Claude Sonnet) ──► updated extraction_prompt
                                          │
                                          └─► next iteration ↺
```

- **Tool 1** identifies document sections from low-res thumbnails (72 DPI).
- **Tool 2** extracts structured JSON from full-res images (100 DPI) using the current `extraction_prompt` as its system prompt.
- **eval.py** scores the output against ground truth: structure 45%, numbers 40%, text 15%.
- **Reflection** reads the score, top issues, and LLM diagnosis, then proposes an improved `extraction_prompt` for the next iteration.

The only thing that changes between iterations is `extraction_prompt`. No model weights are updated.

---

## Repository Layout

```
pipeline/        Core modules (config, pdf_utils, tools, orchestrator, evaluator, reflection, loop)
prompts/         extraction.txt (optimized), structure.txt, reflection.txt (fixed)
Data/Samples/    6 BMW repair-order pairs: {id}.pdf + {id}.json (ground truth)
Scripts/eval.py  Authoritative deterministic scorer
app.py           Streamlit interactive demo
run.py           CLI entry point
results/         Auto-created; gitignored
```

---

## Models

| Role | Model |
|------|-------|
| Structure analysis (Tool 1) | `claude-haiku-4-5-20251001` |
| JSON extraction (Tool 2) | `claude-haiku-4-5-20251001` |
| LLM evaluator | `claude-haiku-4-5-20251001` |
| Reflection / prompt optimization | `claude-sonnet-4-6` |

---

## Research Context

- **GEPA** (ICLR 2026) — reflective prompt evolution via natural-language feedback on execution traces; 35× fewer rollouts than RL.
- **Trace** (NeurIPS 2024, Microsoft) — PyTorch-like computation graph for agents; treats the prompt as a trainable node.
- **DSPy + MIPROv2** — Bayesian prompt optimization; used as a comparison approach.
