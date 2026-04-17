"""
Streamlit interactive app for the BMW repair order extraction pipeline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from pipeline import config
from pipeline.evaluator import evaluate
from pipeline.pdf_utils import pdf_to_images
from pipeline.reflection import propose
from pipeline.tools import analyze_structure, parse_section

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMW RO Extraction Pipeline",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background: #1e1e2e; border-radius: 8px; padding: 12px; }
.section-badge { display: inline-block; background: #2d4a7a; color: white;
                 padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def score_emoji(score: float) -> str:
    if score >= 0.8:
        return "🟢"
    if score >= 0.5:
        return "🟡"
    return "🔴"


def load_samples() -> dict[str, tuple[Path, Path]]:
    samples = {}
    for pdf in sorted(config.SAMPLES_DIR.glob("*.pdf")):
        gt = pdf.with_suffix(".json")
        if gt.exists():
            samples[pdf.stem] = (pdf, gt)
    return samples


def png_to_pil(img_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(img_bytes))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Pipeline Config")
    st.divider()

    # API key status
    if not config.ANTHROPIC_API_KEY:
        st.error("**ANTHROPIC_API_KEY** not found.\nCopy `.env.example` → `.env` and add your key.")
        st.stop()
    st.success("API key loaded ✓")

    # Document selector
    samples = load_samples()
    if not samples:
        st.error(f"No PDF+JSON pairs found in `{config.SAMPLES_DIR}`")
        st.stop()

    doc_id = st.selectbox(
        "Document",
        list(samples.keys()),
        help="Select one of the 6 sample BMW repair orders",
    )
    pdf_path, gt_path = samples[doc_id]

    st.caption(f"PDF: `{pdf_path.name}`")
    st.caption(f"GT:  `{gt_path.name}`")
    st.divider()

    iterations = st.slider(
        "Optimization iterations",
        min_value=1,
        max_value=5,
        value=1,
        help="Each iteration: Extract → Evaluate → Reflect. Prompts evolve across iterations.",
    )

    st.divider()
    st.caption(f"Extraction model: `{config.EXTRACTION_MODEL}`")
    st.caption(f"Reflection model: `{config.REFLECTION_MODEL}`")
    st.caption(f"Extraction DPI: `{config.PDF_DPI}` · Structure DPI: `{config.STRUCTURE_DPI}`")
    st.divider()
    st.caption("💡 Prompt caching enabled — repeated section calls reuse cached system prompts at ~10% cost.")

    run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔧 BMW Repair Order Extraction Pipeline")
st.caption(
    f"Document **{doc_id}** · {iterations} iteration{'s' if iterations != 1 else ''} · "
    f"Tool 1+2: `{config.EXTRACTION_MODEL}` · Reflection: `{config.REFLECTION_MODEL}`"
)

_PIPELINE_MERMAID = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  body { margin: 0; padding: 8px 0; background: transparent; }
  .mermaid { background: transparent; }
</style>
</head>
<body>
<div class="mermaid">
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "13px",
    "fontFamily": "Helvetica, Arial, sans-serif",
    "lineColor": "#9ca3af",
    "primaryColor": "#cfe2ff",
    "primaryTextColor": "#084298",
    "primaryBorderColor": "#0d6efd",
    "edgeLabelBackground": "transparent"
  },
  "flowchart": { "curve": "basis", "nodeSpacing": 50, "rankSpacing": 80 }
}}%%
flowchart LR
    PDF([PDF]):::green
    GT([Ground Truth\\nJSON]):::green
    EP([extraction_prompt]):::green

    T1["Tool 1\\nStructure Analysis\\nClaude Haiku"]:::blue
    T2["Tool 2\\nJSON Extraction\\nClaude Haiku"]:::blue
    EVALPY["eval.py\\nDeterministic Score"]:::gray
    LLMEVAL["LLM Evaluator\\nClaude Haiku"]:::blue
    REF["Reflection\\nClaude Sonnet"]:::blue

    PDF -->|thumbnails| T1
    PDF -->|full images| T2
    T1 -->|section boundaries| T2
    EP --> T2
    T2 -->|prediction JSON| EVALPY
    T2 -->|prediction JSON| LLMEVAL
    GT --> EVALPY
    GT --> LLMEVAL
    EVALPY --> REF
    LLMEVAL --> REF
    REF -.->|updated prompt| EP

    classDef green fill:#d4edda,stroke:#28a745,color:#155724,font-weight:500
    classDef blue  fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,font-weight:500
    classDef gray  fill:#f3f4f6,stroke:#9ca3af,color:#374151,font-weight:500
</div>
<script>mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });</script>
</body>
</html>
"""

if not run_button:
    components.html(_PIPELINE_MERMAID, height=360)
    st.caption(
        "**Green** — inputs (extraction_prompt is optimized each iteration) · "
        "**Blue** — LLM calls · **Gray** — deterministic scorer · "
        "**Red dashed** — GEPA optimization loop"
    )
    st.markdown(
        "Select a document and number of iterations in the sidebar, then click **▶ Run Pipeline**."
    )
    st.stop()

# ── Load ground truth and base prompts ───────────────────────────────────────
client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

with open(gt_path, encoding="utf-8") as f:
    ground_truth = json.load(f)

extraction_prompt = (config.PROMPTS_DIR / "extraction.txt").read_text(encoding="utf-8").strip()

scores_history: list[float] = []
subscores_history: list[dict] = []

# ── Step 1: PDF → Images (once, shared across iterations) ────────────────────
st.header("Step 1 — PDF → Images", divider="gray")

with st.status("📄 Converting PDF to page images...", expanded=True) as step1_status:
    # Low-res thumbnails for display + structure analysis
    thumbnails = pdf_to_images(pdf_path, dpi=config.STRUCTURE_DPI)
    # Full-res images for extraction only
    full_images = pdf_to_images(pdf_path, dpi=config.PDF_DPI)
    step1_status.update(
        label=f"📄 PDF converted — **{len(full_images)} page{'s' if len(full_images) != 1 else ''}** "
              f"({config.STRUCTURE_DPI} DPI thumbnails + {config.PDF_DPI} DPI for extraction)",
        state="complete",
        expanded=False,
    )

# Thumbnail strip (use low-res for display)
thumb_count = min(len(thumbnails), 8)
thumb_cols = st.columns(thumb_count)
for i in range(thumb_count):
    thumb_cols[i].image(png_to_pil(thumbnails[i]), caption=f"p.{i+1}", use_container_width=True)
if len(thumbnails) > 8:
    st.caption(f"*...and {len(thumbnails) - 8} more pages*")


# ── Iteration loop ────────────────────────────────────────────────────────────
for iter_num in range(1, iterations + 1):

    if iterations > 1:
        st.header(f"Iteration {iter_num} / {iterations}", divider="blue")
    else:
        st.header("Extraction & Evaluation", divider="blue")

    # ── Active prompts for this iteration ────────────────────────────────────
    with st.expander("📝 Extraction prompt used this iteration", expanded=False):
        st.code(extraction_prompt, language="text")

    # ── Step 2: Structure Analysis ────────────────────────────────────────────
    st.subheader("Step 2 — Structure Analysis (Tool 1)")

    with st.status("🗂️ Analyzing document structure...", expanded=True) as step2_status:
        structure = analyze_structure(thumbnails, client)
        n_sections = len(structure.get("sections", []))
        step2_status.update(
            label=f"🗂️ Structure: **{n_sections} section{'s' if n_sections != 1 else ''}** identified",
            state="complete",
            expanded=False,
        )

    # Section map visualization
    section_parts = []
    for s in structure.get("sections", []):
        pages_label = ", ".join(str(p + 1) for p in s.get("pages", []))
        section_parts.append(f"**{s.get('prefix', '?')}** (p. {pages_label})")
    st.markdown("**Section map:** " + " → ".join(section_parts))

    with st.expander("Tool 1 output — raw structure JSON", expanded=False):
        st.json(structure)

    # ── Step 3: Per-section JSON Extraction ───────────────────────────────────
    st.subheader("Step 3 — JSON Extraction per Section (Tool 2)")

    sections_json: list[dict] = []
    for si, sec in enumerate(structure.get("sections", [])):
        prefix = sec.get("prefix", f"SEC{si + 1}")
        pages = sec.get("pages", [])
        pages_label = str([p + 1 for p in pages])
        description = sec.get("description", "")

        with st.status(
            f"🔍 Parsing section **{prefix}** (pages {pages_label})...", expanded=True
        ) as step3_status:
            section_images = [full_images[p] for p in pages if p < len(full_images)]
            section_result = parse_section(
                section_images, description, extraction_prompt, client
            )
            sections_json.append(section_result)
            has_error = section_result.get("_parse_error", False)
            step3_status.update(
                label=(
                    f"⚠️ Section **{prefix}** — JSON parse error"
                    if has_error
                    else f"✅ Section **{prefix}** — extracted"
                ),
                state="error" if has_error else "complete",
                expanded=False,
            )

        with st.expander(
            f"Tool 2 output — Section **{prefix}**" + (" ⚠️ parse error" if has_error else ""),
            expanded=has_error,  # only auto-open on errors
        ):
            if has_error:
                st.warning("The model's response could not be parsed as JSON. Raw output:")
                st.code(section_result.get("raw", ""), language="text")
            else:
                st.json(section_result)

    # ── Full merged JSON vs Ground Truth ─────────────────────────────────────
    prediction = {"doc_id": doc_id, "sections": sections_json}

    st.subheader("Output vs Ground Truth")
    col_out, col_gt = st.columns(2)
    with col_out:
        st.caption("**Model output** (full merged JSON)")
        st.json(prediction, expanded=2)
    with col_gt:
        st.caption("**Ground truth**")
        st.json(ground_truth, expanded=2)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    st.subheader("Step 4 — Evaluation")

    with st.status("📊 Evaluating extraction...", expanded=True) as step4_status:
        eval_result = evaluate(prediction, ground_truth, client)
        eval_report = eval_result["eval_report"]
        score = eval_report.get("score", 0.0)
        subscores = eval_report.get("subscores", {})
        step4_status.update(
            label=f"📊 Evaluation complete — Score: {score_emoji(score)} **{score:.4f}**",
            state="complete",
            expanded=False,
        )

    scores_history.append(score)
    subscores_history.append(subscores)

    # Score metrics row
    prev_score = scores_history[-2] if len(scores_history) > 1 else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        f"{score_emoji(score)} Overall Score",
        f"{score:.4f}",
        delta=f"{score - prev_score:+.4f}" if prev_score is not None else None,
    )
    m2.metric("Structure (45%)", f"{subscores.get('structure', 0):.4f}")
    m3.metric("Numbers (40%)", f"{subscores.get('numbers', 0):.4f}")
    m4.metric("Text (15%)", f"{subscores.get('text', 0):.4f}")

    # Issues table
    issues = eval_report.get("issues", [])
    if issues:
        issues_df = pd.DataFrame([
            {
                "severity": x.get("severity", ""),
                "category": x.get("category", ""),
                "kind": x.get("kind", ""),
                "path": x.get("path", ""),
                "expected": str(x.get("expected", ""))[:60],
                "got": str(x.get("got", ""))[:60],
                "penalty": x.get("penalty", 0),
            }
            for x in issues[:40]
        ])
        with st.expander(f"Issues — {len(issues)} total (showing top {min(len(issues), 40)})", expanded=True):
            st.dataframe(
                issues_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "penalty": st.column_config.NumberColumn("penalty", format="%.3f"),
                    "severity": st.column_config.TextColumn("sev", width="small"),
                    "category": st.column_config.TextColumn("cat", width="small"),
                    "kind": st.column_config.TextColumn("kind", width="small"),
                },
            )

    # LLM qualitative diagnosis
    diagnosis = eval_result.get("qualitative_feedback", "")
    if diagnosis:
        st.info(f"**LLM Diagnosis:** {diagnosis}")

    # ── Step 5: Reflection ────────────────────────────────────────────────────
    if iter_num < iterations:
        st.subheader("Step 5 — Reflection (GEPA-style)")

        with st.status(
            f"💡 Proposing improved prompts (iteration {iter_num} → {iter_num + 1})...",
            expanded=True,
        ) as step5_status:
            new_prompts = propose(extraction_prompt, eval_result, client)
            step5_status.update(
                label="💡 New extraction prompt proposed — ready for next iteration",
                state="complete",
                expanded=False,
            )

        with st.expander("Extraction prompt (before → after)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.caption("**Before**")
                st.code(extraction_prompt, language="text")
            with c2:
                st.caption("**After**")
                st.code(new_prompts["extraction_prompt"], language="text")

        extraction_prompt = new_prompts["extraction_prompt"]

    st.divider()


# ── Score progression (multi-iteration only) ──────────────────────────────────
if iterations > 1 and len(scores_history) > 1:
    st.header("📈 Score Progression", divider="green")

    chart_df = pd.DataFrame(
        {"Overall Score": scores_history},
        index=pd.RangeIndex(start=1, stop=len(scores_history) + 1, name="Iteration"),
    )
    st.line_chart(chart_df)

    total_delta = scores_history[-1] - scores_history[0]
    best_score = max(scores_history)
    best_iter = scores_history.index(best_score) + 1

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Final score", f"{scores_history[-1]:.4f}", delta=f"{total_delta:+.4f}")
    col_b.metric("Best score", f"{best_score:.4f}", delta=f"iter {best_iter}")
    col_c.metric("Iterations run", len(scores_history))

    with st.expander("Scores table"):
        full_df = pd.DataFrame(
            {
                "Iteration": range(1, len(scores_history) + 1),
                "Overall": scores_history,
                "Structure": [s.get("structure", 0) for s in subscores_history],
                "Numbers": [s.get("numbers", 0) for s in subscores_history],
                "Text": [s.get("text", 0) for s in subscores_history],
            }
        )
        st.dataframe(full_df, use_container_width=True, hide_index=True)
