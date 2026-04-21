"""
GEPA Results Showcase — Preloaded results presentation for BMW team.

No API calls required. Loads Run 7 (6-doc) and Run 6 (1-doc) results
and presents the prompt evolution story interactively.
"""

from __future__ import annotations

import json
from difflib import unified_diff
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ARCHIVE_DIR = RESULTS_DIR / "archive"
PROMPTS_DIR = RESULTS_DIR / "prompts"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEPA Results — BMW x MIT",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-metric {
    background: linear-gradient(135deg, #1a237e, #0d47a1);
    border-radius: 12px; padding: 20px; text-align: center;
    color: white; margin-bottom: 8px;
}
.hero-metric h1 { margin: 0; font-size: 2.4em; color: #90caf9; }
.hero-metric p { margin: 4px 0 0 0; font-size: 0.9em; color: #bbdefb; }
.phase-badge {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.75em; font-weight: 600; margin-right: 6px;
}
.phase-structure { background: #1b5e20; color: #a5d6a7; }
.phase-numeric { background: #e65100; color: #ffcc80; }
.diff-added { color: #66bb6a; }
.diff-removed { color: #ef5350; }
.reflection-card {
    background: #1e1e2e; border-radius: 8px; padding: 16px;
    border-left: 4px solid #3b82f6; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────

DOC_COLORS = {
    "201414": "#42a5f5",
    "344098": "#66bb6a",
    "629903": "#ffa726",
    "678856": "#ab47bc",
    "809570": "#ef5350",
    "944962": "#26c6da",
}


@st.cache_data
def load_run7():
    log_path = RESULTS_DIR / "gepa_log_20260420_171206.json"
    csv_path = RESULTS_DIR / "gepa_optimization_20260420_171206.csv"
    with open(log_path, encoding="utf-8") as f:
        log = json.load(f)
    csv_df = pd.read_csv(csv_path)
    return log, csv_df


@st.cache_data
def load_run6():
    log_path = ARCHIVE_DIR / "gepa_log_20260420_164005.json"
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_prompts():
    prompts = {}
    for p in sorted(PROMPTS_DIR.glob("p*_extraction.txt")):
        pid = p.stem.split("_")[0]
        prompts[pid] = p.read_text(encoding="utf-8")
    return prompts


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Showcase Controls")
    st.divider()
    run_choice = st.radio(
        "Select Run",
        ["Run 7 — 6 documents (definitive)", "Run 6 — 1 document (proof)"],
        index=0,
    )
    is_run7 = "Run 7" in run_choice
    st.divider()
    st.markdown("**Config used:**")
    st.caption("Model: `claude-sonnet-4-6`")
    st.caption("Extraction: parallel (3 workers)")
    st.caption("Eval: structure 45% / numbers 40% / text 15%")
    st.caption("Reflection: GEPA-style (Pareto + diversity)")
    st.divider()
    st.caption("No API calls — preloaded results only.")

# ── Load data ────────────────────────────────────────────────────────────────
run7_log, run7_csv = load_run7()
run6_log = load_run6()
prompts = load_prompts()

log = run7_log if is_run7 else run6_log
iterations = log["iterations"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# GEPA Prompt Evolution")
st.markdown("### BMW Repair Order Extraction — Automatic Quality Improvement")
st.caption("BMW Group x MIT GenAI Lab | April 2026 | Zero model retraining — prompt-only optimization")

st.divider()

# Hero metrics
baseline_mean = iterations[0]["mean_score"]
if is_run7:
    best_mean = iterations[-1]["mean_score"]
    best_single = max(iterations[-1]["scores"].values())
else:
    best_mean = iterations[-1]["mean_score"]
    best_single = max(iterations[-1]["scores"].values())

improvement_pct = ((best_mean - baseline_mean) / baseline_mean) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="hero-metric">
        <h1>{baseline_mean:.3f}</h1>
        <p>Baseline Score (Erwin's original prompt)</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="hero-metric">
        <h1>{best_mean:.3f}</h1>
        <p>Best Mean Score (evolved prompt)</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="hero-metric">
        <h1>+{improvement_pct:.0f}%</h1>
        <p>Improvement (no retraining)</p>
    </div>""", unsafe_allow_html=True)

if is_run7:
    n_evolutions = len(iterations) - 1  # don't count baseline
    st.info(
        f"**{n_evolutions} evolution steps** across **6 documents**. "
        f"Best single-doc score: **{best_single:.3f}**. "
        f"4 out of 6 documents scored above 0.90. "
        f"*Scroll down to [Peeling Back the Layers](#peeling-back-the-layers) "
        f"for a critical decomposition of what actually improved.*"
    )
else:
    st.info(
        f"**{len(iterations)} iterations** on document **629903**. "
        f"Score: {baseline_mean:.3f} → {best_mean:.3f} in {len(iterations)} steps."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: WHAT IS GEPA?
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("What is GEPA?")

st.markdown("""
**GEPA** (Generalized Evaluation-driven Prompt Augmentation, ICLR 2026) automatically
improves LLM system prompts **without retraining any model**. Instead of gradient descent
on weights, it uses natural-language feedback to evolve prompts — the prompt is the only
trainable parameter.
""")

st.subheader("How one iteration works")

st.markdown("""
Each iteration produces **one new candidate prompt** and evaluates it on **all documents**:
""")

_GEPA_LOOP_MERMAID = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>body { margin: 0; padding: 4px 0; background: transparent; }</style>
</head>
<body>
<div class="mermaid">
%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "13px", "fontFamily": "Helvetica, Arial, sans-serif",
    "lineColor": "#9ca3af", "primaryColor": "#cfe2ff",
    "primaryTextColor": "#084298", "primaryBorderColor": "#0d6efd"
  },
  "flowchart": { "curve": "basis", "nodeSpacing": 40, "rankSpacing": 60 }
}}%%
flowchart LR
    SELECT["1. Select Parent<br/><i>from Pareto front<br/>weighted by wins</i>"]:::blue
    REFLECT["2. Reflect<br/><i>LLM reads worst doc's<br/>eval failures</i>"]:::blue
    MUTATE["3. Mutate<br/><i>LLM proposes<br/>new prompt text</i>"]:::orange
    EVAL["4. Evaluate<br/><i>run new prompt on<br/>ALL 6 docs</i>"]:::blue
    UPDATE["5. Update Pool<br/><i>add to pool,<br/>recompute Pareto front</i>"]:::green

    SELECT --> REFLECT --> MUTATE --> EVAL --> UPDATE
    UPDATE -.->|"next<br/>iteration"| SELECT

    classDef blue fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,font-weight:500
    classDef orange fill:#fff3cd,stroke:#ffc107,color:#856404,font-weight:600
    classDef green fill:#d4edda,stroke:#28a745,color:#155724,font-weight:500
</div>
<script>mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });</script>
</body>
</html>
"""

import streamlit.components.v1 as components
components.html(_GEPA_LOOP_MERMAID, height=160)

st.markdown("""
Key points:
- **The prompt is fixed text** — it never changes mid-iteration. Mutation creates a *new* prompt;
  the old one stays in the pool unchanged.
- **All 6 documents are evaluated with the same prompt** in each iteration. This gives us a
  score vector (one score per doc) for every candidate.
- **The pool only grows** — no prompt is ever deleted. But dominated prompts fall off the
  Pareto front and stop being selected as parents.
""")

st.subheader("The Pareto Frontier and Prompt Selection")

st.markdown("""
When optimizing across multiple documents, improving on one doc might hurt another.
The **Pareto frontier** handles this tradeoff:
""")

col_g1, col_g2 = st.columns(2)
with col_g1:
    st.markdown("""
**What makes a prompt "dominated"?**

A prompt is **dominated** if some other prompt scores **equal or better on every
document**. A dominated prompt can never be the best choice — there's always a
strictly better alternative.

A prompt on the **Pareto front** is one that is NOT dominated — it's best at
*something*, even if it's not best at everything.

**Example from our run:**
- p001 is best on docs 678856 and 944962
- p002 is best on docs 201414 and 809570
- Neither dominates the other → both stay on the Pareto front
- p000 is worse than p001 on every doc → p000 is dominated, drops off
""")

with col_g2:
    st.markdown("""
**How is the next parent chosen?**

From the Pareto front only, using **win-frequency weighting**:
1. Count how many docs each Pareto prompt "wins" (scores highest on)
2. Prompts with more wins are more likely to be selected as parent
3. A small exploration bonus ensures even low-win prompts get a chance

**What happens to the "losing" prompts?**

They stay in the pool but are no longer selected as parents:
- A prompt is "losing" if it's **dominated** — another prompt is strictly better on
  all docs
- It's NOT about being worse than the previous iteration — a prompt can score lower
  than its parent but still be Pareto-optimal if it's best on some subset of docs
- Dominated prompts are dead branches — they contributed to evolution but their
  lineage ends
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OUR ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Our Architecture")

st.markdown("""
We apply GEPA to **BMW repair order extraction** — converting multi-page scanned PDFs into
structured JSON. The pipeline has two passes, with the extraction prompt as the optimizable piece.
""")

_ARCH_MERMAID = """
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
  "flowchart": { "curve": "basis", "nodeSpacing": 50, "rankSpacing": 70 }
}}%%
flowchart LR
    PDF([PDF<br/>Repair Order]):::green
    GT([Ground Truth<br/>JSON]):::green
    EP([Extraction<br/>Prompt]):::orange

    T1["Pass 1<br/>Structure ID<br/>+ Metadata"]:::blue
    T2["Pass 2<br/>Per-Section<br/>Extraction"]:::blue
    EVAL["Deterministic<br/>Scorer<br/>eval.py"]:::gray
    LLMEVAL["LLM<br/>Diagnosis"]:::blue
    REF["GEPA<br/>Reflection"]:::blue

    PDF -->|"low-res<br/>thumbnails"| T1
    PDF -->|"full-res<br/>images"| T2
    T1 -->|"section boundaries<br/>+ prefixes"| T2
    EP -->|"system prompt"| T2
    T2 -->|"prediction"| EVAL
    T2 -->|"prediction"| LLMEVAL
    GT --> EVAL
    GT --> LLMEVAL
    EVAL -->|"score +<br/>issues"| REF
    LLMEVAL -->|"qualitative<br/>feedback"| REF
    REF -.->|"improved<br/>prompt"| EP

    classDef green fill:#d4edda,stroke:#28a745,color:#155724,font-weight:500
    classDef blue  fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a,font-weight:500
    classDef gray  fill:#f3f4f6,stroke:#9ca3af,color:#374151,font-weight:500
    classDef orange fill:#fff3cd,stroke:#ffc107,color:#856404,font-weight:600
</div>
<script>mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });</script>
</body>
</html>
"""

import streamlit.components.v1 as components
components.html(_ARCH_MERMAID, height=340)

st.caption(
    "**Green** — inputs | **Blue** — LLM calls (Claude Sonnet 4.6) | "
    "**Gray** — deterministic scorer | **Yellow** — the optimizable prompt | "
    "**Dashed** — GEPA feedback loop"
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Documents", "6")
col_b.metric("Evolution Steps", "4")
col_c.metric("Total Time", "~50 min")
col_d.metric("API Cost", "~$4")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SCORE PROGRESSION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Score Progression")

fig = go.Figure()

if is_run7:
    # Per-doc traces
    doc_ids = list(iterations[0]["scores"].keys())
    for doc_id in doc_ids:
        x_vals = []
        y_vals = []
        for it in iterations:
            if doc_id in it["scores"]:
                x_vals.append(it["iteration"])
                y_vals.append(it["scores"][doc_id])
        dash = "dot" if not it.get("partial") else "dash"
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            name=doc_id,
            mode="lines+markers",
            line=dict(width=1.5, dash="dot", color=DOC_COLORS.get(doc_id, "#999")),
            marker=dict(size=5),
            opacity=0.7,
        ))

    # Mean trace
    mean_x = [it["iteration"] for it in iterations if it["mean_score"] is not None]
    mean_y = [it["mean_score"] for it in iterations if it["mean_score"] is not None]
    fig.add_trace(go.Scatter(
        x=mean_x, y=mean_y,
        name="Mean",
        mode="lines+markers",
        line=dict(width=3.5, color="#ffffff"),
        marker=dict(size=9, color="#ffffff", line=dict(width=2, color="#1a237e")),
    ))

    # Phase annotations
    fig.add_vrect(x0=0.8, x1=4.2, fillcolor="#1b5e20", opacity=0.08,
                  layer="below", line_width=0)
    fig.add_vrect(x0=4.2, x1=5.2, fillcolor="#e65100", opacity=0.08,
                  layer="below", line_width=0)
    fig.add_annotation(x=2.5, y=0.95, text="Phase 1: Structure",
                       showarrow=False, font=dict(color="#a5d6a7", size=11))
    fig.add_annotation(x=4.7, y=0.95, text="Phase 2: Types",
                       showarrow=False, font=dict(color="#ffcc80", size=11))
else:
    # Single doc — simple line
    doc_id = list(iterations[0]["scores"].keys())[0]
    x_vals = [it["iteration"] for it in iterations]
    y_vals = [it["scores"][doc_id] for it in iterations]
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        name=doc_id,
        mode="lines+markers",
        line=dict(width=3, color="#42a5f5"),
        marker=dict(size=10),
    ))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Iteration", dtick=1),
    yaxis=dict(title="Score", range=[0, 1]),
    height=420,
    margin=dict(l=40, r=40, t=30, b=40),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)

if is_run7:
    st.warning(
        "**Why do iterations 1-4 look flat?** The overall score = Structure (45%) + Numbers (40%) + Text (15%). "
        "In iterations 1-4, Numbers and Text are stuck at ~0.10 and ~0.20 because all values are returned as strings. "
        "Structure IS improving (0.53 -> 0.71 avg), but it's masked by the N+T floor. "
        "Iteration 5 fixes the type issue and scores jump to 0.77+."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PER-DOCUMENT HEATMAP (Run 7 only)
# ══════════════════════════════════════════════════════════════════════════════
if is_run7:
    st.divider()
    st.header("Per-Document Scores")

    doc_ids = list(iterations[0]["scores"].keys())
    n_iters = len(iterations)

    # Build matrix
    z_vals = []
    text_vals = []
    for doc_id in doc_ids:
        row = []
        text_row = []
        for it in iterations:
            score = it["scores"].get(doc_id)
            if score is not None:
                row.append(score)
                text_row.append(f"{score:.3f}")
            else:
                row.append(None)
                text_row.append("—")
        z_vals.append(row)
        text_vals.append(text_row)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=[f"Iter {it['iteration']}" for it in iterations],
        y=doc_ids,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale=[
            [0, "#b71c1c"],
            [0.3, "#f57f17"],
            [0.5, "#fdd835"],
            [0.7, "#66bb6a"],
            [1.0, "#1b5e20"],
        ],
        zmin=0.2,
        zmax=1.0,
        colorbar=dict(title="Score"),
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=60, r=40, t=20, b=40),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Scores across all 6 documents and 5 GEPA iterations.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TWO-PHASE PATTERN
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Two-Phase Improvement Pattern")

st.markdown("""
<span class="phase-badge phase-structure">Phase 1: Structure</span>
The loop first fixes missing sections, fields, and schema issues (iterations 2-4).

<span class="phase-badge phase-numeric">Phase 2: Numeric Types</span>
Once structure is stable, the loop discovers that numbers are returned as strings — fixing this causes a massive jump.
""", unsafe_allow_html=True)

if is_run7:
    # Compute average subscores per iteration
    iter_nums = []
    avg_structure = []
    avg_numbers = []
    avg_text = []

    for it in iterations:
        if not it.get("subscores"):
            continue
        iter_nums.append(it["iteration"])
        s_vals = [v.get("structure", 0) for v in it["subscores"].values()]
        n_vals = [v.get("numbers", 0) for v in it["subscores"].values()]
        t_vals = [v.get("text", 0) for v in it["subscores"].values()]
        avg_structure.append(sum(s_vals) / len(s_vals) if s_vals else 0)
        avg_numbers.append(sum(n_vals) / len(n_vals) if n_vals else 0)
        avg_text.append(sum(t_vals) / len(t_vals) if t_vals else 0)

    col_s, col_n = st.columns(2)
    with col_s:
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=iter_nums, y=avg_structure,
            name="Structure (45%)", mode="lines+markers",
            line=dict(width=3, color="#66bb6a"),
            fill="tozeroy", fillcolor="rgba(102,187,106,0.15)",
        ))
        fig_s.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Subscore", range=[0, 1]),
            xaxis=dict(title="Iteration", dtick=1),
            height=280, margin=dict(l=40, r=20, t=30, b=40),
            title=dict(text="Structure Subscore (avg)", font=dict(size=13)),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col_n:
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(
            x=iter_nums, y=avg_numbers,
            name="Numbers (40%)", mode="lines+markers",
            line=dict(width=3, color="#ffa726"),
            fill="tozeroy", fillcolor="rgba(255,167,38,0.15)",
        ))
        fig_n.add_trace(go.Scatter(
            x=iter_nums, y=avg_text,
            name="Text (15%)", mode="lines+markers",
            line=dict(width=2, color="#ab47bc", dash="dash"),
        ))
        fig_n.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Subscore", range=[0, 1]),
            xaxis=dict(title="Iteration", dtick=1),
            height=280, margin=dict(l=40, r=20, t=30, b=40),
            title=dict(text="Numbers & Text Subscores (avg)", font=dict(size=13)),
        )
        st.plotly_chart(fig_n, use_container_width=True)

    st.markdown(
        "> Structure must be fixed first (sections, fields, schema). "
        "Only then can the loop address value-level accuracy (numeric types, zero vs null)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PROMPT EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Prompt Evolution")

available_prompts = sorted(prompts.keys())
if len(available_prompts) >= 2:
    iter_select = st.slider(
        "Compare iteration",
        min_value=1,
        max_value=len(available_prompts) - 1,
        value=min(len(available_prompts) - 1, 4),
        help="Select which iteration's prompt to compare against its parent",
    )

    prev_id = available_prompts[iter_select - 1]
    curr_id = available_prompts[iter_select]
    prev_text = prompts[prev_id]
    curr_text = prompts[curr_id]

    prev_lines = prev_text.splitlines()
    curr_lines = curr_text.splitlines()

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Previous", f"{len(prev_lines)} lines", delta=f"{prev_id}")
    m2.metric("Current", f"{len(curr_lines)} lines", delta=f"{curr_id}")
    m3.metric("Change", f"+{max(0, len(curr_lines) - len(prev_lines))} lines",
              delta=f"{len(curr_lines) - len(prev_lines):+d} net")

    # Diff view
    diff = list(unified_diff(prev_lines, curr_lines, lineterm="",
                             fromfile=f"{prev_id}_extraction.txt",
                             tofile=f"{curr_id}_extraction.txt"))

    if diff:
        diff_html_parts = []
        for line in diff:
            if line.startswith("+++") or line.startswith("---"):
                diff_html_parts.append(f'<span style="color:#90a4ae">{line}</span>')
            elif line.startswith("+"):
                diff_html_parts.append(f'<span class="diff-added">{line}</span>')
            elif line.startswith("-"):
                diff_html_parts.append(f'<span class="diff-removed">{line}</span>')
            elif line.startswith("@@"):
                diff_html_parts.append(f'<span style="color:#64b5f6">{line}</span>')
            else:
                diff_html_parts.append(f'<span style="color:#9e9e9e">{line}</span>')

        diff_html = "<pre style='font-size:0.8em; line-height:1.4; background:#0d1117; padding:12px; border-radius:8px; overflow-x:auto;'>" + "\n".join(diff_html_parts) + "</pre>"
        with st.expander(f"Diff: {prev_id} → {curr_id}", expanded=True):
            st.markdown(diff_html, unsafe_allow_html=True)

    # Side by side full prompts
    with st.expander("Full prompts side-by-side"):
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"**{prev_id}** ({len(prev_lines)} lines)")
            st.code(prev_text, language="text")
        with c2:
            st.caption(f"**{curr_id}** ({len(curr_lines)} lines)")
            st.code(curr_text, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b: EXTRACTION INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Extraction Inspector")
st.markdown(
    "Flat field-by-field comparison across prompts. "
    "Color coding: "
    ':green[**Match**] · :red[**Wrong**] · :orange[**Type mismatch**] · :violet[**Fixed**] (was wrong in first prompt, corrected in last)'
)

SAMPLES_DIR = ROOT / "Data" / "Samples"
doc_ids_all = sorted(iterations[0]["scores"].keys()) if is_run7 else sorted(iterations[0]["scores"].keys())
prompt_ids_all = [it["prompt_id"] for it in iterations]


@st.cache_data
def load_prediction(prompt_id: str, doc_id: str) -> dict | None:
    path = RESULTS_DIR / f"{prompt_id}_{doc_id}_prediction.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_ground_truth(doc_id: str) -> dict | None:
    path = SAMPLES_DIR / f"{doc_id}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _flatten_json(obj, prefix=""):
    """Recursively flatten nested JSON into {dotted.path: leaf_value} pairs."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(_flatten_json(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten_json(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = obj
    return items


def _get_section(pred: dict, target_prefix: str) -> dict | None:
    """Find a section by prefix, handling the extra nesting some prompts produce."""
    for s in pred.get("sections", []):
        if s.get("prefix") == target_prefix:
            # Some prompts wrap content in sections[i].sections[0]
            if "header" not in s and "sections" in s and isinstance(s["sections"], list) and s["sections"]:
                return s["sections"][0]
            return s
    return None


def _value_display(v):
    """Short display string for a value."""
    if v is None:
        return "null"
    if isinstance(v, str):
        return f'"{v}"' if len(v) <= 50 else f'"{v[:47]}..."'
    return str(v)


def _compare_value(gt_val, pred_val):
    """Return (status, css_class) for a GT-vs-prediction comparison."""
    if gt_val == pred_val:
        return "match", "insp-match"
    if pred_val == "—":  # missing from prediction
        return "missing", "insp-wrong"
    if gt_val == "—":  # extra in prediction
        return "extra", "insp-extra"
    # Type mismatch: same string representation but different Python types
    if str(gt_val) == str(pred_val):
        return "type", "insp-type"
    return "wrong", "insp-wrong"


insp_col1, insp_col2 = st.columns(2)
with insp_col1:
    insp_doc = st.selectbox("Document", doc_ids_all, key="insp_doc")
with insp_col2:
    insp_prompts = st.multiselect(
        "Prompts to compare",
        prompt_ids_all,
        default=[prompt_ids_all[0], prompt_ids_all[-1]],
        max_selections=4,
        key="insp_prompts",
    )

if insp_doc and insp_prompts:
    gt_full = load_ground_truth(insp_doc)

    # Score badges
    score_cols = st.columns(len(insp_prompts) + 1)
    score_cols[0].markdown("**Ground Truth**")
    for j, pid in enumerate(insp_prompts):
        it_data = next((it for it in iterations if it["prompt_id"] == pid), None)
        score = it_data["scores"].get(insp_doc, 0) if it_data else 0
        score_cols[j + 1].metric(pid, f"{score:.3f}")

    if gt_full and gt_full.get("sections"):
        section_prefixes = [s.get("prefix", f"SEC{i}") for i, s in enumerate(gt_full["sections"])]
        insp_section = st.selectbox(
            "Section to inspect",
            section_prefixes,
            key="insp_section",
        )

        # Load data for selected section
        sec_idx = section_prefixes.index(insp_section)
        gt_sec = gt_full["sections"][sec_idx]
        gt_flat = _flatten_json(gt_sec)
        # Remove meta keys we don't want to compare
        for drop_key in ["section_id", "prefix", "page_count"]:
            gt_flat.pop(drop_key, None)

        pred_flats = {}
        for pid in insp_prompts:
            pred_full = load_prediction(pid, insp_doc)
            if pred_full:
                pred_sec = _get_section(pred_full, insp_section)
                if pred_sec:
                    pf = _flatten_json(pred_sec)
                    for drop_key in ["section_id", "prefix", "page_count"]:
                        pf.pop(drop_key, None)
                    pred_flats[pid] = pf
                else:
                    pred_flats[pid] = {}
            else:
                pred_flats[pid] = {}

        # Collect all paths
        all_paths = set(gt_flat.keys())
        for pf in pred_flats.values():
            all_paths.update(pf.keys())
        all_paths = sorted(all_paths)

        # Categorize paths for filtering
        first_pid = insp_prompts[0]
        last_pid = insp_prompts[-1]

        # Build category counts
        n_match = 0
        n_fixed = 0
        n_wrong = 0
        n_type = 0
        n_missing = 0
        for path in all_paths:
            gt_val = gt_flat.get(path, "—")
            last_val = pred_flats.get(last_pid, {}).get(path, "—")
            first_val = pred_flats.get(first_pid, {}).get(path, "—")
            status, _ = _compare_value(gt_val, last_val)
            first_status, _ = _compare_value(gt_val, first_val)
            if status == "match":
                if first_status != "match":
                    n_fixed += 1
                else:
                    n_match += 1
            elif status == "type":
                n_type += 1
            elif status == "missing":
                n_missing += 1
            else:
                n_wrong += 1

        # Summary + filter
        sum_cols = st.columns(5)
        sum_cols[0].metric("Total fields", len(all_paths))
        sum_cols[1].metric("Correct", n_match, help="Match in last prompt")
        sum_cols[2].metric("Fixed", n_fixed, help=f"Wrong in {first_pid}, correct in {last_pid}")
        sum_cols[3].metric("Wrong", n_wrong + n_missing, help="Still wrong in last prompt")
        sum_cols[4].metric("Type mismatch", n_type, help="Value matches but type differs")

        show_filter = st.radio(
            "Show fields",
            ["All", "Fixed only", "Still wrong", "Type mismatches"],
            horizontal=True,
            key="insp_filter",
        )

        # CSS for the comparison table
        st.markdown(
            """<style>
            .insp-table { width: 100%; border-collapse: collapse; font-size: 0.82em; font-family: monospace; }
            .insp-table th { background: #1a1a2e; color: #e0e0e0; padding: 6px 8px; text-align: left;
                             position: sticky; top: 0; z-index: 1; border-bottom: 2px solid #444; }
            .insp-table td { padding: 4px 8px; border-bottom: 1px solid #2a2a3e; max-width: 220px;
                             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .insp-table tr:hover td { background: #1e1e3a !important; }
            .insp-match td { background: #0d2818; }
            .insp-wrong td { background: #2d0a0a; }
            .insp-type td { background: #2d2200; }
            .insp-extra td { background: #1a1a2e; }
            .insp-fixed td { background: #1a0d2e; }
            .insp-badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 0.8em; font-weight: bold; }
            .badge-match { background: #166534; color: #bbf7d0; }
            .badge-wrong { background: #991b1b; color: #fecaca; }
            .badge-type { background: #92400e; color: #fde68a; }
            .badge-fixed { background: #5b21b6; color: #ddd6fe; }
            .badge-missing { background: #6b2121; color: #fca5a5; }
            .badge-extra { background: #374151; color: #d1d5db; }
            </style>""",
            unsafe_allow_html=True,
        )

        # Build HTML table
        header_cells = "<th>Field Path</th><th>Ground Truth</th>"
        for pid in insp_prompts:
            header_cells += f"<th>{pid}</th>"
        header_cells += "<th>Status</th>"

        rows_html = []
        for path in all_paths:
            gt_val = gt_flat.get(path, "—")
            last_val = pred_flats.get(last_pid, {}).get(path, "—")
            first_val = pred_flats.get(first_pid, {}).get(path, "—")

            last_status, row_class = _compare_value(gt_val, last_val)
            first_status, _ = _compare_value(gt_val, first_val)

            # Determine if this was fixed (wrong in first → correct in last)
            is_fixed = last_status == "match" and first_status != "match"
            if is_fixed:
                row_class = "insp-fixed"

            # Apply filter
            if show_filter == "Fixed only" and not is_fixed:
                continue
            if show_filter == "Still wrong" and last_status in ("match",) and not is_fixed:
                continue
            if show_filter == "Still wrong" and is_fixed:
                continue
            if show_filter == "Type mismatches" and last_status != "type":
                continue

            # Build cells
            gt_display = _value_display(gt_val) if gt_val != "—" else '<span style="color:#666">—</span>'

            cells = f"<td>{path}</td><td>{gt_display}</td>"
            for pid in insp_prompts:
                pval = pred_flats.get(pid, {}).get(path, "—")
                pstatus, _ = _compare_value(gt_val, pval)
                disp = _value_display(pval) if pval != "—" else '<span style="color:#666">—</span>'
                # Highlight individual cell if it differs from GT
                if pstatus == "match":
                    cells += f"<td>{disp}</td>"
                elif pstatus == "type":
                    cells += f'<td style="color:#fde68a">{disp} <span style="color:#92400e;font-size:0.75em">({type(pval).__name__})</span></td>'
                else:
                    cells += f'<td style="color:#fca5a5">{disp}</td>'

            # Status badge
            if is_fixed:
                badge = '<span class="insp-badge badge-fixed">FIXED</span>'
            elif last_status == "match":
                badge = '<span class="insp-badge badge-match">OK</span>'
            elif last_status == "type":
                badge = '<span class="insp-badge badge-type">TYPE</span>'
            elif last_status == "missing":
                badge = '<span class="insp-badge badge-missing">MISSING</span>'
            elif last_status == "extra":
                badge = '<span class="insp-badge badge-extra">EXTRA</span>'
            else:
                badge = '<span class="insp-badge badge-wrong">WRONG</span>'

            cells += f"<td>{badge}</td>"
            rows_html.append(f'<tr class="{row_class}">{cells}</tr>')

        table_html = f"""
        <div style="max-height: 600px; overflow-y: auto; border: 1px solid #333; border-radius: 6px;">
        <table class="insp-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
        </table>
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption(f"Showing {len(rows_html)} of {len(all_paths)} fields for section {insp_section}")

        # Expandable raw JSON for those who want it
        with st.expander("Raw JSON (expandable)"):
            raw_cols = st.columns(len(insp_prompts) + 1)
            with raw_cols[0]:
                st.caption("**Ground Truth**")
                st.json(gt_sec, expanded=1)
            for j, pid in enumerate(insp_prompts):
                pred_full = load_prediction(pid, insp_doc)
                with raw_cols[j + 1]:
                    st.caption(f"**{pid}**")
                    if pred_full:
                        pred_sec = _get_section(pred_full, insp_section)
                        if pred_sec:
                            st.json(pred_sec, expanded=1)
                        else:
                            st.warning(f"Section {insp_section} not found")
                    else:
                        st.warning("Prediction not found")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: REFLECTION QUALITY
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Reflection Quality — Why It Works")

st.markdown(
    "Each iteration, the loop picks the worst-scoring document, diagnoses root causes, "
    "and proposes prompt changes. The scores below look low in iterations 2-4 because "
    "**Numbers (40%) and Text (15%) are capped** until the numeric type fix — "
    "but **Structure is climbing steadily** (see the subscore charts above)."
)

for it in iterations[1:]:  # Skip baseline
    feedback = it.get("reflection_target_feedback", "")
    target_doc = it.get("reflection_target_doc", "?")
    target_score = it.get("reflection_target_score", 0)
    prompt_diff_data = it.get("prompt_diff", {})
    deltas = it.get("deltas", {})
    parent_mean = it.get("parent_mean_score", 0)
    new_mean = it.get("mean_score", 0)
    mean_delta = new_mean - parent_mean if parent_mean else 0

    # Build a summary label showing the improvement
    improved = sum(1 for d in deltas.values() if d.get("status") == "improved")
    worsened = sum(1 for d in deltas.values() if d.get("status") == "worsened")
    mean_arrow = f"+{mean_delta:.3f}" if mean_delta > 0 else f"{mean_delta:.3f}"

    with st.expander(
        f"Iteration {it['iteration']}: {it['prompt_id']} — "
        f"Mean {parent_mean:.3f} -> {new_mean:.3f} ({mean_arrow}) | "
        f"{improved} docs improved, {worsened} worsened",
        expanded=(it["iteration"] == 2),
    ):
        # Before → After metrics row
        ba1, ba2, ba3, ba4 = st.columns(4)
        ba1.metric("Mean Before", f"{parent_mean:.3f}")
        ba2.metric("Mean After", f"{new_mean:.3f}", delta=f"{mean_delta:+.3f}")
        ba3.metric("Docs Improved", f"{improved} / {len(deltas)}")
        if prompt_diff_data:
            ba4.metric("Prompt Change",
                       f"+{prompt_diff_data.get('lines_added', 0)} / -{prompt_diff_data.get('lines_removed', 0)}")

        # Per-doc deltas table
        if deltas:
            st.markdown("**Per-document results:**")
            delta_rows = []
            for doc_id, d in sorted(deltas.items()):
                status = d.get("status", "")
                icon = "+" if status == "improved" else ("-" if status == "worsened" else "=")
                delta_rows.append({
                    "": icon,
                    "Document": doc_id,
                    "Before": f"{d.get('parent_score', 0):.3f}",
                    "After": f"{d.get('score', 0):.3f}",
                    "Delta": f"{d.get('delta', 0):+.4f}",
                    "S": f"{d.get('subscores', {}).get('structure', 0):.2f}",
                    "N": f"{d.get('subscores', {}).get('numbers', 0):.2f}",
                    "T": f"{d.get('subscores', {}).get('text', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Diagnosis
        st.markdown(f"**What the loop diagnosed** (worst doc: `{target_doc}`, score: {target_score:.3f}):")
        if feedback:
            st.markdown(f"> {feedback[:600]}{'...' if len(feedback) > 600 else ''}")

        # Top issues
        top_issues = it.get("reflection_target_top_issues", [])
        if top_issues:
            st.markdown("**Top issues found:**")
            issues_df = pd.DataFrame(top_issues[:7])
            st.dataframe(issues_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PARETO FRONTIER EVOLUTION
# ══════════════════════════════════════════════════════════════════════════════
if is_run7:
    st.divider()
    st.header("Pareto Frontier Evolution")

    st.markdown("How the frontier evolved iteration by iteration:")

    # Build a table showing Pareto membership per iteration
    pareto_history = []
    for it in iterations:
        pf = it.get("pareto_front", [it["prompt_id"]])
        wf = it.get("win_frequencies", {})
        pareto_history.append({
            "After Iter": it["iteration"],
            "New Prompt": it["prompt_id"],
            "Pareto Front": ", ".join(pf),
            "Front Size": len(pf),
            "Dominated (dropped)": ", ".join(
                p for p in [f"p{j:03d}" for j in range(it["iteration"])]
                if p not in pf and any(p == prev["prompt_id"] for prev in iterations[:it["iteration"]])
            ) or "none",
        })
    st.dataframe(pd.DataFrame(pareto_history), use_container_width=True, hide_index=True)

    st.markdown("""
**Reading this table:** After each iteration, a new prompt joins the pool. The Pareto front
is recomputed — prompts that are now dominated by a better one drop off. In iteration 5,
p004 dominates all others (scores best on all 6 docs), so the front collapses to just p004.
""")

    # Build Pareto data from iterations
    prompt_data = {}
    for it in iterations:
        pid = it["prompt_id"]
        if it["mean_score"] is not None:
            prompt_data[pid] = {
                "mean": it["mean_score"],
                "min": it["min_score"] if it.get("min_score") else min(it["scores"].values()),
                "scores": it["scores"],
            }

    # Get final Pareto front
    last_complete = [it for it in iterations if not it.get("partial")][-1]
    pareto_ids = last_complete.get("pareto_front", [])
    win_freq = last_complete.get("win_frequencies", {})

    # Scatter plot
    fig_pareto = go.Figure()

    for pid, data in prompt_data.items():
        is_pareto = pid in pareto_ids
        fig_pareto.add_trace(go.Scatter(
            x=[data["mean"]],
            y=[data["min"]],
            mode="markers+text",
            name=pid,
            text=[pid],
            textposition="top center",
            marker=dict(
                size=20 if is_pareto else 12,
                color="#ffa726" if is_pareto else "#616161",
                line=dict(width=2, color="#ffffff" if is_pareto else "#9e9e9e"),
            ),
        ))

    # Connect Pareto front
    if len(pareto_ids) > 1:
        pareto_points = [(prompt_data[p]["mean"], prompt_data[p]["min"]) for p in pareto_ids if p in prompt_data]
        pareto_points.sort()
        fig_pareto.add_trace(go.Scatter(
            x=[p[0] for p in pareto_points],
            y=[p[1] for p in pareto_points],
            mode="lines",
            name="Pareto front",
            line=dict(color="#ffa726", dash="dash", width=1.5),
            showlegend=False,
        ))

    fig_pareto.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mean Score (higher = better average)"),
        yaxis=dict(title="Min Score (higher = more robust)"),
        height=350,
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    # Win frequencies
    if win_freq:
        st.markdown("**Win frequencies** (how many documents each prompt is best on):")
        wf_df = pd.DataFrame([
            {"Prompt": k, "Wins": v} for k, v in sorted(win_freq.items())
        ])
        st.dataframe(wf_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PEELING BACK THE LAYERS
# ══════════════════════════════════════════════════════════════════════════════
if is_run7:
    st.divider()
    st.header("Peeling Back the Layers")
    st.markdown(
        "The headline numbers are impressive — but **what did the loop actually improve?** "
        "Let's progressively decompose the gains to understand what's real extraction "
        "improvement vs. what's formatting compliance."
    )

    # ── Helper: flatten + get_section for analysis ────────────────────────────
    def _flatten_analysis(obj, prefix=""):
        items = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                items.update(_flatten_analysis(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                items.update(_flatten_analysis(v, f"{prefix}[{i}]"))
        else:
            items[prefix] = obj
        return items

    def _get_section_analysis(pred, target_prefix):
        for s in pred.get("sections", []):
            if s.get("prefix") == target_prefix:
                if "header" not in s and "sections" in s and isinstance(s["sections"], list) and s["sections"]:
                    return s["sections"][0]
                return s
        return None

    # ── Compute all analysis data once ────────────────────────────────────────
    analysis_doc_ids = sorted(iterations[0]["scores"].keys())
    first_pid = iterations[0]["prompt_id"]
    last_pid = iterations[-1]["prompt_id"]

    @st.cache_data
    def compute_layer_analysis():
        rows = []
        for doc_id in analysis_doc_ids:
            gt_path = SAMPLES_DIR / f"{doc_id}.json"
            p0_path = RESULTS_DIR / f"{first_pid}_{doc_id}_prediction.json"
            p4_path = RESULTS_DIR / f"{last_pid}_{doc_id}_prediction.json"
            if not all(p.exists() for p in [gt_path, p0_path, p4_path]):
                continue
            gt_data = json.load(open(gt_path))
            p0_data = json.load(open(p0_path))
            p4_data = json.load(open(p4_path))

            exact_p0 = exact_p4 = tolerant_p0 = tolerant_p4 = 0
            total = 0
            type_only_p0 = type_only_p4 = 0
            newly_found_zero = newly_found_null = newly_found_real = newly_found_real_correct = 0
            newly_lost = 0
            val_fixed = val_broke = 0

            for sec in gt_data["sections"]:
                prefix = sec["prefix"]
                gt_flat = _flatten_analysis(sec)
                p0_sec = _get_section_analysis(p0_data, prefix)
                p4_sec = _get_section_analysis(p4_data, prefix)
                p0_flat = _flatten_analysis(p0_sec) if p0_sec else {}
                p4_flat = _flatten_analysis(p4_sec) if p4_sec else {}

                for path in gt_flat:
                    if path in ("section_id", "prefix", "page_count"):
                        continue
                    total += 1
                    gv = gt_flat[path]
                    p0v = p0_flat.get(path)
                    p4v = p4_flat.get(path)

                    # Exact match
                    if gv == p0v: exact_p0 += 1
                    if gv == p4v: exact_p4 += 1
                    # Type-tolerant
                    p0_tok = (gv == p0v) or (p0v is not None and str(gv) == str(p0v))
                    p4_tok = (gv == p4v) or (p4v is not None and str(gv) == str(p4v))
                    if p0_tok: tolerant_p0 += 1
                    if p4_tok: tolerant_p4 += 1
                    if not (gv == p0v) and p0_tok: type_only_p0 += 1
                    if not (gv == p4v) and p4_tok: type_only_p4 += 1

                    # Newly found / lost analysis
                    p0_present = p0v is not None
                    p4_present = p4v is not None
                    if not p0_present and p4_present:
                        is_zero = gv == 0 or gv == 0.0
                        is_null = gv is None
                        if is_zero: newly_found_zero += 1
                        elif is_null: newly_found_null += 1
                        else:
                            newly_found_real += 1
                            if p4_tok: newly_found_real_correct += 1
                    elif p0_present and not p4_present:
                        newly_lost += 1
                    elif p0_present and p4_present:
                        if not p0_tok and p4_tok: val_fixed += 1
                        elif p0_tok and not p4_tok: val_broke += 1

            rows.append({
                "doc_id": doc_id,
                "total": total,
                "exact_p0": exact_p0, "exact_p4": exact_p4,
                "tolerant_p0": tolerant_p0, "tolerant_p4": tolerant_p4,
                "type_only_p0": type_only_p0, "type_only_p4": type_only_p4,
                "newly_found_zero": newly_found_zero,
                "newly_found_null": newly_found_null,
                "newly_found_real": newly_found_real,
                "newly_found_real_correct": newly_found_real_correct,
                "newly_lost": newly_lost,
                "val_fixed": val_fixed, "val_broke": val_broke,
            })
        return rows

    layer_data = compute_layer_analysis()

    # ── LAYER 1: Type formatting ──────────────────────────────────────────────
    st.subheader("Layer 1 — Most of the gain is type formatting")
    st.markdown(
        f"The eval penalizes `\"629903\"` (string) vs `629903` (int) the same as a completely wrong value. "
        f"What if we compare type-tolerantly — treating `str(gt) == str(pred)` as a match for numeric fields?"
    )

    layer1_rows = []
    for r in layer_data:
        layer1_rows.append({
            "Document": r["doc_id"],
            f"{first_pid} exact": f"{r['exact_p0']/r['total']*100:.1f}%",
            f"{first_pid} type-tolerant": f"{r['tolerant_p0']/r['total']*100:.1f}%",
            f"{last_pid} exact": f"{r['exact_p4']/r['total']*100:.1f}%",
            f"{last_pid} type-tolerant": f"{r['tolerant_p4']/r['total']*100:.1f}%",
            "Type-tolerant delta": f"{(r['tolerant_p4'] - r['tolerant_p0'])/r['total']*100:+.1f}pp",
        })
    st.dataframe(pd.DataFrame(layer1_rows), use_container_width=True, hide_index=True)

    avg_tol_p0 = sum(r["tolerant_p0"]/r["total"] for r in layer_data) / len(layer_data) * 100
    avg_tol_p4 = sum(r["tolerant_p4"]/r["total"] for r in layer_data) / len(layer_data) * 100
    avg_exact_p0 = sum(r["exact_p0"]/r["total"] for r in layer_data) / len(layer_data) * 100
    avg_exact_p4 = sum(r["exact_p4"]/r["total"] for r in layer_data) / len(layer_data) * 100

    l1c1, l1c2, l1c3 = st.columns(3)
    l1c1.metric("Eval headline", f"{avg_exact_p0:.0f}% → {avg_exact_p4:.0f}%",
                delta=f"+{avg_exact_p4 - avg_exact_p0:.0f}pp")
    l1c2.metric("Type-tolerant", f"{avg_tol_p0:.0f}% → {avg_tol_p4:.0f}%",
                delta=f"+{avg_tol_p4 - avg_tol_p0:.0f}pp")
    l1c3.metric("Inflated by types", f"{(avg_exact_p4 - avg_exact_p0) - (avg_tol_p4 - avg_tol_p0):.0f}pp",
                help="How many percentage points of improvement are purely type formatting")

    st.info(
        f"**Under type-tolerant comparison, the real improvement is ~{avg_tol_p4 - avg_tol_p0:.0f}pp** "
        f"({avg_tol_p0:.0f}% → {avg_tol_p4:.0f}%), not the +{avg_exact_p4 - avg_exact_p0:.0f}pp the eval reports. "
        f"Two documents actually regress under this lens."
    )

    # ── LAYER 2: Zero/null completeness ───────────────────────────────────────
    st.subheader('Layer 2 — "Newly found" fields are mostly zeros')
    st.markdown(
        f"Of the remaining type-tolerant improvement, much comes from {last_pid} outputting fields "
        f"that {first_pid} omitted. But what are those fields?"
    )

    total_found = sum(r["newly_found_zero"] + r["newly_found_null"] + r["newly_found_real"] for r in layer_data)
    total_zeros = sum(r["newly_found_zero"] for r in layer_data)
    total_nulls = sum(r["newly_found_null"] for r in layer_data)
    total_real = sum(r["newly_found_real"] for r in layer_data)
    total_real_correct = sum(r["newly_found_real_correct"] for r in layer_data)
    total_lost = sum(r["newly_lost"] for r in layer_data)

    # Donut chart
    fig_donut = go.Figure(go.Pie(
        labels=["GT value = 0", "GT value = null", "Real values (correct)", "Real values (wrong)"],
        values=[total_zeros, total_nulls, total_real_correct, total_real - total_real_correct],
        hole=0.55,
        marker=dict(colors=["#374151", "#4b5563", "#166534", "#991b1b"]),
        textinfo="label+value",
        textfont=dict(size=12),
    ))
    fig_donut.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        annotations=[dict(text=f"{total_found}<br>newly<br>found", x=0.5, y=0.5,
                          font=dict(size=16, color="#e0e0e0"), showarrow=False)],
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )

    l2c1, l2c2 = st.columns([2, 1])
    with l2c1:
        st.plotly_chart(fig_donut, use_container_width=True)
    with l2c2:
        st.metric("Newly found", total_found)
        st.metric("GT = 0 or null", total_zeros + total_nulls,
                  delta=f"{(total_zeros + total_nulls) * 100 // total_found}% of total")
        st.metric("Real values", total_real, delta=f"{total_real_correct} correct")
        st.metric("Newly lost", total_lost, delta="regression", delta_color="inverse")

    st.markdown(
        f"**{total_zeros + total_nulls}** of {total_found} newly found fields ({(total_zeros + total_nulls) * 100 // total_found}%) "
        f"have a ground-truth value of `0` or `null`. The model already *saw* these on the page — "
        f"it just didn't bother outputting them because the value was empty or zero. "
        f"The loop taught it to be more complete, but this is **schema compliance**, not extraction capability."
    )

    # Per-doc breakdown
    layer2_rows = []
    for r in layer_data:
        nf = r["newly_found_zero"] + r["newly_found_null"] + r["newly_found_real"]
        layer2_rows.append({
            "Document": r["doc_id"],
            "Newly found": nf,
            "GT = 0": r["newly_found_zero"],
            "GT = null": r["newly_found_null"],
            "Real values": r["newly_found_real"],
            "Real correct": r["newly_found_real_correct"],
            "Newly lost": r["newly_lost"],
            "Net real gain": r["newly_found_real_correct"] - r["newly_lost"],
        })
    st.dataframe(pd.DataFrame(layer2_rows), use_container_width=True, hide_index=True)

    # ── LAYER 3: What's actually left ─────────────────────────────────────────
    st.subheader("Layer 3 — What the loop genuinely improved")
    st.markdown("After removing type formatting and zero/null completeness, the real extraction gains are:")

    total_val_fixed = sum(r["val_fixed"] for r in layer_data)
    total_val_broke = sum(r["val_broke"] for r in layer_data)
    total_fields = sum(r["total"] for r in layer_data)

    l3c1, l3c2, l3c3, l3c4 = st.columns(4)
    l3c1.metric("New real correct values", f"{total_real_correct}",
                help="Fields that were missing in p000, present and correct in p004, with a meaningful GT value")
    l3c2.metric("Values fixed", f"{total_val_fixed}",
                help="Fields present in both but wrong in p000, correct in p004")
    l3c3.metric("Values broken", f"{total_val_broke}",
                help="Fields correct in p000, wrong in p004", delta_color="inverse")
    l3c4.metric("Total GT fields", f"{total_fields}")

    real_gains = total_real_correct + total_val_fixed
    st.markdown(
        f"Across all 6 documents and ~{total_fields} ground-truth fields, the loop produced "
        f"**{real_gains} genuinely new correct extractions** — about "
        f"**{real_gains // len(layer_data)} per document**. It also broke {total_val_broke} previously "
        f"correct values and lost {total_lost} fields entirely."
    )

    # ── THE FINDING ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("The Finding: Eval Design Is the Bottleneck")

    st.markdown("""
This is a textbook case of **Goodhart's Law**: *"When a measure becomes a target,
it ceases to be a good measure."*

The GEPA loop optimizes whatever the eval rewards. Our eval conflates three very
different things under a single score:

| What the eval measures | What it means | Real importance |
|---|---|---|
| JSON type correctness (`"629903"` vs `629903`) | Serialization convention | Low — trivially fixable in post-processing |
| Schema completeness (outputting `cost: 0`) | Structural compliance | Medium — matters for downstream systems |
| Extraction accuracy (reading the right value) | Core capability | **High — this is what we actually care about** |

The loop spent most of its evolutionary budget on the first two — because they were
the highest-penalty items in the eval feedback. The eval was *correct* in its scoring,
but it was **measuring the wrong things** relative to what matters.
""")

    st.success("""
**Research implication:** In any self-improving agent system, the evaluation function
is not just a measurement tool — it's the *objective function* the system optimizes against.
A misaligned eval doesn't just give misleading scores; it actively steers the loop toward
low-value improvements. **Eval design deserves as much attention as the agent architecture itself.**
""")

    st.markdown("""
**What a better eval would look like:**

1. **Type-tolerant numeric comparison** — `str(gt) == str(pred)` for numbers.
   Post-processing can fix types; the model's job is to read the PDF.
2. **Weighted field importance** — a wrong dollar amount matters more than a missing
   `null` field. Not all fields are equal.
3. **Separate sub-metrics** — report extraction accuracy, schema compliance, and
   format correctness independently so the reflection can prioritize correctly.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: ARCHITECTURE DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("Architecture Decisions")

decisions = pd.DataFrame([
    {"Decision": "Sonnet 4.6 over Haiku 4.5", "Rationale": "Reliable instruction following for section prefixes and JSON types. Haiku inconsistently applies prefix codes."},
    {"Decision": "Parallel extraction (3 workers)", "Rationale": "4x speedup vs sequential. Max 3 to avoid API rate limits."},
    {"Decision": "Prefix injection from structure step", "Rationale": "Structure step correctly IDs section codes (ASI, BWO, CSI...). Extraction model may hallucinate its own — orchestrator overwrites."},
    {"Decision": "Erwin's original prompt as baseline", "Rationale": "True starting point (41 lines). Demonstrates the system improves ANY prompt, not just pre-optimized ones."},
    {"Decision": "Structure penalty in evaluator", "Rationale": "Without it, wrong prefixes scored S=1.0 (no penalties assigned to structure category). Bug fix was critical."},
    {"Decision": "Reflection diversity (bottom-2 random)", "Rationale": "Prevents over-fitting to single worst doc. Randomly selects from bottom-2 to broaden feedback signal."},
    {"Decision": "Section alignment feedback", "Rationale": "Evaluator shows 'CRITICAL: Expected ASI, BWO, CSI. Got: PRE-INVOICE, WORKORDER.' Guides reflection."},
])
st.dataframe(decisions, use_container_width=True, hide_index=True,
             column_config={
                 "Decision": st.column_config.TextColumn(width="medium"),
                 "Rationale": st.column_config.TextColumn(width="large"),
             })

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.caption("""
**Methodology:** GEPA-style prompt evolution (ICLR 2026). Pareto frontier tracking with win-frequency selection.
Deterministic evaluation (deepdiff-based JSON comparison) + LLM qualitative diagnosis.
Models: claude-sonnet-4-6 (extraction, reflection). Evaluation: structure 45%, numbers 40%, text 15%.
| April 20, 2026 | Bernardo Chalita + Claude
""")
