"""
Paths and settings for the Stack AI + eval + GEPA pipeline.

Adjust SAMPLES_DIR to your local Samples folder (copy of Data/Samples).
REPO_ROOT is used to import Scripts.eval from the repo.
"""

from pathlib import Path

# Experimentation/Bernardo folder (parent of scripts/)
BERNARDO_ROOT = Path(__file__).resolve().parent.parent

# Samples: ground truth JSONs, PDFs, and where we save extraction outputs / eval reports
SAMPLES_DIR = BERNARDO_ROOT / "Samples"
RESULTS_DIR = BERNARDO_ROOT / "results"

# Repo root (for importing Scripts.eval)
REPO_ROOT = BERNARDO_ROOT.parent.parent

# Eval script lives at repo Scripts/eval.py
EVAL_SCRIPT = REPO_ROOT / "Scripts" / "eval.py"

# Reflection LLM (GEPA-style): set to your API key env var or None to use stub
REFLECTION_MODEL = "gpt-4o"  # or "claude-3-5-sonnet-20241022", etc.
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Optional: eval config override (JSON path)
EVAL_CONFIG_PATH = None  # e.g. BERNARDO_ROOT / "eval_config.json"

# --- GEPA reflection via Stack AI (use free tokens) ---
# Run reflection inside your Stack AI "GEPA Reflection" workflow.
# Set STACK_AI_API_KEY in the environment to your Bearer token (do not commit the token).
STACK_AI_REFLECTION_ORG_ID = "475c0540-6417-4995-ae46-62b1f9b8fe4a"
STACK_AI_REFLECTION_FLOW_ID = "69b1cb21de9f3bbf5ecc8ac7"
STACK_AI_API_KEY_ENV = "STACK_AI_API_KEY"
STACK_AI_BASE_URL = "https://api.stackai.com/inference/v0/run"

# --- Extraction flow (core flow with document + prompt → LLM → output) ---
# Set these to your extraction workflow's org ID and flow ID (from Export → API).
# Document can be provided as: (1) public URL in run body, or (2) upload first via Documents API, then run with same user_id.
STACK_AI_EXTRACTION_ORG_ID = "475c0540-6417-4995-ae46-62b1f9b8fe4a"  # same org as reflection; change if different
STACK_AI_EXTRACTION_FLOW_ID = "69b1b1ecfe850cd80ce3a6fe"  # Gen AI - BMW - Bernardo Experimentation extraction flow
# Document node ID in the extraction flow (default doc-0 when there is one Files node).
STACK_AI_EXTRACTION_NODE_ID = "doc-0"
# Input key for the document when passing a URL in the run body. Often "doc-0" or "url-0"; check Export → API.
STACK_AI_EXTRACTION_DOC_INPUT_KEY = "doc-0"

# Documents API (upload files): Stack AI uses a *private* API key for document upload, not the public run key.
# In Stack AI: Settings → API Keys → copy the *Private* key. Set STACK_AI_PRIVATE_API_KEY in the environment.
# If unset, upload_samples_to_stack_ai.py falls back to STACK_AI_API_KEY (often 401 if that is the public key).
STACK_AI_DOCUMENTS_API_KEY_ENV = "STACK_AI_PRIVATE_API_KEY"


def ensure_dirs() -> None:
    """Create results dir if missing."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
