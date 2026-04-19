from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Stack AI — used for text-only LLM calls (reflection, evaluation, merge)
# to avoid burning Anthropic credits on non-vision tasks.
STACKAI_API_URL = os.environ.get("STACKAI_API_URL", "")
STACKAI_API_KEY = os.environ.get("STACKAI_API_KEY", "")

# Vision tasks (structure ID, extraction) — direct Anthropic API
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
# Text-only tasks (reflection) — routed through Stack AI when available
REFLECTION_MODEL = "claude-sonnet-4-6"

PDF_DPI = 100         # Extraction: needs to read fine text
STRUCTURE_DPI = 72   # Structure analysis: thumbnails sufficient to identify sections

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "Data" / "Samples"
PROMPTS_DIR = REPO_ROOT / "prompts"
RESULTS_DIR = REPO_ROOT / "results"
EVAL_SCRIPT = REPO_ROOT / "Scripts" / "eval.py"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
