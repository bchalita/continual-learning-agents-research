from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

from . import config
from .merge import merge_sections
from .normalize import normalize_document
from .pdf_utils import pdf_to_images
from .tools import analyze_structure, parse_section


def extract(
    pdf_path: Path | str,
    extraction_prompt: str,
    client: anthropic.Anthropic,
    dpi: int = config.PDF_DPI,
    normalize: bool = False,
) -> dict:
    """Structured agent pipeline: PDF → images → structure+metadata → per-section parse → merged JSON.

    Pipeline stages (2026-04-19, Bernardo Chalita):
      Pass 1 (low DPI, all pages): Structure ID + metadata extraction
        - Identifies section boundaries (ASI, BWO, CSI, etc.)
        - Extracts document-level metadata (RO#, VIN, customer, etc.)
        - Single vision API call for both

      Pass 2 (full DPI, per section): Content extraction
        - Each section extracted independently
        - Receives metadata from Pass 1 as shared context
        - Prevents cross-section inconsistencies (e.g., different RO# readings)

      Post-processing: LLM merge (text-only, via Stack AI)
        - Deduplicates header fields across sections
        - Reconciles conflicts via majority vote or LLM
        - Falls back to deterministic merge if Stack AI unavailable

    Returns:
        {"doc_id": str, "sections": [section_dict, ...]}
    """
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem

    # ── Pass 1: structure + metadata (low-res, cheap) ──
    thumbnails = pdf_to_images(pdf_path, dpi=config.STRUCTURE_DPI)
    full_images = pdf_to_images(pdf_path, dpi=dpi)
    structure = analyze_structure(thumbnails, client)

    # Metadata extracted in the same pass — shared context for all sections
    metadata = structure.get("metadata", {})

    # ── Pass 2: per-section extraction (full-res, parallel) ──
    def _extract_one(sec: dict) -> tuple[int, dict]:
        idx = structure["sections"].index(sec)
        section_images = [full_images[p] for p in sec["pages"]]
        # Include prefix in description so extraction model knows the section type
        desc = sec.get("description", "")
        prefix = sec.get("prefix", "UNKNOWN")
        section_desc = f"{prefix} ({desc})" if desc else prefix
        section_json = parse_section(
            section_images,
            section_desc,
            extraction_prompt,
            client,
            metadata=metadata,
        )
        # Inject correct prefix/section_id from structure step
        # (extraction model may hallucinate its own codes)
        section_json["prefix"] = prefix
        section_json["section_id"] = f"{prefix}-{doc_id}"
        return idx, section_json

    sections = [None] * len(structure["sections"])
    with ThreadPoolExecutor(max_workers=min(3, len(structure["sections"]))) as executor:
        futures = {executor.submit(_extract_one, sec): sec for sec in structure["sections"]}
        for future in as_completed(futures):
            idx, section_json = future.result()
            sections[idx] = section_json

    # ── Post-processing: merge / reconcile headers ──
    document = {"doc_id": doc_id, "sections": sections}
    document = merge_sections(document)

    # ── Post-processing: type normalization (optional) ──
    if normalize:
        document = normalize_document(document)

    return document
