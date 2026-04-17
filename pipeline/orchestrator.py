from __future__ import annotations

from pathlib import Path

import anthropic

from . import config
from .pdf_utils import pdf_to_images
from .tools import analyze_structure, parse_section


def extract(
    pdf_path: Path | str,
    extraction_prompt: str,
    client: anthropic.Anthropic,
    dpi: int = config.PDF_DPI,
) -> dict:
    """Structured agent pipeline: PDF → images → structure → per-section parse → merged JSON.

    Returns:
        {"doc_id": str, "sections": [section_dict, ...]}
    """
    pdf_path = Path(pdf_path)
    doc_id = pdf_path.stem

    # Low-res thumbnails for structure analysis (cheap); full-res only for extraction
    thumbnails = pdf_to_images(pdf_path, dpi=config.STRUCTURE_DPI)
    full_images = pdf_to_images(pdf_path, dpi=dpi)
    structure = analyze_structure(thumbnails, client)

    sections = []
    for sec in structure["sections"]:
        section_images = [full_images[p] for p in sec["pages"]]
        # TOOL 3 HOOK: replace section_images with cropped regions
        # e.g. section_images = splitter.crop(section_images, sec.get("layout"))

        section_json = parse_section(
            section_images,
            sec.get("description", ""),
            extraction_prompt,
            client,
        )
        sections.append(section_json)

    return {"doc_id": doc_id, "sections": sections}
