from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from . import config
from .pdf_utils import image_content_block

_structure_prompt: str | None = None


def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def analyze_structure(page_images: list[bytes], client: anthropic.Anthropic) -> dict:
    """Tool 1: send all page images to Claude and return section boundaries + metadata.

    Combined pass (2026-04-19, Bernardo Chalita): extracts both structure and
    document-level metadata in a single vision call. The metadata (RO#, VIN,
    customer, etc.) is later passed to Tool 2 as shared context so each
    section extraction has consistent anchor values.

    Returns:
        {
            "metadata": {"ro_number": str, "vin": str, ...},
            "sections": [{"prefix": str, "pages": [int], "description": str}]
        }
    Falls back to empty metadata + single section on failure.
    """
    global _structure_prompt
    if _structure_prompt is None:
        _structure_prompt = _load_prompt("structure.txt")

    content = [image_content_block(img) for img in page_images]
    content.append({"type": "text", "text": "Analyze this document and return the metadata and section structure as JSON."})

    response = client.messages.create(
        model=config.EXTRACTION_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": _structure_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    raw = _strip_fences(response.content[0].text)

    fallback_metadata: dict = {}
    fallback = {
        "metadata": fallback_metadata,
        "sections": [{"prefix": "UNKNOWN", "pages": list(range(len(page_images))), "description": ""}],
    }

    try:
        result = json.loads(raw)
        # Ensure metadata exists (even if empty)
        if "metadata" not in result or not isinstance(result.get("metadata"), dict):
            result["metadata"] = fallback_metadata
        # Validate sections
        if "sections" in result and isinstance(result["sections"], list) and result["sections"]:
            n = len(page_images)
            for s in result["sections"]:
                s["pages"] = [p for p in s.get("pages", []) if isinstance(p, int) and 0 <= p < n]
            result["sections"] = [s for s in result["sections"] if s.get("pages")]
            if result["sections"]:
                return result
        # Sections invalid but metadata might be good — keep metadata
        return {**fallback, "metadata": result.get("metadata", fallback_metadata)}
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return fallback


def parse_section(
    images: list[bytes],
    section_description: str,
    extraction_prompt: str,
    client: anthropic.Anthropic,
    metadata: dict | None = None,
) -> dict:
    """Tool 2: extract structured JSON for one document section.

    Args:
        images: Page images for this section (full DPI).
        section_description: Brief description from Tool 1.
        extraction_prompt: The optimizable system prompt.
        client: Anthropic API client.
        metadata: Document-level metadata from Tool 1 (RO#, VIN, customer, etc.).
            Passed as context so the extraction is consistent across sections.
            For example, if metadata says RO# is 344098, the extraction won't
            misread it as 344099 on a blurry page.

    Returns a section dict or {"_parse_error": True, "raw": str} on failure.
    """
    # Build context text with section info and shared metadata
    context_parts = [f"Section: {section_description}"]
    if metadata:
        context_parts.append(
            "Document-level metadata (shared across all sections, use as reference):\n"
            + json.dumps(metadata, indent=2)
        )
    context_text = "\n\n".join(context_parts)

    content = [{"type": "text", "text": context_text}]
    for img in images:
        content.append(image_content_block(img))
    content.append({"type": "text", "text": "\nReturn only the JSON object for this section."})

    response = client.messages.create(
        model=config.EXTRACTION_MODEL,
        max_tokens=4096,
        system=[{"type": "text", "text": extraction_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    raw = _strip_fences(response.content[0].text)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": raw[:2000]}
