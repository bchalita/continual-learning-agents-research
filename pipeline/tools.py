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
    """Tool 1: send all page images to Claude and return section boundary info.

    Returns:
        {"sections": [{"prefix": str, "pages": [int], "description": str}]}
    Falls back to a single section covering all pages on any failure.
    """
    global _structure_prompt
    if _structure_prompt is None:
        _structure_prompt = _load_prompt("structure.txt")

    content = [image_content_block(img) for img in page_images]
    content.append({"type": "text", "text": "Analyze this document and return the section structure as JSON."})

    response = client.messages.create(
        model=config.EXTRACTION_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": _structure_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    raw = _strip_fences(response.content[0].text)

    try:
        result = json.loads(raw)
        if "sections" in result and isinstance(result["sections"], list) and result["sections"]:
            n = len(page_images)
            for s in result["sections"]:
                s["pages"] = [p for p in s.get("pages", []) if isinstance(p, int) and 0 <= p < n]
            result["sections"] = [s for s in result["sections"] if s.get("pages")]
            if result["sections"]:
                return result
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return {"sections": [{"prefix": "UNKNOWN", "pages": list(range(len(page_images))), "description": ""}]}


def parse_section(
    images: list[bytes],
    section_description: str,
    extraction_prompt: str,
    client: anthropic.Anthropic,
) -> dict:
    """Tool 2: extract structured JSON for one document section.

    TOOL 3 HOOK: `images` would be replaced with cropped region images
    (header / content / footer) from a layout splitter when available.

    Returns a section dict or {"_parse_error": True, "raw": str} on failure.
    """
    content = [{"type": "text", "text": f"Section structure context: {section_description}"}]
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
