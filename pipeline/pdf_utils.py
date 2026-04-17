from __future__ import annotations

import base64
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: Path | str, dpi: int = 150) -> list[bytes]:
    """Convert every page of a PDF to PNG bytes at the given DPI."""
    doc = fitz.open(str(pdf_path))
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def images_to_base64(images: list[bytes]) -> list[str]:
    """Base64-encode a list of PNG byte strings."""
    return [base64.standard_b64encode(img).decode("utf-8") for img in images]


def image_content_block(image_bytes: bytes) -> dict:
    """Return an Anthropic API image content block for a PNG."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": b64},
    }
