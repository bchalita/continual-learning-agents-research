"""
Stack AI client for text-only LLM calls.

Routes reflection, evaluation, and merge calls through Stack AI to avoid
burning Anthropic credits on non-vision tasks. Uses a simple two-input
flow: instructions (system prompt) + prompt (user message) → response.

Requires STACKAI_API_URL and STACKAI_API_KEY in .env.
"""

from __future__ import annotations

import requests

from . import config


def call(instructions: str, prompt: str) -> str:
    """Call the Stack AI flow with instructions (system prompt) and prompt (user message).

    Args:
        instructions: System prompt / instructions for the LLM (in-0).
        prompt: User message / task content (in-1).

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If Stack AI is not configured or the call fails.
    """
    if not config.STACKAI_API_URL or not config.STACKAI_API_KEY:
        raise RuntimeError(
            "Stack AI not configured. Set STACKAI_API_URL and STACKAI_API_KEY in .env"
        )

    headers = {
        "Authorization": f"Bearer {config.STACKAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "user_id": "",
        "in-0": instructions,
        "in-1": prompt,
    }

    response = requests.post(config.STACKAI_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    # Stack AI returns the output in various formats depending on flow config.
    # Try common output keys.
    if isinstance(result, dict):
        for key in ["outputs", "output", "out-0", "result"]:
            if key in result:
                val = result[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    # Nested output — try to find the text
                    for subkey in ["output", "text", "result", "out-0"]:
                        if subkey in val and isinstance(val[subkey], str):
                            return val[subkey]
                    return str(val)
        # Fallback: return the whole response as string
        return str(result)

    return str(result)
