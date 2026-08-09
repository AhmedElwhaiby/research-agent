"""
Shared Groq client + a small helper for JSON-structured calls.

Centralized here so the model name/provider is a one-line change later,
and so every node calls the LLM the same way.
"""

import os
import json
from typing import Any
from groq import Groq

MODEL = "openai/gpt-oss-120b"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env or environment."
            )
        _client = Groq(api_key=api_key)
    return _client


def call_llm(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    """
    Plain text completion.

    max_tokens defaults to 4096 rather than the API default (often ~1024
    depending on model) — the synthesizer and reviser produce full markdown
    reports with a Sources section, and a low cap was silently truncating
    output mid-report with no error raised. If a response still gets cut off,
    check response.choices[0].finish_reason == "length" (see below) before
    assuming the draft is simply short.
    """
    client = _get_client()
    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        # Truncated output that looks complete is a silent failure mode —
        # a report cut off before reaching "## Sources" will pass length checks
        # but fail citation consistency. Emit a warning so it is not overlooked.
        print(
            f"[warning] LLM response truncated at max_tokens={max_tokens}. "
            "Output is incomplete."
        )
    return choice.message.content or ""


def call_llm_json(
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Any:
    """
    Completion where the system prompt instructs the model to return
    ONLY JSON. Raises if parsing fails so callers can retry/handle it
    explicitly rather than silently getting a malformed structure.
    """
    raw = call_llm(system, user, temperature=temperature, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)