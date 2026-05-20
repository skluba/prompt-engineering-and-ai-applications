"""Heuristic PII masking before text is sent to the LLM."""

from __future__ import annotations

import re
from typing import NamedTuple

# Order matters: apply broader patterns before shorter overlapping ones.
_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{13,19}\b"
        ),  # credit-card-like or long digit runs
        "[CARD_REDACTED]",
        "card-like number",
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[SSN_REDACTED]",
        "SSN-like",
    ),
    (
        re.compile(
            r"\b[\w.+-]+@[\w-]+(?:\.[\w.-]+)+\b",
            re.IGNORECASE,
        ),
        "[EMAIL_REDACTED]",
        "email",
    ),
    (
        re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "[PHONE_REDACTED]",
        "phone",
    ),
)


class PiiMaskResult(NamedTuple):
    text: str
    labels: tuple[str, ...]


def mask_pii_in_text(raw: str) -> PiiMaskResult:
    """Replace common PII-shaped substrings; return cleaned text and mask labels applied."""
    if not raw:
        return PiiMaskResult("", ())
    out = raw
    applied: list[str] = []
    for rx, repl, label in _PATTERNS:
        if rx.search(out):
            applied.append(label)
            out = rx.sub(repl, out)
    return PiiMaskResult(out, tuple(dict.fromkeys(applied)))
