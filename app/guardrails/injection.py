"""Heuristic detection of prompt-injection / jailbreak phrasing."""

from __future__ import annotations

import re

_INJECTION_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)\b",
            re.IGNORECASE,
        ),
        "Requests to disregard prior instructions are not allowed.",
    ),
    (
        re.compile(
            r"\b(disregard|forget)\s+(the\s+)?(above|prior|previous|system)\b",
            re.IGNORECASE,
        ),
        "Attempts to override the assistant context are not allowed.",
    ),
    (
        re.compile(
            r"\byou\s+are\s+now\s+(a|an|the)\b",
            re.IGNORECASE,
        ),
        "Role-manipulation prompts are not allowed.",
    ),
    (
        re.compile(
            r"\b(system|developer)\s*:\s*",
            re.IGNORECASE,
        ),
        "Fake system or developer messages are not allowed.",
    ),
    (
        re.compile(
            r"\b(reveal|print|show|leak)\s+(your|the)\s+(system\s+)?(prompt|instructions?)\b",
            re.IGNORECASE,
        ),
        "Prompt-extraction requests are not allowed.",
    ),
    (
        re.compile(
            r"\b(DAN\s+mode|jailbreak|bypass\s+(safety|filter|guard))\b",
            re.IGNORECASE,
        ),
        "Jailbreak-style requests are not allowed.",
    ),
    (
        re.compile(
            r"```\s*(system|assistant)\b",
            re.IGNORECASE,
        ),
        "Embedded fake message blocks are not allowed.",
    ),
)


def detect_injection(text: str) -> str | None:
    """Return a user-facing reason if the text looks like injection; else ``None``."""
    t = text.strip()
    if not t:
        return None
    for rx, reason in _INJECTION_RES:
        if rx.search(t):
            return reason
    return None
