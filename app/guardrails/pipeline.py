"""Run injection, topic, and PII checks for chat-with-data."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.guardrails.injection import detect_injection
from app.guardrails.pii import mask_pii_in_text
from app.guardrails.topic import detect_off_topic


class GuardrailViolation(Exception):
    """Raised when the user message must be blocked before calling the LLM."""

    def __init__(self, code: str, user_message: str) -> None:
        self.code = code
        self.user_message = user_message
        super().__init__(user_message)


@dataclass
class GuardrailReport:
    """Non-fatal outcomes (e.g. PII masking)."""

    pii_labels: tuple[str, ...] = field(default_factory=tuple)


def apply_chat_guardrails(
    user_message: str,
    *,
    table_names: list[str],
) -> tuple[str, GuardrailReport]:
    """
    Block injection / obvious off-topic; mask PII in the string passed to the model.

    Returns ``(safe_text, report)``. Raises ``GuardrailViolation`` when the turn
    should not call the LLM.
    """
    inj = detect_injection(user_message)
    if inj:
        raise GuardrailViolation("injection", inj)

    off = detect_off_topic(user_message, table_names)
    if off:
        raise GuardrailViolation("topic", off)

    masked, labels = mask_pii_in_text(user_message)
    return masked, GuardrailReport(pii_labels=labels)
