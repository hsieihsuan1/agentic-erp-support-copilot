"""Guardrails: input screening, PII redaction, tool-output sanitising, scope checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (all |any |your )?(previous|prior|above)?\s*(instructions|rules)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (?!a customer)",
    r"without (human )?approval",
    r"developer mode",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

ERP_KEYWORDS = re.compile(
    r"\b(invoice|inv|order|ord|ticket|tck|payment|pay|fee|fees|credit|refund|memo|po|purchase|"
    r"shipping|shipped|delivery|tracking|return|rma|tax|vat|ledger|journal|period|close|"
    r"access|role|password|login|sign in|hold|release|backorder|backordered|edi|account|"
    r"customer|support|human|agent|manager|escalate|escalation|priority|sla|terms|status|erp)\b",
    re.IGNORECASE,
)
ID_RE = re.compile(r"\b(INV|ORD|TCK)-\d{4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class InputVerdict:
    allowed: bool
    reason: str | None = None
    message: str | None = None


def redact_pii(text: str) -> str:
    out = text
    for label, pattern in PII_PATTERNS:
        out = pattern.sub(f"[REDACTED_{label.upper()}]", out)
    return out


def contains_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


def check_input(text: str) -> InputVerdict:
    if contains_injection(text):
        return InputVerdict(
            False,
            "prompt_injection",
            "I can't follow instructions that try to override my rules. "
            "I can help with invoices, orders, tickets and ERP how-to questions.",
        )
    if not ERP_KEYWORDS.search(text) and not ID_RE.search(text):
        return InputVerdict(
            False,
            "out_of_scope",
            "I can only help with ERP support topics such as invoices, orders, tickets, "
            "payments and access. Could you rephrase your question in that context?",
        )
    return InputVerdict(True)


def sanitize_tool_output(text: str) -> tuple[str, bool]:
    """Neutralise instruction-like content inside data returned by tools (indirect injection)."""
    if contains_injection(text):
        return _INJECTION_RE.sub("[removed: suspected prompt injection]", text), True
    return text, False
