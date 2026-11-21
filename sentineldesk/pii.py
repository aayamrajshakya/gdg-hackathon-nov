"""PII guardrail: mask emails, phones, IPs, IBANs, cards and account numbers before any LLM call."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns first so e.g. an IBAN isn't eaten by the phone rule.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("URL_CREDENTIALS", re.compile(r"\b[a-z]+://[^\s:@/]+:[^\s@/]+@\S+", re.I)),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("IPV6", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){4,7}[0-9a-fA-F]{1,4}\b")),
    ("PHONE", re.compile(r"(?<![\w-])\+?\(?\d{1,4}\)?(?:[ ./-]?\(?\d{2,5}\)?){2,5}(?![\w-])")),
    ("ACCOUNT_ID", re.compile(
        r"(?i)\b(?:account|acct|konto|kundennummer|customer\s*(?:id|no|number)|"
        r"vertragsnummer|contract\s*(?:id|no))[\s#:.-]*(?:number|nr|no)?[\s#:.-]*([A-Z0-9-]{5,})")),
    ("SECRET", re.compile(r"(?i)\b(?:password|passwort|pwd|api[_ -]?key|token)\s*[:=]\s*\S+")),
]


@dataclass
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def mask(text: str) -> MaskResult:
    counts: dict[str, int] = {}
    out = text or ""
    for label, pat in _PATTERNS:
        def _sub(m: re.Match, label=label) -> str:
            # Don't treat short digit runs (years, versions, quantities) as phone numbers.
            if label == "PHONE" and sum(c.isdigit() for c in m.group(0)) < 7:
                return m.group(0)
            counts[label] = counts.get(label, 0) + 1
            if label == "ACCOUNT_ID" and m.lastindex:
                start, end = m.span(1)
                s0 = m.start(0)
                g = m.group(0)
                return g[: start - s0] + f"[{label}]" + g[end - s0:]
            return f"[{label}]"
        out = pat.sub(_sub, out)
    return MaskResult(out, counts)
