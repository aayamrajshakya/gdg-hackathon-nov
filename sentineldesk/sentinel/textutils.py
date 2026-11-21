"""Text helpers: input assembly, language detection, PII masking and risk patterns."""
from __future__ import annotations

import re
from collections import Counter


def clean_str(value) -> str:
    """Return a stripped string; None, NaN and other non-strings become ''."""
    return value.strip() if isinstance(value, str) else ""


def build_text(subject, body) -> str:
    """Combine subject and body. Works when either one is missing."""
    subject, body = clean_str(subject), clean_str(body)
    if subject and body:
        return f"{subject}\n\n{body}"
    return subject or body


# ---------------------------------------------------------------- language detection
# Function-word voting is fast, deterministic and very reliable for EN vs DE.
_DE_WORDS = frozenset("""
der die das und ist nicht ich wir sie mit für auf bei ein eine einen einem einer zu von den dem des
bitte danke wird werden kann können haben hat habe unser unsere unseren ihr ihre ihnen sehr geehrte
geehrter hallo guten leider seit noch auch wie wenn oder aber dass sich es im ins vom zum zur nach
über wurde wurden keine kein mein meine gibt viele grüße freundlichen mehr jetzt heute schon nur
""".split())
_EN_WORDS = frozenset("""
the and is not i we you with for on at to of please thanks thank can have has our your very dear
hello hi since also how what when or but that this it are be my me there would could should been any
all after from they them their its were which into about just now today already only
""".split())
STOPWORDS = _DE_WORDS | _EN_WORDS | frozenset("""
a an in was will issue problem help support team regards best kind sincerely customer
cannot can't unable working work works since getting get got need needs us am do does
""".split())


def detect_language(text: str) -> str:
    """Return 'de' or 'en' from raw ticket text."""
    low = text.lower()
    tokens = re.findall(r"[a-zäöüß]+", low)
    de_score = sum(tok in _DE_WORDS for tok in tokens) + min(5, len(re.findall(r"[äöüß]", low)))
    en_score = sum(tok in _EN_WORDS for tok in tokens)
    return "de" if de_score > en_score else "en"


def keywords(texts: list[str], n: int = 5) -> list[str]:
    """Most frequent content words across texts (used for outage banners)."""
    counts = Counter(
        tok for t in texts for tok in re.findall(r"[a-zäöüß0-9]{3,}", t.lower()) if tok not in STOPWORDS
    )
    return [w for w, _ in counts.most_common(n)]


# ---------------------------------------------------------------- PII guardrail
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")
_IPV6 = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){3,7}[0-9A-Fa-f]{1,4}\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE = re.compile(r"(?<!\w)(?:\+|00)?\d[\d ()/.-]{6,}\d(?!\w)")
_ACCOUNT = re.compile(
    r"\b(account|acct|konto|kundennummer|customer|kunden|contract|vertrag)"
    r"(\s*(?:no\.?|number|nr\.?|id|#|nummer)?\s*[:#]?\s*)"
    r"((?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{4,})",
    re.IGNORECASE,
)


def _luhn_ok(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        n = int(ch)
        if double:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
        double = not double
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, IBANs, IPs, card numbers, phone and account numbers with tags."""
    if not text:
        return "", {}
    counts: Counter = Counter()

    def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            counts[label] += 1
            return f"[{label}]"
        return pattern.sub(repl, s)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_IBAN, "IBAN", text)
    text = sub(_IPV4, "IP", text)
    text = sub(_IPV6, "IP", text)
    text = sub(_CARD, "CARD", text, check=lambda s: _luhn_ok(re.sub(r"\D", "", s)))
    text = sub(_PHONE, "PHONE", text, check=lambda s: 9 <= len(re.sub(r"\D", "", s)) <= 15)

    def account_repl(m: re.Match) -> str:
        counts["ACCOUNT"] += 1
        return f"{m.group(1)}{m.group(2)}[ACCOUNT]"

    text = _ACCOUNT.sub(account_repl, text)
    return text, dict(counts)


# ---------------------------------------------------------------- risk patterns
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules)",
    r"disregard\s+(?:all\s+|your\s+|the\s+)?(?:instructions|rules|guidelines)",
    r"system\s+prompt",
    r"\byou\s+are\s+now\b",
    r"developer\s+mode",
    r"\bjailbreak",
    r"pretend\s+(?:to\s+be|you\s+are)",
    r"ignorier\w*\s+(?:alle\s+)?(?:vorherigen|bisherigen|obigen)\s+(?:anweisungen|instruktionen|regeln)",
    r"vergiss\s+(?:alle\s+)?(?:deine\s+)?(?:anweisungen|regeln)",
]

# Requests that must be backed by a historical resolution before AI may answer them.
RISKY_ACTIONS = {
    "a refund or compensation": [
        r"\brefund", r"money\s+back", r"reimburs", r"compensat", r"chargeback", r"service\s+credit",
        r"\bcredit\s+note", r"erstatt", r"geld\s+zurück", r"entschädig", r"gutschrift",
    ],
    "specific pricing or a discount": [
        r"\bdiscount", r"\bpric(?:e|es|ing)\b", r"\bquote\b", r"how\s+much", r"\bcosts?\b",
        r"rabatt", r"\bpreis", r"\bkosten\b", r"\bangebot\b",
    ],
    "an SLA, legal or warranty commitment": [
        r"\bsla\b", r"guarantee", r"warrant(?:y|ies)", r"\blegal(?:ly)?\b", r"lawsuit", r"\bsue\b",
        r"liabilit", r"\bgdpr\b", r"\bdsgvo\b", r"garantie", r"gewährleistung", r"haftung",
        r"rechtlich", r"anwalt", r"\bklage",
    ],
    "account or data deletion": [
        r"delete\s+(?:my|our|all)\s+(?:account|data|records)", r"right\s+to\s+(?:be\s+forgotten|erasure)",
        r"(?:konto|daten|account)\s+(?:vollständig\s+)?löschen", r"löschung\s+(?:meiner|unserer)",
    ],
    "privileged access or someone else's credentials": [
        r"admin(?:istrator)?\s+(?:rights|access|password|privileges)",
        r"(?:give|send|tell)\s+me\s+(?:the|his|her|their)\s+password", r"\bbypass",
        r"disable\s+(?:mfa|2fa|two.factor|security)", r"admin.?rechte", r"zugangsdaten\s+(?:von|für|eines)",
    ],
}

# Safety net for outages and security events, in both languages.
OUTAGE_PATTERNS = [
    r"\boutage\b",
    r"\b(?:system|service|server|site|website|network|vpn|production|platform|email|login)\s+(?:is\s+|are\s+|went\s+)?(?:completely\s+)?down\b",
    r"\bdown\s+for\s+(?:all|every)",
    r"\b(?:all|every)\s+(?:users|employees|staff|clinicians|customers|colleagues)\b.{0,40}\b(?:cannot|can't|unable|lost)",
    r"\bdata\s+breach\b", r"\bsecurity\s+(?:breach|incident)\b", r"\bransomware\b", r"\bhacked\b",
    r"\bdata\s+loss\b",
    r"\bausfall\b", r"\bausgefallen\b", r"\bnicht\s+(?:mehr\s+)?erreichbar\b", r"\bdatenleck\b",
    r"\bsicherheitsvorfall\b", r"\bgehackt\b", r"\bdatenverlust\b",
]


def find_terms(text: str, patterns: list[str], limit: int = 5) -> list[str]:
    """Return up to `limit` distinct matched snippets (lowercased)."""
    found: list[str] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            term = m.group(0).lower().strip()
            if term not in found:
                found.append(term)
            if len(found) >= limit:
                return found
    return found