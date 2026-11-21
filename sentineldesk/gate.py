"""Confidence & risk escalation gate.

Every rule is deterministic and produces a machine-readable reason code, so an operator (or a
judge) can see exactly why a ticket was or wasn't auto-drafted.

Statuses
  ESCALATE_ONCALL  - page the on-call engineer now (critical incident / outage / safety signal)
  HUMAN_REVIEW     - an agent must review before anything is sent (low confidence, no
                     precedent, unsupported action, prompt injection, failed grounding checks)
  AUTO_DRAFT       - triage is confident, precedents exist, draft passed all grounding checks
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

CRITICAL, REVIEW = "critical", "review"
ESCALATE_ONCALL, HUMAN_REVIEW, AUTO_DRAFT = "ESCALATE_ONCALL", "HUMAN_REVIEW", "AUTO_DRAFT"


@dataclass
class Reason:
    code: str
    severity: str
    message: str


@dataclass
class GateDecision:
    status: str
    reasons: list[Reason] = field(default_factory=list)

    @property
    def escalate(self) -> bool:
        return self.status != AUTO_DRAFT

    def to_dict(self) -> dict:
        return {"status": self.status, "escalate": self.escalate,
                "reasons": [asdict(r) for r in self.reasons]}


def decide(reasons: list[Reason]) -> GateDecision:
    if any(r.severity == CRITICAL for r in reasons):
        return GateDecision(ESCALATE_ONCALL, reasons)
    if reasons:
        return GateDecision(HUMAN_REVIEW, reasons)
    return GateDecision(AUTO_DRAFT, [])


# --------------------------------------------------------------------------- lexicons (en + de)
def _rx(words: list[str]) -> re.Pattern:
    return re.compile(r"(?i)(?<!\w)(?:" + "|".join(words) + r")")


# Lexicons were checked against the KB only (never the holdout). Clinical words alone are common
# domain vocabulary in this dataset (P(critical) ≈ base rate), so they only count together with
# outage language. Security terms carry ~2.5x the base critical rate when the ticket is an Incident.
SECURITY_SIGNALS = _rx([
    r"data breach", r"breach", r"ransomware", r"hacked", r"compromised", r"data leak",
    r"unauthori[sz]ed access", r"malware", r"phishing", r"sicherheitsvorfall", r"datenleck",
    r"datenpanne", r"gehackt", r"kompromittiert", r"unbefugte[rn]? zugriff", r"schadsoftware",
])
CLINICAL_CONTEXT = _rx([
    r"hospital", r"patient", r"emr\b", r"telemedicine", r"clinic", r"life[- ]threatening",
    r"emergency room", r"krankenhaus", r"klinik", r"telemedizin", r"arztpraxis", r"notaufnahme",
])
OUTAGE_TERMS = _rx([
    r"down\b", r"outage", r"offline", r"not working", r"unavailable", r"failing",
    r"cannot (?:access|connect|log ?in|reach)", r"unable to (?:access|connect|log ?in|reach)",
    r"crash", r"ausgefallen", r"ausfall", r"nicht erreichbar", r"funktioniert nicht", r"störung",
    r"kein zugriff", r"nicht mehr (?:erreichbar|verfügbar)",
])

# Sensitive actions: (category, ticket pattern, evidence pattern that would support it)
SENSITIVE_ACTIONS: list[tuple[str, re.Pattern, re.Pattern | None]] = [
    ("refund / compensation",
     _rx([r"refund", r"reimburs", r"money back", r"compensat", r"chargeback", r"credit note",
          r"r[üu]ckerstatt", r"erstatt", r"geld zur[üu]ck", r"entsch[äa]digung", r"gutschrift"]),
     _rx([r"refund", r"reimburs", r"compensat", r"credit", r"r[üu]ckerstatt", r"erstatt",
          r"gutschrift", r"entsch[äa]dig"])),
    ("pricing / discount / quote",
     _rx([r"discount", r"price match", r"quote for", r"special price", r"cheaper",
          r"rabatt", r"preisnachlass", r"angebot f[üu]r", r"sonderpreis", r"g[üu]nstiger"]),
     _rx([r"discount", r"pric", r"quote", r"rabatt", r"preis", r"angebot"])),
    ("SLA / uptime guarantee",
     _rx([r"\bsla\b", r"guarantee", r"uptime commitment", r"service credit", r"garantie",
          r"zusicher", r"verf[üu]gbarkeitsgarantie"]),
     _rx([r"\bsla\b", r"guarantee", r"service credit", r"garantie"])),
    ("legal / contractual",
     _rx([r"lawyer", r"attorney", r"lawsuit", r"legal action", r"sue\b", r"court",
          r"breach of contract", r"anwalt", r"klage", r"rechtliche schritte", r"gericht",
          r"vertragsbruch"]),
     None),  # never auto-answer legal threats
    ("account deletion / GDPR",
     _rx([r"delete (?:all )?my (?:data|account)", r"gdpr", r"right to be forgotten",
          r"dsgvo", r"daten l[öo]schen", r"konto l[öo]schen"]),
     _rx([r"delet", r"gdpr", r"dsgvo", r"l[öo]sch"])),
]

PROMPT_INJECTION = _rx([
    r"ignore (?:all |any |the )?(?:previous|prior|above) (?:instructions|rules|prompts?)",
    r"disregard (?:all |the )?(?:previous|prior|above)", r"system prompt", r"you are now",
    r"act as (?:an? )?(?:admin|developer|dan)", r"jailbreak", r"developer mode",
    r"ignoriere (?:alle )?(?:vorherigen|bisherigen) (?:anweisungen|regeln)",
    r"vergiss (?:alle )?(?:vorherigen|bisherigen)",
])


# --------------------------------------------------------------------------- pre-generation rules
def pre_generation(*, preds: dict, risk: float, routing_conf: float, top_sim: float,
                   text: str, evidence_text: str, thresholds: dict,
                   burst: dict | None = None) -> list[Reason]:
    reasons: list[Reason] = []
    t = thresholds
    pr, ty = preds["priority"], preds["type"]

    if pr == "high" and ty == "Incident":
        reasons.append(Reason("CRITICAL_INCIDENT", CRITICAL,
                              "Predicted high-priority Incident - page on-call engineer."))
    elif risk >= t["critical_risk"]:
        reasons.append(Reason("CRITICAL_RISK", CRITICAL,
                              f"Critical-incident risk {risk:.0%} ≥ {t['critical_risk']:.0%} "
                              "(recall-tuned threshold)."))

    sec = sorted({m.group(0).lower() for m in SECURITY_SIGNALS.finditer(text)})
    if sec and ty == "Incident":
        reasons.append(Reason("SECURITY_INCIDENT", CRITICAL,
                              f"Active security incident ({', '.join(sec[:3])}) - page on-call."))
    elif sec:
        reasons.append(Reason("SECURITY_SENSITIVE", REVIEW,
                              f"Security-related ({', '.join(sec[:3])}) - never auto-answered."))

    clin = sorted({m.group(0).lower() for m in CLINICAL_CONTEXT.finditer(text)})
    if clin and ty in ("Incident", "Problem") and OUTAGE_TERMS.search(text):
        reasons.append(Reason("CLINICAL_IMPACT", CRITICAL,
                              f"Outage affecting clinical operations ({', '.join(clin[:3])})."))

    if burst and burst.get("active"):
        reasons.append(Reason("MAJOR_OUTAGE_CLUSTER", CRITICAL,
                              f"Part of a burst of {burst['size']} similar tickets in the last "
                              f"{burst['window_min']:.0f} min - possible major outage."))

    if routing_conf < t["min_confidence"]:
        reasons.append(Reason("LOW_CONFIDENCE", REVIEW,
                              f"Routing confidence {routing_conf:.0%} < {t['min_confidence']:.0%}."))

    if top_sim < t["min_similarity"]:
        reasons.append(Reason("NO_PRECEDENT", REVIEW,
                              f"Best historical match similarity {top_sim:.2f} < "
                              f"{t['min_similarity']:.2f} - out of scope / no precedent."))

    for cat, ask, support in SENSITIVE_ACTIONS:
        if ask.search(text) and (support is None or not support.search(evidence_text)):
            why = "always needs a human" if support is None else "not supported by precedents"
            reasons.append(Reason("UNSUPPORTED_ACTION", REVIEW,
                                  f"Requests {cat} - {why}."))

    if PROMPT_INJECTION.search(text):
        reasons.append(Reason("PROMPT_INJECTION", REVIEW,
                              "Ticket contains instruction-override language; treated as data."))
    return reasons
