"""Escalation and safety gate.

Explicit, explainable rules decide whether a ticket can get an AI draft or must
go to a human. Every decision carries human-readable reasons for the UI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import config
from .textutils import INJECTION_PATTERNS, OUTAGE_PATTERNS, RISKY_ACTIONS, find_terms

ON_CALL = "ESCALATE_ON_CALL"
REVIEW = "HUMAN_REVIEW"
AUTO = "AUTO_DRAFT_READY"
STATUS_LABEL = {
    ON_CALL: "Escalate to on-call engineer",
    REVIEW: "Human review required",
    AUTO: "Auto-draft ready",
}


@dataclass
class GateSettings:
    min_confidence: float = config.MIN_TRIAGE_CONFIDENCE
    sim_threshold: float = 0.55
    critical_threshold: float = 0.5
    use_critical_detector: bool = True
    use_outage_keywords: bool = True


@dataclass
class RiskSignals:
    injection: list[str] = field(default_factory=list)
    outage: list[str] = field(default_factory=list)
    requested_actions: dict[str, list[str]] = field(default_factory=dict)
    unsupported_actions: list[str] = field(default_factory=list)


@dataclass
class GateDecision:
    status: str
    reasons: list[str]

    @property
    def label(self) -> str:
        return STATUS_LABEL[self.status]

    @property
    def escalate(self) -> bool:
        return self.status != AUTO

    def to_dict(self) -> dict:
        return {**asdict(self), "label": self.label, "escalate": self.escalate}


def assess_risk(text: str, hits: list[dict]) -> RiskSignals:
    """Find injection attempts, outage language, and risky requests without precedent."""
    requested = {cat: terms for cat, pats in RISKY_ACTIONS.items() if (terms := find_terms(text, pats))}
    evidence = " ".join(h.get("answer", "") for h in hits)
    unsupported = [cat for cat in requested if not find_terms(evidence, RISKY_ACTIONS[cat], limit=1)]
    return RiskSignals(
        injection=find_terms(text, INJECTION_PATTERNS),
        outage=find_terms(text, OUTAGE_PATTERNS),
        requested_actions=requested,
        unsupported_actions=unsupported,
    )


def decide(triage: dict, hits: list[dict], risk: RiskSignals, s: GateSettings) -> GateDecision:
    on_call: list[str] = []
    review: list[str] = []

    # Rule 1: high-priority incidents page the on-call engineer.
    if triage["type"]["label"].lower() == "incident" and triage["priority"]["label"].lower() == "high":
        on_call.append("Predicted high-priority incident")
    if s.use_critical_detector and triage["critical_score"] >= s.critical_threshold:
        on_call.append(
            f"Critical-incident detector fired ({triage['critical_score']:.2f} ≥ {s.critical_threshold:.2f})"
        )
    if s.use_outage_keywords and risk.outage:
        on_call.append("Outage or security language: " + ", ".join(f"“{t}”" for t in risk.outage[:3]))

    # Rule 2: adversarial input never gets an automatic answer.
    if risk.injection:
        review.append("Possible prompt-injection attempt: " + ", ".join(f"“{t}”" for t in risk.injection[:2]))

    # Rule 3: the classifier is unsure.
    if triage["confidence"] < s.min_confidence:
        review.append(f"Low triage confidence ({triage['confidence']:.2f} < {s.min_confidence:.2f})")

    # Rule 4: nothing similar has been solved before.
    top = hits[0]["similarity"] if hits else 0.0
    if top < s.sim_threshold:
        review.append(f"No close historical precedent (best match {top:.2f} < {s.sim_threshold:.2f})")

    # Rule 5: the customer asks for something no precedent supports.
    for cat in risk.unsupported_actions:
        review.append(f"Asks for {cat}, which no retrieved resolution supports")

    if on_call:
        return GateDecision(ON_CALL, on_call + review)
    if review:
        return GateDecision(REVIEW, review)
    return GateDecision(AUTO, ["Confident triage, close precedents and no risky requests"])


def post_check(decision: GateDecision, draft: dict) -> GateDecision:
    """Downgrade an auto-draft to human review if the draft fails grounding checks."""
    issues = draft.get("issues") or []
    if not issues:
        return decision
    if decision.status == AUTO:
        return GateDecision(REVIEW, ["Draft failed grounding check: " + "; ".join(issues)])
    return GateDecision(decision.status, decision.reasons + ["Draft notes: " + "; ".join(issues)])