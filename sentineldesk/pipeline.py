"""End-to-end SentinelDesk pipeline: raw ticket -> triage -> gate -> precedents -> grounded draft."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

import joblib
import numpy as np

from . import config, embeddings, gate, lang, pii
from .burst import BurstDetector
from .classifier import KNN_K, TriageModel
from .data import clean, ticket_text
from .generator import (MAX_TOKENS, Draft, build_prompt, build_translation_prompt,
                        clean_translation, finalize)
from .llm import LLMChain, LLMError, get_llm
from .retriever import Precedent, Retriever

DEFAULT_THRESHOLDS = {"min_confidence": 0.35, "min_similarity": 0.55, "critical_risk": 0.30}

ESCALATION_NOTE = {
    gate.ESCALATE_ONCALL: ("\nCONTEXT: This ticket has been escalated to the on-call engineering "
                           "team. Acknowledge the urgency and say an engineer is already engaged "
                           "(no resolution time). Then, if the resolutions contain safe steps the "
                           "customer can take meanwhile, list them briefly with citations.\n"),
}


@dataclass
class TriageResult:
    ref: str
    subject: str
    body: str
    language: str
    language_confidence: float
    reply_language: str
    predictions: dict            # head -> {"label", "confidence", "probs"}
    routing_confidence: float
    critical_risk: float
    precedents: list[Precedent]
    gate: gate.GateDecision
    burst: dict
    pii: dict
    draft: Draft | None
    llm_error: str | None
    latency_ms: dict = field(default_factory=dict)
    # internal: pre-generation gate reasons + the PII-masked text the LLM is allowed to see
    pre_reasons: list = field(default_factory=list, repr=False)
    masked_subject: str = field(default="", repr=False)
    masked_body: str = field(default="", repr=False)

    @property
    def pre_status(self) -> str:
        return gate.decide(self.pre_reasons).status

    def to_dict(self) -> dict:
        return {
            "ref": self.ref, "language": self.language, "reply_language": self.reply_language,
            "language_confidence": self.language_confidence,
            "predictions": self.predictions, "routing_confidence": self.routing_confidence,
            "critical_risk": self.critical_risk,
            "precedents": [p.to_dict() for p in self.precedents],
            "gate": self.gate.to_dict(), "burst": self.burst, "pii": self.pii,
            "draft": self.draft.to_dict() if self.draft else None,
            "llm_error": self.llm_error, "latency_ms": self.latency_ms,
        }


class SentinelDesk:
    def __init__(self, llm: LLMChain | None = None, load_llm: bool = True):
        self.model: TriageModel = joblib.load(config.CLASSIFIER_PATH)
        self.retriever = Retriever.load()
        self.model.set_neighbor_labels(self.retriever.kb)
        self.burst = BurstDetector()
        self.thresholds = {**DEFAULT_THRESHOLDS, **self.model.gate_thresholds}
        for key, env in (("min_confidence", config.GATE_MIN_CONFIDENCE),
                         ("min_similarity", config.GATE_MIN_SIMILARITY),
                         ("critical_risk", config.GATE_CRITICAL_RISK)):
            if env is not None:
                self.thresholds[key] = float(env)
        self._llm = llm
        self._load_llm = load_llm
        embeddings.warmup()  # load + warm the query encoder so the first ticket isn't slow

    @property
    def llm(self) -> LLMChain | None:
        if self._llm is None and self._load_llm:
            try:
                self._llm = get_llm()
            except LLMError:
                self._llm = None
        return self._llm

    # ------------------------------------------------------------------ core
    def classify(self, subject: str, body: str, emb: np.ndarray, sims, idx):
        probs = self.model.predict_proba([ticket_text(subject, body)], emb[None, :], sims, idx)
        preds = {}
        for head, p in probs.items():
            p = p[0]
            classes = self.model.heads[head].classes
            order = np.argsort(-p)
            preds[head] = {
                "label": str(classes[order[0]]),
                "confidence": float(p[order[0]]),
                "probs": {str(classes[i]): float(p[i]) for i in order},
            }
        return preds

    def triage(self, subject: str | None, body: str, *, generate: bool = True,
               translate: bool = False, ref: str | None = None,
               now: float | None = None) -> TriageResult:
        """Full pipeline. Use analyze() + draft_stream() to stream the draft instead."""
        r = self.analyze(subject, body, ref=ref, now=now)
        if generate:
            for _ in self.draft_stream(r):
                pass
            if translate:
                for _ in self.translate_stream(r):
                    pass
        return r

    def analyze(self, subject: str | None, body: str, *, ref: str | None = None,
                now: float | None = None) -> TriageResult:
        """Everything except generation: language, embed, retrieve, classify, burst, gate, PII."""
        t0 = time.perf_counter()
        lat: dict[str, float] = {}

        def lap(name: str, start: float) -> float:
            t = time.perf_counter()
            lat[name] = round((t - start) * 1000, 1)
            return t

        ref = ref or f"LIVE-{uuid.uuid4().hex[:6].upper()}"
        subject, body = clean(subject), clean(body)
        if not body and not subject:
            raise ValueError("ticket needs a subject or a body")
        text = ticket_text(subject, body)
        detected, reply_lang, lconf = lang.reply_language(text)
        t = lap("language", t0)

        emb = embeddings.encode_one(subject, body)
        t = lap("embed", t)

        sims, idx = self.retriever.search_raw(emb, max(KNN_K, config.TOP_K))
        precedents = self.retriever.precedents(sims[0, :config.TOP_K], idx[0, :config.TOP_K])
        t = lap("retrieve", t)

        preds = self.classify(subject, body, emb, sims, idx)
        risk = preds["critical"]["probs"].get("critical", 0.0)
        routing_conf = float(np.sqrt(preds["type"]["confidence"] * preds["queue"]["confidence"]))
        t = lap("classify", t)

        burst = self.burst.observe(emb, ref=ref, subject=subject or body[:60],
                                   queue=preds["queue"]["label"], type_=preds["type"]["label"],
                                   now=now)
        top_sim = precedents[0].similarity if precedents else 0.0
        evidence = " ".join(p.answer for p in precedents)
        reasons = gate.pre_generation(
            preds={h: v["label"] for h, v in preds.items()}, risk=risk,
            routing_conf=routing_conf, top_sim=top_sim, text=text, evidence_text=evidence,
            thresholds=self.thresholds, burst=burst)
        t = lap("gate", t)

        masked_subject, masked_body = pii.mask(subject), pii.mask(body)
        pii_info = {"masked": masked_subject.total + masked_body.total,
                    "counts": {k: masked_subject.counts.get(k, 0) + masked_body.counts.get(k, 0)
                               for k in {**masked_subject.counts, **masked_body.counts}}}
        lat["total"] = round((time.perf_counter() - t0) * 1000, 1)
        lat["triage"] = lat["total"]
        return TriageResult(
            ref=ref, subject=subject, body=body, language=detected, language_confidence=lconf,
            reply_language=reply_lang, predictions=preds, routing_confidence=routing_conf,
            critical_risk=risk, precedents=precedents, gate=gate.decide(reasons), burst=burst,
            pii=pii_info, draft=None, llm_error=None, latency_ms=lat, pre_reasons=reasons,
            masked_subject=masked_subject.text, masked_body=masked_body.text)

    def draft_stream(self, r: TriageResult) -> Iterator[str]:
        """Stream Gemma's raw output; on completion run grounding checks and update `r` in place
        (draft, final gate decision, latency)."""
        t0 = time.perf_counter()
        draft, err = None, None
        llm = self.llm
        if llm is None:
            err = "No Gemma backend available"
        else:
            prompt = build_prompt(subject=r.masked_subject, body=r.masked_body,
                                  reply_lang=r.reply_language, precedents=r.precedents,
                                  escalation_note=ESCALATION_NOTE.get(r.pre_status, ""))
            stream = llm.stream(prompt, max_tokens=MAX_TOKENS, temperature=0.1)
            try:
                yield from stream
                draft = finalize(stream.text, subject=r.masked_subject, body=r.masked_body,
                                 reply_lang=r.reply_language, precedents=r.precedents,
                                 backend=stream.backend, model=stream.model,
                                 latency_ms=stream.latency_ms)
            except LLMError as e:
                err = str(e)
        reasons = list(r.pre_reasons)
        if draft:
            reasons += draft.reasons
        if err:
            reasons.append(gate.Reason("LLM_UNAVAILABLE", gate.REVIEW,
                                       "Draft could not be generated; agent must reply."))
        r.draft, r.llm_error, r.gate = draft, err, gate.decide(reasons)
        r.latency_ms["generate"] = round((time.perf_counter() - t0) * 1000, 1)
        r.latency_ms["total"] = round(r.latency_ms["triage"] + r.latency_ms["generate"], 1)

    @staticmethod
    def needs_translation(r: TriageResult) -> bool:
        return bool(r.draft and r.draft.reply and r.reply_language != "en")

    def translate_stream(self, r: TriageResult) -> Iterator[str]:
        """Stream an operator-only English translation of a non-English draft. The customer-facing
        reply and its grounding checks are untouched; the result lands in r.draft.translation_en."""
        if not self.needs_translation(r) or self.llm is None:
            return
        t0 = time.perf_counter()
        prompt = build_translation_prompt(r.draft.reply, r.reply_language)
        stream = self.llm.stream(prompt, max_tokens=MAX_TOKENS, temperature=0.0)
        try:
            yield from stream
            r.draft.translation_en = clean_translation(stream.text)
        except LLMError as e:
            r.draft.translation_error = str(e)
        r.latency_ms["translate"] = round((time.perf_counter() - t0) * 1000, 1)
