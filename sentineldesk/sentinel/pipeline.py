"""End-to-end SentinelDesk pipeline: one call per incoming ticket."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import config
from .gate import GateSettings, assess_risk, decide, post_check
from .generator import DraftGenerator
from .retrieval import Retriever
from .textutils import build_text, clean_str, detect_language, mask_pii
from .triage import TriageModel


@dataclass
class RunSettings(GateSettings):
    top_k: int = config.TOP_K
    mask_pii: bool = True
    use_llm: bool = True


class SentinelDesk:
    def __init__(self, artifact_dir: str | Path = config.ARTIFACT_DIR) -> None:
        d = Path(artifact_dir)
        if not (d / "triage.joblib").exists():
            raise FileNotFoundError(f"No trained models in {d}. Run `python train.py` first.")
        self.meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        self.triage = TriageModel.load(d / "triage.joblib")
        self.retriever = Retriever.load(d)
        self.generator = DraftGenerator()

    def default_settings(self) -> RunSettings:
        sim = self.meta.get("sim_threshold", config.SIM_THRESHOLD[self.retriever.backend])
        if config.SIM_THRESHOLD_OVERRIDE:
            sim = float(config.SIM_THRESHOLD_OVERRIDE)
        return RunSettings(
            min_confidence=config.MIN_TRIAGE_CONFIDENCE,
            sim_threshold=sim,
            critical_threshold=self.triage.critical_threshold,
        )

    def run(self, subject, body, settings: RunSettings | None = None) -> dict:
        s = settings or self.default_settings()
        subject, body = clean_str(subject), clean_str(body)
        timings: dict[str, float] = {}
        start = lap = time.perf_counter()

        def tick(name: str) -> None:
            nonlocal lap
            now = time.perf_counter()
            timings[name] = round((now - lap) * 1000, 1)
            lap = now

        text = build_text(subject, body)
        if not text:
            raise ValueError("A ticket needs a subject or a body.")
        language = detect_language(text)
        tick("preprocess")

        qvec = self.retriever.encode([text])
        tick("embed")

        knn = self.retriever.label_votes(qvec, self.triage.classes()) if self.triage.knn_weight else None
        triage = self.triage.predict([text], knn=knn)[0]
        tick("triage")

        hits, _ = self.retriever.search(
            text, k=s.top_k, queue=triage["queue"]["label"], ticket_type=triage["type"]["label"], qvec=qvec
        )
        tick("retrieval")

        risk = assess_risk(text, hits)
        decision = decide(triage, hits, risk, s)
        tick("gate")

        # PII guardrail: nothing leaves the machine unmasked.
        pii: dict[str, int] = {}
        llm_subject, llm_body, llm_hits = subject, body, hits
        if s.mask_pii:
            llm_subject, found_s = mask_pii(subject)
            llm_body, found_b = mask_pii(body)
            for found in (found_s, found_b):
                for k, v in found.items():
                    pii[k] = pii.get(k, 0) + v
            llm_hits = [
                {**h, **{f: mask_pii(h[f])[0] for f in ("subject", "body", "answer")}} for h in hits
            ]

        draft = self.generator.generate(llm_subject, llm_body, language, llm_hits, decision, use_llm=s.use_llm)
        decision = post_check(decision, draft)
        tick("draft")

        timings["total"] = round((time.perf_counter() - start) * 1000, 1)
        return {
            "language": language,
            "triage": triage,
            "decision": decision.to_dict(),
            "risk": asdict(risk),
            "hits": hits,
            "draft": draft,
            "pii_masked": pii,
            "latency_ms": timings,
            "settings": asdict(s),
            "query_vec": qvec,  # used by the burst detector; remove before serialising
        }