"""Holdout evaluation (20% split). The only place where holdout labels are read - for scoring.

Inference sees test subject+body only (and their cached EmbeddingGemma vectors). Holdout
answers were never persisted. Retrieval searches the KB index only.

    uv run python scripts/evaluate.py              # triage, gate, retrieval, latency
    uv run python scripts/evaluate.py --gen 30     # + grounded-generation audit on 30 tickets
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.metrics import classification_report, f1_score

from sentineldesk import config, data, gate
from sentineldesk.classifier import HEADS, KNN_K, critical_labels
from sentineldesk.pipeline import SentinelDesk


def ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=0, help="audit grounded generation on N tickets")
    ap.add_argument("--latency-n", type=int, default=200)
    args = ap.parse_args()

    desk = SentinelDesk(load_llm=args.gen > 0)
    model, retr = desk.model, desk.retriever
    test = data.load_test_inputs()
    emb = np.load(config.TEST_EMB_PATH)
    assert len(emb) == len(test)
    texts = [data.ticket_text(s, b) for s, b in zip(test["subject"], test["body"])]

    sims, idx = retr.search_raw(emb, max(KNN_K, config.TOP_K))
    probs = model.predict_proba(texts, emb, sims, idx)

    # --------------------------------------------------- labels: scoring only from here on
    lab = data.load_test_labels().set_index("ticket_id").loc[test["ticket_id"]]
    y = {h: lab[h].to_numpy() for h in ("type", "queue", "priority")}
    y["critical"] = critical_labels(lab["type"], lab["priority"])

    out: dict = {"n_test": len(test), "n_kb": len(retr.kb), "triage": {}, "ablation": {}}
    pred = {}
    X = model.vectorizer.transform(texts)
    for h in HEADS:
        head = model.heads[h]
        pred[h] = head.classes[probs[h].argmax(1)]
        y_idx = np.searchsorted(head.classes, y[h])
        out["triage"][h] = {
            "macro_f1": f1_score(y[h], pred[h], average="macro"),
            "weighted_f1": f1_score(y[h], pred[h], average="weighted"),
            "accuracy": float((pred[h] == y[h]).mean()),
            "ece": ece(probs[h], y_idx),
            "per_class": classification_report(y[h], pred[h], output_dict=True, zero_division=0),
        }
        comps = head.components(X, emb, sims[:, :KNN_K], idx[:, :KNN_K], model.neighbor_labels[h])
        out["ablation"][h] = {
            name: f1_score(y[h], head.classes[c.argmax(1)], average="macro")
            for name, c in zip(("tfidf_lr", "embeddinggemma_lr", "knn_precedents"), comps)
        } | {"ensemble": out["triage"][h]["macro_f1"],
             "ensemble_uncalibrated_ece": ece(head.blend(comps, temperature=1.0), y_idx)}

    # --------------------------------------------------- robustness: novel vs near-duplicate
    # The synthetic dataset contains paraphrased / translated near-duplicates, which the kNN
    # component exploits. Report metrics on holdout tickets with NO close KB neighbour too.
    top1 = sims[:, 0]
    out["robustness"] = {"top1_sim_quantiles": {q: float(np.quantile(top1, q))
                                                for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    for name, mask in (("near_duplicate_top1>=0.95", top1 >= 0.95),
                       ("novel_top1<0.90", top1 < 0.90), ("novel_top1<0.85", top1 < 0.85)):
        out["robustness"][name] = {"share": float(mask.mean())} | {
            h: {"ensemble": f1_score(y[h][mask], pred[h][mask], average="macro"),
                "tfidf_lr": f1_score(y[h][mask], model.heads[h].classes[
                    model.heads[h].lr_tfidf.predict_proba(X[mask]).argmax(1)], average="macro")}
            for h in ("type", "queue", "priority")}

    # --------------------------------------------------- high-priority incident recall
    is_crit = y["critical"] == "critical"
    joint = (pred["priority"] == "high") & (pred["type"] == "Incident")
    risk = probs["critical"][:, list(model.heads["critical"].classes).index("critical")]
    gate_flag = joint | (risk >= desk.thresholds["critical_risk"])
    is_high = y["priority"] == "high"
    out["high_priority_incident"] = {
        "support": int(is_crit.sum()),
        "joint_label_recall": float(joint[is_crit].mean()),
        "joint_label_precision": float(is_crit[joint].mean()),
        "gate_escalation_recall": float(gate_flag[is_crit].mean()),
        "gate_escalation_precision": float(is_crit[gate_flag].mean()),
        "gate_escalation_rate": float(gate_flag.mean()),
        "high_priority_recall": float((pred["priority"] == "high")[is_high].mean()),
    }

    # --------------------------------------------------- full gate (pre-generation) on holdout
    statuses, codes = [], {}
    for i in range(len(test)):
        p = {h: pred[h][i] for h in HEADS}
        conf_t = float(probs["type"][i].max())
        conf_q = float(probs["queue"][i].max())
        ev = " ".join(retr.kb.answer.iloc[j] for j in idx[i, :config.TOP_K])
        reasons = gate.pre_generation(preds=p, risk=float(risk[i]),
                                      routing_conf=float(np.sqrt(conf_t * conf_q)),
                                      top_sim=float(sims[i, 0]), text=texts[i], evidence_text=ev,
                                      thresholds=desk.thresholds)
        statuses.append(gate.decide(reasons).status)
        for r in reasons:
            codes[r.code] = codes.get(r.code, 0) + 1
    statuses = np.array(statuses)
    auto = statuses == gate.AUTO_DRAFT
    route_ok = (pred["type"] == y["type"]) & (pred["queue"] == y["queue"])
    out["gate"] = {
        "thresholds": desk.thresholds,
        "status_rates": {s: float((statuses == s).mean()) for s in
                         (gate.AUTO_DRAFT, gate.HUMAN_REVIEW, gate.ESCALATE_ONCALL)},
        "reason_counts": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
        "critical_missed_by_gate": float((auto & is_crit).sum() / max(is_crit.sum(), 1)),
        "auto_draft_routing_accuracy": float(route_ok[auto].mean()) if auto.any() else None,
        "overall_routing_accuracy": float(route_ok.mean()),
    }

    # --------------------------------------------------- retrieval proxy quality
    k = config.TOP_K
    nb = idx[:, :k]
    kb = retr.kb
    same_lang = kb.language.to_numpy()[nb] == lab["language"].to_numpy()[:, None]
    out["retrieval"] = {
        f"queue_precision@{k}": float((kb.queue.to_numpy()[nb] == y["queue"][:, None]).mean()),
        f"type_precision@{k}": float((kb.type.to_numpy()[nb] == y["type"][:, None]).mean()),
        "top1_similarity_median": float(np.median(sims[:, 0])),
        "cross_lingual_share": float(1 - same_lang.mean()),
    }

    # --------------------------------------------------- latency (no LLM)
    n = min(args.latency_n, len(test))
    lats = []
    for s, b in zip(test["subject"][:n], test["body"][:n]):
        t0 = time.perf_counter()
        desk.triage(s, b, generate=False)
        lats.append((time.perf_counter() - t0) * 1000)
    desk.burst.reset()
    out["latency_ms_triage_no_llm"] = {"p50": float(np.percentile(lats, 50)),
                                       "p95": float(np.percentile(lats, 95)), "n": n}

    # --------------------------------------------------- generation audit (optional)
    if args.gen:
        rng = np.random.default_rng(config.SEED)
        rows = rng.choice(len(test), size=min(args.gen, len(test)), replace=False)
        audit = []
        for i in rows:
            r = desk.triage(test.subject.iloc[i], test.body.iloc[i], generate=True)
            d = r.draft
            audit.append({
                "ticket_id": test.ticket_id.iloc[i], "ok": d is not None,
                "has_valid_citation": bool(d and d.cited),
                "hallucinated_citation": bool(d and d.invalid_citations),
                "unsupported_claim": bool(d and d.unsupported_claims),
                "language_match": bool(d and not any(x.code == "LANGUAGE_MISMATCH"
                                                     for x in d.reasons)),
                "self_grounding": d.grounding if d else None,
                "status": r.gate.status, "latency_ms": r.latency_ms["total"],
                "backend": f"{d.backend}:{d.model}" if d else r.llm_error,
            })
            print(f"  gen {len(audit)}/{len(rows)} {audit[-1]['ticket_id']} "
                  f"{audit[-1]['latency_ms']:.0f}ms cited={audit[-1]['has_valid_citation']}",
                  flush=True)
        ok = [a for a in audit if a["ok"]]
        out["generation"] = {
            "n": len(audit), "n_generated": len(ok),
            "valid_citation_rate": float(np.mean([a["has_valid_citation"] for a in ok])) if ok else None,
            "hallucinated_citation_rate": float(np.mean([a["hallucinated_citation"] for a in ok])) if ok else None,
            "unsupported_claim_rate": float(np.mean([a["unsupported_claim"] for a in ok])) if ok else None,
            "language_match_rate": float(np.mean([a["language_match"] for a in ok])) if ok else None,
            "latency_ms_p50": float(np.median([a["latency_ms"] for a in ok])) if ok else None,
            "backend": ok[0]["backend"] if ok else None,
            "samples": audit,
        }

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps(out, indent=2, default=float))
    print_summary(out)


def print_summary(o: dict) -> None:
    print("\n=== Holdout triage (n=%d) ===" % o["n_test"])
    print(f"{'head':10s} {'macroF1':>8s} {'acc':>7s} {'ECE':>6s} | {'tfidf':>6s} {'emb':>6s} {'knn':>6s}")
    for h, m in o["triage"].items():
        a = o["ablation"][h]
        print(f"{h:10s} {m['macro_f1']:8.3f} {m['accuracy']:7.3f} {m['ece']:6.3f} | "
              f"{a['tfidf_lr']:6.3f} {a['embeddinggemma_lr']:6.3f} {a['knn_precedents']:6.3f}")
    print("\n=== High-priority Incident ===")
    print(json.dumps(o["high_priority_incident"], indent=2))
    print("\n=== Gate ===")
    print(json.dumps({k: v for k, v in o["gate"].items()}, indent=2))
    print("\n=== Robustness (novel vs near-duplicate holdout tickets) ===")
    print(json.dumps(o["robustness"], indent=2))
    print("\n=== Retrieval ===")
    print(json.dumps(o["retrieval"], indent=2))
    print("\n=== Latency (triage, no LLM) ===", o["latency_ms_triage_no_llm"])
    if "generation" in o:
        g = {k: v for k, v in o["generation"].items() if k != "samples"}
        print("\n=== Generation audit ===")
        print(json.dumps(g, indent=2))


if __name__ == "__main__":
    main()
