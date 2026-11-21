"""Evaluate SentinelDesk on the 20% holdout split.

Predictions are made from subject + body only. Holdout labels are loaded after all
predictions exist and are used only to score them. Holdout answers were never saved.

Usage:
    python evaluate.py                       # triage metrics + 200-ticket end-to-end sample
    python evaluate.py --pipeline-sample 500
    python evaluate.py --with-llm            # include LLM drafting in the latency sample
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sentinel import config
from sentinel.data import read_csv
from sentinel.pipeline import SentinelDesk
from sentinel.textutils import OUTAGE_PATTERNS, build_text, find_terms


def expected_calibration_error(y_true, y_pred, conf, bins: int = 10) -> float:
    y_true, y_pred, conf = np.asarray(y_true), np.asarray(y_pred), np.asarray(conf)
    correct = (y_true == y_pred).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline-sample", type=int, default=200)
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    desk = SentinelDesk()
    inputs = read_csv(config.DATA_DIR / "test_inputs.csv")

    # Leakage guard: no holdout ticket may exist in the precedent index.
    overlap = set(inputs["ticket_id"]) & set(desk.retriever.meta["ticket_id"])
    assert not overlap, f"{len(overlap)} holdout tickets found in the index"

    texts = [build_text(s, b) for s, b in zip(inputs["subject"], inputs["body"])]
    t0 = time.perf_counter()
    preds = desk.triage.predict(texts)
    triage_ms = (time.perf_counter() - t0) * 1000 / max(1, len(texts))

    # Labels are read only now, after every prediction has been made.
    labels = read_csv(config.DATA_DIR / "test_labels.csv").set_index("ticket_id").loc[inputs["ticket_id"]]

    report: dict = {"n_test": len(inputs), "triage": {}, "triage_ms_per_ticket": round(triage_ms, 2)}
    for t in config.TARGETS:
        y_true = labels[t].to_numpy()
        y_pred = [p[t]["label"] for p in preds]
        conf = [p[t]["confidence"] for p in preds]
        report["triage"][t] = {
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "ece": round(expected_calibration_error(y_true, y_pred, conf), 4),
            "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        }

    # High-priority incident recall, measured two ways.
    is_hpi = ((labels["type"].str.lower() == "incident") & (labels["priority"].str.lower() == "high")).to_numpy()
    strict = np.array([p["type"]["label"].lower() == "incident" and p["priority"]["label"].lower() == "high"
                       for p in preds])
    detector = np.array([p["critical_score"] >= desk.triage.critical_threshold for p in preds])
    outage = np.array([bool(find_terms(x, OUTAGE_PATTERNS, limit=1)) for x in texts])
    on_call = strict | detector | outage
    n_hpi = int(is_hpi.sum())
    report["high_priority_incident"] = {
        "support": n_hpi,
        "recall_predicted_labels": round(float(strict[is_hpi].mean()) if n_hpi else 0.0, 4),
        "recall_on_call_gate": round(float(on_call[is_hpi].mean()) if n_hpi else 0.0, 4),
        "precision_on_call_gate": round(float(is_hpi[on_call].mean()) if on_call.any() else 0.0, 4),
        "on_call_rate": round(float(on_call.mean()), 4),
    }
    is_high = (labels["priority"].str.lower() == "high").to_numpy()
    pred_high = np.array([p["priority"]["label"].lower() == "high" for p in preds])
    report["high_priority_recall"] = round(float(pred_high[is_high].mean()) if is_high.any() else 0.0, 4)

    # Coverage vs. accuracy for the confidence threshold (helps choose the gate setting).
    overall = np.array([p["confidence"] for p in preds])
    joint = np.array([all(p[t]["label"] == labels[t].iloc[i] for t in config.TARGETS) for i, p in enumerate(preds)])
    queue_ok = np.array([p["queue"]["label"] == labels["queue"].iloc[i] for i, p in enumerate(preds)])
    report["confidence_coverage"] = [
        {"threshold": thr, "coverage": round(float((overall >= thr).mean()), 4),
         "queue_accuracy": round(float(queue_ok[overall >= thr].mean()), 4) if (overall >= thr).any() else None,
         "all_three_correct": round(float(joint[overall >= thr].mean()), 4) if (overall >= thr).any() else None}
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7)
    ]

    # End-to-end sample: full pipeline latency and escalation mix.
    sample = inputs.sample(n=min(args.pipeline_sample, len(inputs)), random_state=0)
    settings = desk.default_settings()
    settings.use_llm = args.with_llm
    latencies, statuses = [], Counter()
    for row in sample.itertuples():
        result = desk.run(row.subject, row.body, settings)
        latencies.append(result["latency_ms"]["total"])
        statuses[result["decision"]["status"]] += 1
    report["end_to_end"] = {
        "n": len(sample),
        "llm": args.with_llm and desk.generator.provider != "template",
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "status_rates": {k: round(v / len(sample), 4) for k, v in statuses.items()},
    }

    out = config.ARTIFACT_DIR / "eval_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== SentinelDesk holdout evaluation ===")
    print(f"Holdout tickets: {report['n_test']}")
    for t in config.TARGETS:
        m = report["triage"][t]
        print(f"  {t:<9} macro-F1 {m['macro_f1']:.3f}   accuracy {m['accuracy']:.3f}   ECE {m['ece']:.3f}")
    h = report["high_priority_incident"]
    print(f"High-priority incidents: {h['support']}")
    print(f"  recall from predicted labels   {h['recall_predicted_labels']:.3f}")
    print(f"  recall of the on-call gate     {h['recall_on_call_gate']:.3f} "
          f"(precision {h['precision_on_call_gate']:.3f}, pages {h['on_call_rate']:.1%} of tickets)")
    print("Confidence threshold coverage:")
    for row in report["confidence_coverage"]:
        print(f"  ≥{row['threshold']:.1f}: covers {row['coverage']:.1%}, queue accuracy {row['queue_accuracy']}, "
              f"all three correct {row['all_three_correct']}")
    e = report["end_to_end"]
    print(f"End-to-end ({e['n']} tickets, LLM={e['llm']}): p50 {e['latency_p50_ms']} ms, "
          f"p95 {e['latency_p95_ms']} ms, status mix {e['status_rates']}")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
