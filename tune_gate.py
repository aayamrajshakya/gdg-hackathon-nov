"""
Tune the SentinelDesk escalation gate on Tobi-Bueck/customer-support-tickets.

    python tune_gate.py --data-dir ./customer-support-tickets            # tune only
    python tune_gate.py --data-dir ./customer-support-tickets --eval-test

Writes gate_config.json (load it with GateConfig.load) and tuning_report.json.

Leakage discipline (the rules in section 7 of the problem statement):
  * The 80/20 split is the exact snippet from the problem statement (make_split).
    Teammates should import load_and_split() from here so every ticket_id matches.
  * Everything is fit and tuned inside kb_train_df, carved three ways:
        fit   (80% of KB)  -> trains the classifier and is the retrieval index
        calib (10% of KB)  -> temperature scaling, so confidences are calibrated
        tune  (10% of KB)  -> chooses every gate threshold
  * test_df is touched only with --eval-test, and only its labels are read, for
    scoring. Its answers are never read or indexed.

The TF-IDF classifier and retriever are BASELINES. If your team ships different
models, swap them in (see predict_heads / retrieve_topk). Thresholds, especially
similarity ones, are only valid for the models they were tuned with.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd

from sentinel_gate import (
    GateConfig, GateStatus, HeadPrediction, RetrievedTicket, TriageResult,
    detect_actions, evaluate_gate, gate_report,
)

HEADS = ("type", "queue", "priority")
RECOMMENDED_FILES = (
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "dataset-tickets-multi-lang-4-20k.csv",
)
# The HF data has 5 priority levels; the spec scores high/medium/low.
PRIORITY_MAP = {"critical": "high", "urgent": "high", "highest": "high",
                "very_low": "low", "very low": "low", "lowest": "low"}


# ------------------------------------------------------------------ data loading

def is_lfs_pointer(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def find_csv(data_dir: str) -> str:
    for name in RECOMMENDED_FILES:
        hits = glob.glob(os.path.join(data_dir, "**", name), recursive=True)
        if hits:
            return hits[0]
    hits = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True),
                  key=os.path.getsize, reverse=True)
    if not hits:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    return hits[0]


def load_raw(csv_path: str) -> pd.DataFrame:
    if is_lfs_pointer(csv_path):
        raise RuntimeError(
            f"{csv_path} is a Git LFS pointer, not the data. In the cloned repo run:\n"
            "    git lfs install && git lfs pull")
    return pd.read_csv(csv_path)


def make_split(df: pd.DataFrame):
    """Exactly the snippet from the problem statement."""
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def normalize_priority(s: pd.Series) -> pd.Series:
    def f(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        x = str(x).strip().lower()
        return PRIORITY_MAP.get(x, x)
    return s.map(f)


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning happens AFTER the split so ticket_ids match the official snippet."""
    df = df.copy()
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df = df[df["body"].str.len() > 0]
    if "priority" in df:
        df["priority"] = normalize_priority(df["priority"])
    labels = [c for c in HEADS if c in df]
    df = df.dropna(subset=labels)               # unlabeled rows can't train or be scored
    for col in ("type", "queue"):
        if col in df:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_and_split(csv_path: str):
    raw = load_raw(csv_path)
    kb, test = make_split(raw)
    return clean_split(kb), clean_split(test), raw


def text_of(df: pd.DataFrame) -> list[str]:
    return (df["subject"] + "\n" + df["body"]).tolist()


# ------------------------------------------------------------------ calibration

def softmax_with_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    """Single-parameter temperature scaling: pick T that minimizes held-out NLL."""
    best_T, best_nll = 1.0, np.inf
    for T in np.logspace(-1.5, 1.3, 140):
        p = softmax_with_temperature(logits, T)[np.arange(len(y_idx)), y_idx]
        nll = -np.mean(np.log(np.clip(p, 1e-12, None)))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


# ------------------------------------------------------------- baseline models

class BaselineTriage:
    """TF-IDF + logistic regression per head, temperature-scaled on the calib split."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                   max_features=200_000, strip_accents=None)
        self.models, self.classes, self.T = {}, {}, {}

    def fit(self, df: pd.DataFrame):
        from sklearn.linear_model import LogisticRegression
        X = self.vec.fit_transform(text_of(df))
        for h in HEADS:
            clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
            clf.fit(X, df[h].values)
            self.models[h], self.classes[h] = clf, list(clf.classes_)
        return self

    def _logits(self, X, h):
        z = self.models[h].decision_function(X)
        return z if z.ndim == 2 else np.column_stack([-z, z]) / 2  # binary safety

    def calibrate(self, df: pd.DataFrame) -> dict:
        X = self.vec.transform(text_of(df))
        report = {}
        for h in HEADS:
            z = self._logits(X, h)
            idx = {c: i for i, c in enumerate(self.classes[h])}
            keep = df[h].isin(idx).values           # labels unseen in fit can't be scored
            y = np.array([idx[v] for v in df[h].values[keep]])
            before = softmax_with_temperature(z[keep], 1.0)
            self.T[h] = fit_temperature(z[keep], y)
            after = softmax_with_temperature(z[keep], self.T[h])
            report[h] = {
                "temperature": round(self.T[h], 3),
                "ece_before": round(expected_calibration_error(before.max(1), before.argmax(1) == y), 4),
                "ece_after": round(expected_calibration_error(after.max(1), after.argmax(1) == y), 4),
            }
        return report

    def predict_heads(self, texts: list[str]) -> dict[str, np.ndarray]:
        X = self.vec.transform(texts)
        return {h: softmax_with_temperature(self._logits(X, h), self.T.get(h, 1.0)) for h in HEADS}


class BaselineRetriever:
    """TF-IDF cosine retrieval over KB tickets that have an answer."""

    def __init__(self, vec, kb: pd.DataFrame):
        kb = kb[kb["answer"].notna() & (kb["answer"].astype(str).str.strip() != "")]
        self.kb = kb.reset_index(drop=True)
        self.vec = vec
        self.X = vec.transform(text_of(self.kb)).astype(np.float32).tocsr()

    def retrieve_topk(self, texts: list[str], k: int = 5, chunk: int = 256):
        Q = self.vec.transform(texts).astype(np.float32).tocsr()
        idx_out = np.zeros((len(texts), k), dtype=np.int64)
        sim_out = np.zeros((len(texts), k), dtype=np.float32)
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ self.X.T).toarray()
            top = np.argpartition(-S, kth=min(k, S.shape[1] - 1), axis=1)[:, :k]
            rows = np.arange(S.shape[0])[:, None]
            order = np.argsort(-S[rows, top], axis=1)
            idx_out[s:s + chunk] = top[rows, order]
            sim_out[s:s + chunk] = S[rows, top[rows, order]]
        return idx_out, sim_out

    def to_hits(self, idx_row, sim_row) -> list[RetrievedTicket]:
        return [RetrievedTicket(self.kb.at[i, "ticket_id"], float(s), str(self.kb.at[i, "answer"]))
                for i, s in zip(idx_row, sim_row)]


# ------------------------------------------------------------- threshold tuning

def floor4(x: float) -> float:
    """Round thresholds DOWN, so rounding never rejects a ticket the tuning accepted."""
    return float(np.floor(x * 1e4) / 1e4)


def choose_conf_threshold(conf, correct, target_accuracy=0.85, fallback_coverage=0.4) -> dict:
    """Lowest threshold whose auto-accepted set reaches target accuracy. If that keeps fewer
    than `fallback_coverage` of tickets (or the target is unreachable, common for noisy
    heads), relax to keep the most-confident `fallback_coverage` and flag it."""
    order = np.argsort(-np.asarray(conf), kind="stable")
    c, ok = np.asarray(conf)[order], np.asarray(correct, bool)[order]
    acc = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    meets = np.where(acc >= target_accuracy)[0]
    floor_i = max(int(np.ceil(fallback_coverage * len(c))) - 1, 0)
    met = bool(len(meets))
    i = meets[-1] if met else floor_i
    relaxed = i < floor_i
    i = max(i, floor_i)
    return {"threshold": float(c[i]), "coverage": (i + 1) / len(c),
            "accuracy": float(acc[i]), "met_target": met, "relaxed": relaxed or not met}


def choose_soft_threshold(joint, hard_hit, is_pos, target_recall=0.90) -> dict:
    """Largest P(high)*P(Incident) cut that, together with the hard rule, recalls
    `target_recall` of true high-priority Incidents. Larger cut = fewer false pages."""
    joint, hard_hit, is_pos = map(np.asarray, (joint, hard_hit, is_pos))
    P = int(is_pos.sum())
    if P == 0:
        return {"threshold": None, "recall": None, "extra_pages": 0}
    need = int(np.ceil(target_recall * P)) - int((hard_hit & is_pos).sum())
    if need <= 0:
        return {"threshold": None, "recall": float((hard_hit & is_pos).sum() / P), "extra_pages": 0}
    missed = np.sort(joint[is_pos & ~hard_hit])[::-1]
    t = float(missed[min(need, len(missed)) - 1])
    fired = hard_hit | (joint >= t)
    return {"threshold": t, "recall": float((fired & is_pos).sum() / P),
            "extra_pages": int((fired & ~hard_hit & ~is_pos).sum())}


# -------------------------------------------------------------------- pipeline

def build_triage(probs: dict, classes: dict, i: int) -> TriageResult:
    heads = {}
    for h in HEADS:
        p = probs[h][i]
        j = int(p.argmax())
        heads[h] = HeadPrediction(classes[h][j], float(p[j]),
                                  {c: float(v) for c, v in zip(classes[h], p)})
    return TriageResult(**heads)


def run_gate(df, probs, classes, retriever, idx, sims, cfg):
    return [evaluate_gate(df.at[i, "subject"], df.at[i, "body"],
                          build_triage(probs, classes, i),
                          retriever.to_hits(idx[i], sims[i]), cfg)
            for i in range(len(df))]


def tune(kb: pd.DataFrame, args) -> tuple[GateConfig, dict, BaselineTriage, BaselineRetriever]:
    kb = kb.sample(frac=1.0, random_state=7).reset_index(drop=True)
    n = len(kb)
    fit, calib, tun = kb.iloc[: int(.8 * n)], kb.iloc[int(.8 * n): int(.9 * n)], kb.iloc[int(.9 * n):]
    fit, calib, tun = (d.reset_index(drop=True) for d in (fit, calib, tun))
    print(f"KB carve: fit={len(fit)}  calib={len(calib)}  tune={len(tun)}")

    t0 = time.time()
    model = BaselineTriage().fit(fit)
    print(f"Trained baseline classifier in {time.time() - t0:.1f}s")
    calib_report = model.calibrate(calib)
    print("Calibration:", json.dumps(calib_report))

    retriever = BaselineRetriever(model.vec, fit)
    texts = text_of(tun)
    probs = model.predict_heads(texts)
    idx, sims = retriever.retrieve_topk(texts, k=args.k)

    cfg = GateConfig()
    report = {"calibration": calib_report, "heads": {}}

    # 1. Per-head confidence thresholds.
    for h in HEADS:
        pred = np.array(model.classes[h])[probs[h].argmax(1)]
        r = choose_conf_threshold(probs[h].max(1), pred == tun[h].values,
                                  args.target_accuracy[h], args.fallback_coverage)
        cfg.min_confidence[h] = min(floor4(r["threshold"]), 0.99)  # >0.99 never generalizes
        report["heads"][h] = r
        if r["relaxed"]:
            print(f"  ! {h}: accuracy target {args.target_accuracy[h]:.2f} would auto-trust < "
                  f"{args.fallback_coverage:.0%} of tickets; relaxed to {r['coverage']:.0%} "
                  f"(accuracy {r['accuracy']:.2f} on that slice)")

    # 2. Retrieval floors from the in-distribution similarity distribution.
    m = cfg.min_supporting_precedents
    cfg.min_top1_similarity = floor4(np.quantile(sims[:, 0], args.sim_quantile))
    cfg.support_similarity = floor4(np.quantile(sims[:, m - 1], args.sim_quantile))
    report["retrieval"] = {
        "top1_median": float(np.median(sims[:, 0])),
        "min_top1_similarity": cfg.min_top1_similarity,
        "support_similarity": cfg.support_similarity,
    }

    # 3. Soft high-priority-incident cut, aimed at a recall target.
    hi = {h.lower() for h in cfg.high_labels}
    pcls = model.classes["priority"]
    p_high = probs["priority"][:, [i for i, c in enumerate(pcls) if c.lower() in hi]].sum(1)
    p_inc = probs["type"][:, model.classes["type"].index(cfg.incident_label)]
    hard = (np.array(pcls)[probs["priority"].argmax(1)] == "high") & \
           (np.array(model.classes["type"])[probs["type"].argmax(1)] == cfg.incident_label)
    pos = tun["priority"].str.lower().isin(hi).values & (tun["type"] == cfg.incident_label).values
    soft = choose_soft_threshold(p_high * p_inc, hard, pos, args.target_hi_recall)
    cfg.soft_high_incident_prob = None if soft["threshold"] is None else floor4(soft["threshold"])
    report["soft_high_incident"] = soft

    # 4. Run the tuned gate on the tune split and report what each rule costs.
    decisions = run_gate(tun, probs, model.classes, retriever, idx, sims, cfg)
    report["gate_on_tune"] = gate_report(decisions, tun["type"].tolist(), tun["priority"].tolist(),
                                         high=cfg.high_labels)
    rule_counts = {}
    for d in decisions:
        for r in set(d.rules):
            rule_counts[r] = rule_counts.get(r, 0) + 1
    report["rule_fire_rates"] = {r: c / len(decisions) for r, c in sorted(rule_counts.items())}
    kw = [i for i, d in enumerate(decisions) if "critical_impact_keywords" in d.rules]
    report["critical_keyword_precision"] = (
        float(tun.loc[kw, "priority"].isin(["high"]).mean()) if kw else None)
    return cfg, report, model, retriever


def evaluate_on_test(test: pd.DataFrame, model, retriever, cfg, k: int) -> dict:
    from sklearn.metrics import f1_score
    texts = text_of(test)
    t0 = time.time()
    probs = model.predict_heads(texts)
    idx, sims = retriever.retrieve_topk(texts, k=k)
    decisions = run_gate(test, probs, model.classes, retriever, idx, sims, cfg)
    batch_ms = (time.time() - t0) * 1000 / len(test)

    out = {"n": len(test), "macro_f1": {}, "batch_ms_per_ticket": round(batch_ms, 2)}
    for h in HEADS:
        pred = np.array(model.classes[h])[probs[h].argmax(1)]
        out["macro_f1"][h] = round(float(f1_score(test[h].values, pred, average="macro")), 4)

    hi = {h.lower() for h in cfg.high_labels}
    pos = test["priority"].str.lower().isin(hi).values & (test["type"] == cfg.incident_label).values
    pred_hi = (np.array(model.classes["priority"])[probs["priority"].argmax(1)] == "high") & \
              (np.array(model.classes["type"])[probs["type"].argmax(1)] == cfg.incident_label)
    out["classifier_high_incident_recall"] = float(pred_hi[pos].mean()) if pos.any() else None
    out["gate"] = gate_report(decisions, test["type"].tolist(), test["priority"].tolist(),
                              high=cfg.high_labels)

    # Single-ticket latency, the way the live console will see it.
    sample = test.sample(min(200, len(test)), random_state=0).reset_index(drop=True)
    lat = []
    for i in range(len(sample)):
        s = time.perf_counter()
        t = [text_of(sample.iloc[[i]])[0]]
        p = model.predict_heads(t)
        ix, sm = retriever.retrieve_topk(t, k=k)
        evaluate_gate(sample.at[i, "subject"], sample.at[i, "body"],
                      build_triage(p, model.classes, 0), retriever.to_hits(ix[0], sm[0]), cfg)
        lat.append((time.perf_counter() - s) * 1000)
    out["single_ticket_latency_ms"] = {"p50": round(float(np.median(lat)), 2),
                                       "p95": round(float(np.percentile(lat, 95)), 2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="folder you cloned the HF dataset into")
    src.add_argument("--csv", help="path to a specific CSV")
    ap.add_argument("--out", default="gate_config.json")
    ap.add_argument("--report", default="tuning_report.json")
    ap.add_argument("--k", type=int, default=5, help="precedents retrieved per ticket")
    ap.add_argument("--acc-type", type=float, default=0.85)
    ap.add_argument("--acc-queue", type=float, default=0.80)
    ap.add_argument("--acc-priority", type=float, default=0.80)
    ap.add_argument("--min-coverage", dest="fallback_coverage", type=float, default=0.4,
                    help="minimum share of tickets each head auto-trusts; relaxes accuracy targets")
    ap.add_argument("--sim-quantile", type=float, default=0.10)
    ap.add_argument("--target-hi-recall", type=float, default=0.90,
                    help="gate recall target for true high-priority Incidents")
    ap.add_argument("--eval-test", action="store_true", help="score on the 20%% holdout")
    ap.add_argument("--save-models", help="joblib path to save the baseline models for the UI")
    args = ap.parse_args()
    args.target_accuracy = {"type": args.acc_type, "queue": args.acc_queue, "priority": args.acc_priority}

    csv_path = args.csv or find_csv(args.data_dir)
    print(f"Loading {csv_path}")
    kb, test, raw = load_and_split(csv_path)
    print(f"Rows: raw={len(raw)}  kb={len(kb)}  test={len(test)}")
    print("Raw priority values:", raw["priority"].value_counts(dropna=False).to_dict())
    print("Normalized priority (KB):", kb["priority"].value_counts(dropna=False).to_dict())
    print(f"Queues: {kb['queue'].nunique()}   Types: {sorted(kb['type'].unique())}")

    cfg, report, model, retriever = tune(kb, args)
    report["source_csv"] = os.path.abspath(csv_path)

    if args.eval_test:
        print("Scoring holdout (labels used for metrics only)...")
        report["holdout"] = evaluate_on_test(test, model, retriever, cfg, args.k)

    cfg.save(args.out)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    if args.save_models:
        import joblib
        joblib.dump({"triage": model, "retriever": retriever}, args.save_models)

    print(f"\nSaved {args.out} and {args.report}\n")
    print(json.dumps({"config": json.load(open(args.out)),
                      "gate_on_tune": report["gate_on_tune"],
                      "holdout": report.get("holdout")}, indent=2, default=float))


if __name__ == "__main__":
    main()
