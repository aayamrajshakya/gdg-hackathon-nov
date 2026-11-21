"""Train the triage model and tune the escalation gate.

Uses ONLY the 80% KB split:
  * KB[10%:]  -> fit TF-IDF / embedding logistic regressions, kNN neighbour pool
  * KB[:10%]  -> tune blend weights, temperatures and gate thresholds (validation)
The holdout split is never read here.
"""

import json
import time

import _bootstrap  # noqa: F401
import joblib
import numpy as np

from sentineldesk import config, data, embeddings
from sentineldesk.classifier import (HEADS, KNN_K, TriageModel, critical_labels, fit_head,
                                     make_vectorizer)
from sentineldesk.retriever import build_index

TARGET_CRITICAL_RECALL = 0.90
OOD_PROBES = [
    ("", "Write me a short poem about the ocean and the moon."),
    ("Recipe", "What is a good recipe for vegetarian lasagna for six people?"),
    ("", "Wie wird das Wetter morgen in Berlin?"),
    ("Football", "Who won the Champions League final last year?"),
    ("", "Kannst du mir bei meinen Mathe-Hausaufgaben helfen? 3x + 5 = 20"),
]


def main() -> None:
    t_start = time.time()
    kb = data.load_kb()
    emb = np.load(config.KB_EMB_PATH)
    assert len(emb) == len(kb), "run scripts/build_embeddings.py first"

    n_val = int(len(kb) * config.VAL_FRAC_OF_KB)
    va, tr = kb.iloc[:n_val], kb.iloc[n_val:]
    e_va, e_tr = emb[:n_val], emb[n_val:]
    labels_tr = {h: tr[h].to_numpy() for h in ("type", "queue", "priority")}
    labels_va = {h: va[h].to_numpy() for h in ("type", "queue", "priority")}
    labels_tr["critical"] = critical_labels(tr["type"], tr["priority"])
    labels_va["critical"] = critical_labels(va["type"], va["priority"])
    print(f"train={len(tr)} val={len(va)}")

    vec = make_vectorizer()
    X_tr = vec.fit_transform(tr["text"])
    X_va = vec.transform(va["text"])
    print(f"tfidf {X_tr.shape} ({time.time() - t_start:.0f}s)")

    # kNN pool = training rows only, so validation neighbours never include themselves.
    pool = build_index(e_tr)
    knn_sims, knn_idx = pool.search(e_va, KNN_K)

    heads, report = {}, {}
    for h in HEADS:
        t0 = time.time()
        head = fit_head(h, X_tr, e_tr, labels_tr[h])
        comps = head.components(X_va, e_va, knn_sims, knn_idx, labels_tr[h])
        info = head.tune(comps, labels_va[h])
        heads[h] = head
        report[h] = info
        print(f"[{h}] {info} ({time.time() - t0:.0f}s)", flush=True)

    model = TriageModel(vectorizer=vec, heads=heads)
    model.neighbor_labels = labels_tr
    probs = model.predict_proba(va["text"], e_va, knn_sims, knn_idx)

    # ---- gate thresholds ---------------------------------------------------------------
    pred = {h: heads[h].classes[probs[h].argmax(1)] for h in HEADS}
    conf = {h: probs[h].max(1) for h in HEADS}
    risk = probs["critical"][:, list(heads["critical"].classes).index("critical")]
    is_crit = labels_va["critical"] == "critical"
    argmax_rule = (pred["priority"] == "high") & (pred["type"] == "Incident")

    crit_t = 0.5
    for t in np.round(np.arange(0.95, 0.0, -0.01), 2):
        if (argmax_rule | (risk >= t))[is_crit].mean() >= TARGET_CRITICAL_RECALL:
            crit_t = float(t)
            break
    crit_flag = argmax_rule | (risk >= crit_t)

    routing = np.sqrt(conf["type"] * conf["queue"])
    conf_t = float(np.quantile(routing, 0.15))
    route_ok = (pred["type"] == labels_va["type"]) & (pred["queue"] == labels_va["queue"])

    top1 = knn_sims[:, 0]
    ood_emb = embeddings.encode([s for s, _ in OOD_PROBES], [b for _, b in OOD_PROBES])
    ood_top = pool.search(ood_emb, 1)[0][:, 0]
    in_p01 = float(np.quantile(top1, 0.01))
    sim_t = float(np.round((in_p01 + ood_top.max()) / 2, 3)) if ood_top.max() < in_p01 else in_p01

    model.gate_thresholds = {"critical_risk": crit_t, "min_confidence": round(conf_t, 3),
                             "min_similarity": sim_t}
    gate_report = {
        "critical_risk_threshold": crit_t,
        "val_critical_recall": float(crit_flag[is_crit].mean()),
        "val_critical_escalation_rate": float(crit_flag.mean()),
        "val_argmax_rule_recall": float(argmax_rule[is_crit].mean()),
        "min_confidence_threshold": round(conf_t, 3),
        "val_routing_acc_above": float(route_ok[routing >= conf_t].mean()),
        "val_routing_acc_below": float(route_ok[routing < conf_t].mean()),
        "min_similarity_threshold": sim_t,
        "val_top1_sim_p01": in_p01, "val_top1_sim_median": float(np.median(top1)),
        "ood_probe_top1_sims": [round(float(s), 3) for s in ood_top],
    }
    print(json.dumps(gate_report, indent=2))

    model.meta = {"val": report, "gate": gate_report, "n_train": len(tr), "n_val": len(va),
                  "embed_model": config.EMBED_MODEL}
    model.neighbor_labels = {}  # rebuilt from the full KB at load time
    joblib.dump(model, config.CLASSIFIER_PATH, compress=3)

    print(f"saved {config.CLASSIFIER_PATH.name} in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
