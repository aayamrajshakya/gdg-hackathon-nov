"""Train SentinelDesk: split the data, fit the triage models, tune thresholds, build the index.

Usage:
    python train.py                        # downloads the official CSV on first run
    python train.py --max-rows 8000        # fast prototype on a subsample
    python train.py --data path/to.csv     # use a local CSV
    python train.py --backend tfidf        # skip the embedding model
"""
from __future__ import annotations

import argparse
import json
import time

from sklearn.metrics import f1_score

from sentinel import config
from sentinel.data import clean, load_raw, save_split, split
from sentinel.retrieval import Retriever
from sentinel.textutils import build_text
from sentinel.triage import TriageModel


def texts_of(df) -> list[str]:
    return [build_text(s, b) for s, b in zip(df["subject"], df["body"])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="local CSV path (default: official file from Hugging Face)")
    ap.add_argument("--backend", default=config.RETRIEVAL_BACKEND, choices=["auto", "embeddings", "tfidf"])
    ap.add_argument("--max-rows", type=int, help="subsample the cleaned data for quick iterations")
    args = ap.parse_args()
    t0 = time.time()

    df = clean(load_raw(args.data))
    if args.max_rows and args.max_rows < len(df):
        df = df.sample(n=args.max_rows, random_state=config.RANDOM_STATE)
    kb, test = split(df)
    save_split(kb, test)
    print(f"[split] knowledge base / training: {len(kb)}   holdout test: {len(test)}")

    # Validation comes from the knowledge base only; the holdout is not touched here.
    val = kb.sample(frac=config.VAL_FRAC, random_state=config.RANDOM_STATE)
    fit = kb.drop(index=val.index)

    model = TriageModel().fit(texts_of(fit), fit)
    critical = model.tune_critical_threshold(texts_of(val), val)
    print(f"[triage] critical detector on validation: {critical}")

    val_preds = model.predict(texts_of(val))
    val_f1 = {
        t: round(f1_score(val[t], [p[t]["label"] for p in val_preds], average="macro", zero_division=0), 4)
        for t in config.TARGETS
    }
    print(f"[triage] validation macro-F1: {val_f1}")

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.ARTIFACT_DIR / "triage.joblib")

    retriever = Retriever(args.backend).build(kb)  # full 80% split, answers included
    retriever.save(config.ARTIFACT_DIR)

    probe = kb.sample(n=min(400, len(kb)), random_state=config.RANDOM_STATE)
    calibrated = retriever.calibrate_threshold(probe, percentile=5.0)
    sim_threshold = config.SIM_THRESHOLD[retriever.backend]
    if calibrated is not None:
        lo, hi = config.SIM_BOUNDS[retriever.backend]
        sim_threshold = round(min(hi, max(lo, calibrated)), 4)
    print(f"[retrieval] precedent similarity threshold: {sim_threshold} (5th percentile of in-domain best matches)")

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_file": args.data or config.DATA_FILE,
        "n_kb": len(kb),
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(test),
        "labels": {t: sorted(kb[t].unique().tolist()) for t in config.TARGETS},
        "language_mix": kb["language"].value_counts().to_dict(),
        "retrieval_backend": retriever.backend,
        "embed_model": retriever.embed_model if retriever.backend == "embeddings" else None,
        "sim_threshold": sim_threshold,
        "critical_detector": critical,
        "val_macro_f1": val_f1,
        "train_seconds": round(time.time() - t0, 1),
    }
    (config.ARTIFACT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[done] artifacts written to {config.ARTIFACT_DIR} in {meta['train_seconds']}s")


if __name__ == "__main__":
    main()
