"""Multi-task triage engine.

Predicts type, queue and priority with calibrated probabilities, plus a binary
critical-incident detector (type == Incident and priority == high) whose
threshold is tuned on validation data for high recall.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from . import config

MIN_CLASS_COUNT = 6  # 3-fold calibration needs a few examples of every class


def critical_labels(labels: pd.DataFrame) -> np.ndarray:
    is_incident = labels["type"].str.lower().eq("incident")
    is_high = labels["priority"].str.lower().eq("high")
    return (is_incident & is_high).astype(int).to_numpy()


def _make_vectorizer() -> FeatureUnion:
    # Word n-grams capture vocabulary; character n-grams handle German compounds,
    # typos and mixed-language tickets without a per-language tokenizer.
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                 sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)),
    ])


def _make_classifier() -> CalibratedClassifierCV:
    # LinearSVC is fast and strong on sparse text. Sigmoid calibration on held-out
    # folds turns its margins into probabilities the escalation gate can threshold.
    return CalibratedClassifierCV(
        LinearSVC(C=0.5, class_weight="balanced"), method="sigmoid", cv=3, n_jobs=-1
    )


class TriageModel:
    def __init__(self) -> None:
        self.vectorizer = _make_vectorizer()
        self.heads: dict[str, CalibratedClassifierCV] = {}
        self.critical: CalibratedClassifierCV | None = None
        self.critical_threshold = 0.5

    # ------------------------------------------------------------ training
    def fit(self, texts: list[str], labels: pd.DataFrame) -> "TriageModel":
        X = self.vectorizer.fit_transform(texts)
        print(f"[triage] feature matrix {X.shape[0]} x {X.shape[1]}")
        for target in config.TARGETS:
            y = labels[target].astype(str).to_numpy()
            counts = pd.Series(y).value_counts()
            keep = np.isin(y, counts[counts >= MIN_CLASS_COUNT].index.to_numpy())
            print(f"[triage] '{target}' head: {keep.sum()} rows, {len(np.unique(y[keep]))} classes")
            self.heads[target] = _make_classifier().fit(X[keep], y[keep])

        y_crit = critical_labels(labels)
        if y_crit.sum() >= MIN_CLASS_COUNT and (len(y_crit) - y_crit.sum()) >= MIN_CLASS_COUNT:
            print(f"[triage] critical-incident detector: {y_crit.mean():.1%} positives")
            self.critical = _make_classifier().fit(X, y_crit)
        else:
            print("[triage] too few high-priority incidents to train the critical detector")
        return self

    def tune_critical_threshold(self, texts: list[str], labels: pd.DataFrame,
                                target_recall: float = config.CRITICAL_TARGET_RECALL) -> dict:
        """Pick the highest threshold that still reaches `target_recall` on validation."""
        scores = self._critical_scores(self.vectorizer.transform(texts))
        y = critical_labels(labels)
        positives = np.sort(scores[y == 1])
        if self.critical is None or len(positives) == 0:
            return {"threshold": self.critical_threshold, "note": "detector not trained"}
        idx = int(np.floor((1 - target_recall) * len(positives)))
        self.critical_threshold = float(positives[idx])
        flagged = scores >= self.critical_threshold
        return {
            "threshold": round(self.critical_threshold, 4),
            "val_recall": round(float(flagged[y == 1].mean()), 4),
            "val_precision": round(float(y[flagged].mean()) if flagged.any() else 0.0, 4),
            "val_flag_rate": round(float(flagged.mean()), 4),
        }

    # ------------------------------------------------------------ inference
    def _critical_scores(self, X) -> np.ndarray:
        if self.critical is None:
            return np.zeros(X.shape[0])
        pos = list(self.critical.classes_).index(1)
        return self.critical.predict_proba(X)[:, pos]

    def predict(self, texts: list[str]) -> list[dict]:
        """Return one dict per text with label, confidence and full distribution per head."""
        X = self.vectorizer.transform(list(texts))
        n = X.shape[0]
        results: list[dict] = [{} for _ in range(n)]
        for target, head in self.heads.items():
            proba = head.predict_proba(X)
            classes = [str(c) for c in head.classes_]
            best = proba.argmax(axis=1)
            for i in range(n):
                results[i][target] = {
                    "label": classes[best[i]],
                    "confidence": float(proba[i, best[i]]),
                    "probs": {c: round(float(p), 4) for c, p in zip(classes, proba[i])},
                }
        crit = self._critical_scores(X)
        for i, r in enumerate(results):
            confs = np.clip([r[t]["confidence"] for t in self.heads], 1e-9, 1.0)
            r["confidence"] = float(np.exp(np.mean(np.log(confs))))  # geometric mean of heads
            r["critical_score"] = float(crit[i])
        return results

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        joblib.dump({
            "vectorizer": self.vectorizer,
            "heads": self.heads,
            "critical": self.critical,
            "critical_threshold": self.critical_threshold,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "TriageModel":
        state = joblib.load(path)
        model = cls()
        model.vectorizer = state["vectorizer"]
        model.heads = state["heads"]
        model.critical = state["critical"]
        model.critical_threshold = state["critical_threshold"]
        return model
