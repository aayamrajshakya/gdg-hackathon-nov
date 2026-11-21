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
from sklearn.metrics import f1_score
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
        self.knn_weight: dict[str, float] = {}   # per target; 0 means classifier only
        self.temperature: dict[str, float] = {}  # per target; 1 means unchanged

    def _final_proba(self, target: str, base: np.ndarray, knn: dict | None,
                     weight: float | None = None, temperature: float | None = None) -> np.ndarray:
        """Blend neighbour votes into classifier probabilities, then apply temperature scaling."""
        w = self.knn_weight.get(target, 0.0) if weight is None else weight
        proba = (1 - w) * base + w * knn[target] if (knn is not None and w > 0) else base
        t = self.temperature.get(target, 1.0) if temperature is None else temperature
        if t != 1.0:
            proba = np.clip(proba, 1e-12, 1.0) ** (1.0 / t)
            proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def classes(self) -> dict[str, list[str]]:
        return {t: [str(c) for c in h.classes_] for t, h in self.heads.items()}

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

    def tune_knn_weights(self, texts: list[str], labels: pd.DataFrame, knn: dict[str, np.ndarray],
                         grid=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> dict:
        """Pick, per target, the neighbour-vote weight with the best validation macro-F1.

        Weight 0 (classifier only) is always a candidate, so the blend is only used
        when it actually helps on held-out data.
        """
        X = self.vectorizer.transform(texts)
        report = {}
        for target, head in self.heads.items():
            base = head.predict_proba(X)
            classes = np.asarray(head.classes_).astype(str)
            y = labels[target].astype(str).to_numpy()
            scores = []
            for w in grid:
                pred = classes[self._final_proba(target, base, knn, weight=w, temperature=1.0).argmax(axis=1)]
                scores.append(f1_score(y, pred, average="macro", zero_division=0))
            best = int(np.argmax(scores))  # first maximum, so the smallest weight wins ties
            self.knn_weight[target] = float(grid[best])
            report[target] = {
                "weight": float(grid[best]),
                "val_macro_f1_classifier_only": round(float(scores[0]), 4),
                "val_macro_f1_with_votes": round(float(scores[best]), 4),
            }
        return report

    def tune_temperature(self, texts: list[str], labels: pd.DataFrame, knn: dict | None = None) -> dict:
        """Fit one temperature per target on validation data so confidence matches accuracy.

        Minimises log-loss. It never changes which label wins, only how confident the
        model says it is (below 1 sharpens under-confident probabilities).
        """
        X = self.vectorizer.transform(texts)
        grid = np.geomspace(0.15, 3.0, 45)
        report = {}
        for target, head in self.heads.items():
            base = head.predict_proba(X)
            classes = [str(c) for c in head.classes_]
            pos = {c: j for j, c in enumerate(classes)}
            y = labels[target].astype(str).to_numpy()
            rows = np.array([i for i, v in enumerate(y) if v in pos])
            cols = np.array([pos[y[i]] for i in rows])
            if len(rows) == 0:
                continue
            losses = []
            for t in grid:
                proba = self._final_proba(target, base, knn, temperature=float(t))
                losses.append(-np.mean(np.log(np.clip(proba[rows, cols], 1e-12, 1.0))))
            best_t = float(grid[int(np.argmin(losses))])
            before = self._final_proba(target, base, knn, temperature=1.0)
            self.temperature[target] = best_t
            after = self._final_proba(target, base, knn)
            report[target] = {
                "temperature": round(best_t, 3),
                "val_accuracy": round(float((before.argmax(1)[rows] == cols).mean()), 4),
                "val_mean_confidence_before": round(float(before.max(1).mean()), 4),
                "val_mean_confidence_after": round(float(after.max(1).mean()), 4),
            }
        return report

    # ------------------------------------------------------------ inference
    def _critical_scores(self, X) -> np.ndarray:
        if self.critical is None:
            return np.zeros(X.shape[0])
        pos = list(self.critical.classes_).index(1)
        return self.critical.predict_proba(X)[:, pos]

    def predict(self, texts: list[str], knn: dict[str, np.ndarray] | None = None) -> list[dict]:
        """Return one dict per text with label, confidence and full distribution per head.

        `knn` holds neighbour label votes (from Retriever.label_votes); they are blended in
        with the tuned weights. Without them, the classifier alone is used.
        """
        X = self.vectorizer.transform(list(texts))
        n = X.shape[0]
        results: list[dict] = [{} for _ in range(n)]
        for target, head in self.heads.items():
            proba = self._final_proba(target, head.predict_proba(X), knn)
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
            "knn_weight": self.knn_weight,
            "temperature": self.temperature,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "TriageModel":
        state = joblib.load(path)
        model = cls()
        model.vectorizer = state["vectorizer"]
        model.heads = state["heads"]
        model.critical = state["critical"]
        model.critical_threshold = state["critical_threshold"]
        model.knn_weight = state.get("knn_weight", {})    # older models: classifier only
        model.temperature = state.get("temperature", {})  # older models: no rescaling
        return model