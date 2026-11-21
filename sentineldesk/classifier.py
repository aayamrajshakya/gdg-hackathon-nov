"""Multi-task triage model: type / queue / priority + a dedicated `critical` head.

Each head is a calibrated blend of three views of the ticket:
  1. TF-IDF (word 1-2grams + char 2-5grams) -> logistic regression  (lexical, language-specific)
  2. EmbeddingGemma vector -> logistic regression                    (semantic, cross-lingual)
  3. kNN vote over the retrieved precedents (similarity-weighted)    (instance-based)

Blend weights are chosen on a validation slice of the KB to maximise macro-F1, then a single
temperature per head is fitted on the same slice (minimum NLL) so the reported confidence is
calibrated. The `critical` head is a binary "priority == high AND type == Incident" detector that
gives the escalation gate a direct risk probability instead of a product of two noisy heads.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

HEADS = ("type", "queue", "priority", "critical")
CRITICAL_POS = "critical"
CRITICAL_NEG = "routine"
KNN_K = 20
KNN_TAU = 0.05


def critical_labels(types, priorities) -> np.ndarray:
    return np.where((np.asarray(priorities) == "high") & (np.asarray(types) == "Incident"),
                    CRITICAL_POS, CRITICAL_NEG)


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=300_000, sublinear_tf=True)),
    ])


def knn_proba(sims: np.ndarray, idx: np.ndarray, neighbor_labels: np.ndarray,
              classes: np.ndarray, tau: float = KNN_TAU) -> np.ndarray:
    """Similarity-weighted label distribution of the k nearest precedents."""
    w = np.exp((sims - sims[:, :1]) / tau)  # stable softmax over neighbours
    lab = neighbor_labels[idx]
    out = np.zeros((len(sims), len(classes)))
    for ci, c in enumerate(classes):
        out[:, ci] = (w * (lab == c)).sum(1)
    out += 1e-3
    return out / out.sum(1, keepdims=True)


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(p, 1e-9, 1)) / t
    logp -= logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def _nll(p: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-9, 1)).mean())


@dataclass
class Head:
    name: str
    classes: np.ndarray
    lr_tfidf: LogisticRegression
    lr_emb: LogisticRegression
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    temperature: float = 1.0

    def components(self, X_tfidf, emb, knn_sims, knn_idx, neighbor_labels):
        return (
            self.lr_tfidf.predict_proba(X_tfidf),
            self.lr_emb.predict_proba(emb),
            knn_proba(knn_sims, knn_idx, neighbor_labels, self.classes),
        )

    def blend(self, comps, weights=None, temperature=None) -> np.ndarray:
        w = weights or self.weights
        p = sum(wi * ci for wi, ci in zip(w, comps))
        return _apply_temperature(p, temperature or self.temperature)

    def tune(self, comps, y: np.ndarray) -> dict:
        """Grid-search blend weights for macro-F1, then fit temperature for NLL."""
        y_idx = np.searchsorted(self.classes, y)
        best = (-1.0, self.weights)
        grid = np.round(np.arange(0, 1.01, 0.1), 2)
        for a, b in itertools.product(grid, grid):
            c = round(1 - a - b, 2)
            if c < 0:
                continue
            pred = self.classes[self.blend(comps, (a, b, c), 1.0).argmax(1)]
            f1 = f1_score(y, pred, average="macro")
            if f1 > best[0] + 1e-9:
                best = (f1, (float(a), float(b), float(c)))
        self.weights = best[1]
        temps = np.exp(np.linspace(np.log(0.05), np.log(5), 120))
        nlls = [_nll(self.blend(comps, temperature=t), y_idx) for t in temps]
        self.temperature = float(temps[int(np.argmin(nlls))])
        return {"val_macro_f1": best[0], "weights": self.weights, "temperature": self.temperature,
                "val_nll": float(min(nlls))}


@dataclass
class TriageModel:
    vectorizer: FeatureUnion
    heads: dict[str, Head]
    neighbor_labels: dict[str, np.ndarray] = field(default_factory=dict)  # aligned with KB index
    gate_thresholds: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def set_neighbor_labels(self, kb) -> None:
        self.neighbor_labels = {h: np.asarray(kb[h]) for h in ("type", "queue", "priority")}
        self.neighbor_labels["critical"] = critical_labels(kb["type"], kb["priority"])

    def predict_proba(self, texts, emb, knn_sims, knn_idx) -> dict[str, np.ndarray]:
        X = self.vectorizer.transform(texts)
        out = {}
        for name, head in self.heads.items():
            comps = head.components(X, emb, knn_sims[:, :KNN_K], knn_idx[:, :KNN_K],
                                    self.neighbor_labels[name])
            out[name] = head.blend(comps)
        return out


def fit_head(name: str, X_tfidf, emb, y, c_tfidf: float = 10.0, c_emb: float = 10.0) -> Head:
    lr_t = LogisticRegression(C=c_tfidf, max_iter=3000, class_weight="balanced")
    lr_t.fit(X_tfidf, y)
    lr_e = LogisticRegression(C=c_emb, max_iter=3000, class_weight="balanced")
    lr_e.fit(emb, y)
    assert list(lr_t.classes_) == list(lr_e.classes_)
    return Head(name=name, classes=lr_t.classes_, lr_tfidf=lr_t, lr_emb=lr_e)
