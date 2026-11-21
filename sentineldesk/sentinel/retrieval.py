"""Precedent retrieval over the historical knowledge base (the 80% split only).

Default backend: multilingual sentence embeddings + FAISS, which also gives
cross-lingual matching (a German ticket can find English precedents).
Fallback backend: TF-IDF cosine similarity, used when sentence-transformers
is not installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .textutils import build_text

META_COLS = ["ticket_id", "subject", "body", "answer", "type", "queue", "priority", "language"]


def resolve_backend(requested: str) -> str:
    if requested in ("embeddings", "tfidf"):
        return requested
    try:
        import sentence_transformers  # noqa: F401
        return "embeddings"
    except ImportError:
        return "tfidf"


def similarity(a, b) -> float:
    """Cosine similarity of two encoded rows (both L2-normalised)."""
    if sparse.issparse(a):
        return float((a @ b.T).toarray()[0, 0])
    return float(np.dot(np.ravel(a), np.ravel(b)))


class Retriever:
    def __init__(self, backend: str = "auto") -> None:
        self.backend = resolve_backend(backend)
        self.embed_model = config.EMBED_MODEL
        self.meta: pd.DataFrame | None = None
        self.matrix = None
        self.vectorizer: TfidfVectorizer | None = None
        self._encoder = None
        self._index = None

    # ------------------------------------------------------------ encoding
    def _encoder_model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model)
        return self._encoder

    def encode(self, texts: list[str], progress: bool = False):
        if self.backend == "embeddings":
            vecs = self._encoder_model().encode(
                list(texts), batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=progress,
            )
            return vecs.astype("float32")
        return self.vectorizer.transform(list(texts))  # rows are L2-normalised

    # ------------------------------------------------------------ build
    def build(self, kb: pd.DataFrame) -> "Retriever":
        self.meta = kb[META_COLS].reset_index(drop=True).astype(str)
        texts = [build_text(s, b) for s, b in zip(self.meta["subject"], self.meta["body"])]
        print(f"[retrieval] indexing {len(texts)} knowledge-base tickets with '{self.backend}'")
        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8,
                                              sublinear_tf=True, max_features=150_000)
            self.matrix = self.vectorizer.fit_transform(texts).astype(np.float32).tocsr()
        else:
            self.matrix = self.encode(texts, progress=True)
        self._build_index()
        return self

    def _build_index(self) -> None:
        self._index = None
        if self.backend != "embeddings":
            return
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.matrix.shape[1])
            self._index.add(self.matrix)
        except ImportError:
            pass  # numpy dot product is fast enough for tens of thousands of rows

    # ------------------------------------------------------------ search
    def _nearest(self, qvec, n: int) -> tuple[np.ndarray, np.ndarray]:
        n = min(n, len(self.meta))
        if self._index is not None:
            sims, idx = self._index.search(np.asarray(qvec, dtype="float32").reshape(1, -1), n)
            return sims[0], idx[0]
        if sparse.issparse(qvec):
            sims = (self.matrix @ qvec.T).toarray().ravel()
        else:
            sims = self.matrix @ np.ravel(qvec)
        idx = np.argpartition(-sims, n - 1)[:n]
        idx = idx[np.argsort(-sims[idx])]
        return sims[idx], idx

    def search(self, text: str, k: int = config.TOP_K, queue: str | None = None,
               ticket_type: str | None = None):
        """Return (top-k precedent dicts, query vector)."""
        qvec = self.encode([text])
        sims, idx = self._nearest(qvec, config.CANDIDATES)
        hits = []
        for sim, i in zip(sims, idx):
            if i < 0:
                continue
            row = self.meta.iloc[int(i)]
            score = float(sim) + config.QUEUE_BONUS * (row["queue"] == queue) \
                + config.TYPE_BONUS * (row["type"] == ticket_type)
            hits.append({**row.to_dict(), "similarity": round(float(sim), 4), "score": round(score, 4)})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k], qvec

    def calibrate_threshold(self, sample: pd.DataFrame, percentile: float = 5.0) -> float | None:
        """Best-match similarity of known in-domain tickets, excluding the ticket itself.

        Returns a low percentile of that distribution: a new ticket whose best precedent
        is worse than ~95% of real tickets' best precedents is treated as out of scope.
        """
        texts = [build_text(s, b) for s, b in zip(sample["subject"], sample["body"])]
        vecs = self.encode(texts)
        best: list[float] = []
        for i, tid in enumerate(sample["ticket_id"]):
            sims, idx = self._nearest(vecs[i:i + 1], 3)
            for sim, j in zip(sims, idx):
                if j >= 0 and self.meta.iloc[int(j)]["ticket_id"] != tid:
                    best.append(float(sim))
                    break
        return float(np.percentile(best, percentile)) if best else None

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.meta.to_csv(d / "kb_meta.csv", index=False)
        if self.backend == "tfidf":
            joblib.dump(self.vectorizer, d / "retrieval_tfidf.joblib")
            sparse.save_npz(d / "retrieval_matrix.npz", self.matrix)
        else:
            np.save(d / "retrieval_embeddings.npy", self.matrix)
        (d / "retrieval.json").write_text(json.dumps(
            {"backend": self.backend, "embed_model": self.embed_model, "size": len(self.meta)}, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "Retriever":
        d = Path(directory)
        info = json.loads((d / "retrieval.json").read_text())
        r = cls(info["backend"])
        r.embed_model = info.get("embed_model", config.EMBED_MODEL)
        r.meta = pd.read_csv(d / "kb_meta.csv", keep_default_na=False, dtype=str)
        if r.backend == "tfidf":
            r.vectorizer = joblib.load(d / "retrieval_tfidf.joblib")
            r.matrix = sparse.load_npz(d / "retrieval_matrix.npz").tocsr()
        else:
            r.matrix = np.load(d / "retrieval_embeddings.npy")
        r._build_index()
        return r
