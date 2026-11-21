"""Precedent retrieval over the historical KB (80% split only).

Exact inner-product search on L2-normalised EmbeddingGemma vectors (= cosine similarity) using a
single BLAS matmul (Apple Accelerate / OpenBLAS). At ~23k x 768 this takes a few ms per query
and is exact. FAISS was dropped deliberately: on macOS its bundled OpenMP runtime deadlocks
against PyTorch's in the same process. The `ntotal` / `search(q, k)` interface mirrors FAISS so
an ANN index can be swapped in if the KB grows to millions of tickets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Precedent:
    ticket_id: str
    similarity: float
    subject: str
    body: str
    answer: str
    type: str
    queue: str
    priority: str
    language: str
    tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class ExactIndex:
    """Exact top-k inner-product search; FAISS-compatible `ntotal` / `search` API."""

    def __init__(self, emb: np.ndarray):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.emb)

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(q).astype(np.float32)
        k = min(k, self.ntotal)
        sims = q @ self.emb.T
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part = np.take_along_axis(sims, idx, 1)
        order = np.argsort(-part, axis=1)
        return np.take_along_axis(part, order, 1), np.take_along_axis(idx, order, 1)


def build_index(emb: np.ndarray) -> ExactIndex:
    return ExactIndex(emb)


class Retriever:
    def __init__(self, kb: pd.DataFrame, index: ExactIndex):
        if index.ntotal != len(kb):
            raise ValueError(f"index has {index.ntotal} vectors but KB has {len(kb)} rows")
        self.kb = kb.reset_index(drop=True)
        self.index = index

    @classmethod
    def load(cls) -> "Retriever":
        kb = pd.read_parquet(config.KB_PATH)
        return cls(kb, build_index(np.load(config.KB_EMB_PATH)))

    def search_raw(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.index.search(q, k)

    def search(self, q: np.ndarray, k: int = config.TOP_K) -> list[Precedent]:
        sims, idx = self.search_raw(q, k)
        return self.precedents(sims[0], idx[0])

    def precedents(self, sims: np.ndarray, idx: np.ndarray) -> list[Precedent]:
        out = []
        for s, i in zip(sims, idx):
            if i < 0:
                continue
            r = self.kb.iloc[int(i)]
            out.append(Precedent(
                ticket_id=r.ticket_id, similarity=float(s), subject=r.subject or "",
                body=r.body, answer=r.answer or "", type=r.type, queue=r.queue,
                priority=r.priority, language=r.language, tags=list(r.tags),
            ))
        return out
