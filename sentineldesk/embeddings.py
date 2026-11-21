"""EmbeddingGemma (google/embeddinggemma-300m) encoder.

EmbeddingGemma is multilingual (100+ languages) and shares one vector space across them, which
gives cross-lingual precedent matching for free: a German ticket can retrieve an English
resolution and vice versa.

Ticket-to-ticket retrieval is symmetric, so both the KB and incoming tickets are encoded with
the same document prompt ("title: <subject> | text: <body>").
"""

from __future__ import annotations

import threading

import numpy as np

from . import config
from .data import clean

_models: dict[str, object] = {}
_lock = threading.Lock()


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(device: str | None = None, dtype: str = "float32"):
    """dtype: "float32" (index building) or "bfloat16" (live queries on MPS/CUDA)."""
    import torch

    device = device or _device()
    if device == "cpu":
        dtype = "float32"  # bf16 matmuls are slow on CPU
    key = f"{device}:{dtype}"
    with _lock:
        if key not in _models:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(config.EMBED_MODEL, device=device,
                                    model_kwargs={"torch_dtype": getattr(torch, dtype)})
            m.max_seq_length = config.EMBED_MAX_TOKENS
            _models[key] = m
    return _models[key]


def format_ticket(subject: object, body: object) -> str:
    """Null-safe: subject may be None/NaN/empty (falls back to EmbeddingGemma's "none" title)."""
    return f"title: {clean(subject) or 'none'} | text: {clean(body)}"


def encode(subjects, bodies, batch_size: int = 32, show_progress: bool = False,
           device: str | None = None, dtype: str = "float32") -> np.ndarray:
    """L2-normalised float32 embeddings (cosine similarity == inner product)."""
    model = get_model(device, dtype)
    texts = [format_ticket(s, b) for s, b in zip(subjects, bodies)]
    # Prompts are passed explicitly (prompt="") so behaviour doesn't depend on the model's
    # registered default prompt.
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        prompt="",
    )
    return emb.astype(np.float32)


def encode_one(subject: str, body: str) -> np.ndarray:
    """Live single-ticket query embedding (bfloat16 on GPU by default, see config)."""
    emb = encode([subject], [body], dtype=config.EMBED_QUERY_DTYPE)[0]
    _release_gpu_cache()
    return emb


def warmup() -> None:
    encode_one("warmup", "warmup")


def _release_gpu_cache() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
