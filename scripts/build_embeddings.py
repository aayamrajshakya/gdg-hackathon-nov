"""Encode KB tickets and holdout *inputs* (subject+body only) with EmbeddingGemma.

Holdout embeddings contain no labels or answers; they are simply what the live system would
compute for an incoming ticket, cached so evaluation is fast.

Work is checkpointed in chunks under artifacts/emb_chunks/, so an interrupted run (e.g. GPU
out-of-memory on a small laptop) resumes where it stopped.
"""

import shutil
import time

import _bootstrap  # noqa: F401
import numpy as np

from sentineldesk import config, data, embeddings

CHUNK = 2048


def encode_chunked(name: str, subjects, bodies) -> np.ndarray:
    chunk_dir = config.ARTIFACTS / "emb_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    n_chunks = (len(subjects) + CHUNK - 1) // CHUNK
    for c in range(n_chunks):
        path = chunk_dir / f"{name}_{c:03d}.npy"
        sl = slice(c * CHUNK, (c + 1) * CHUNK)
        if path.exists():
            parts.append(np.load(path))
            continue
        t0 = time.time()
        e = embeddings.encode(list(subjects[sl]), list(bodies[sl]), batch_size=32)
        np.save(path, e)
        parts.append(e)
        print(f"[{name}] chunk {c + 1}/{n_chunks} ({len(e)} rows, {time.time() - t0:.0f}s)",
              flush=True)
    return np.concatenate(parts)


def main() -> None:
    if not config.KB_PATH.exists():
        data.prepare_splits()
    for name, df, path in (
        ("kb", data.load_kb(), config.KB_EMB_PATH),
        ("test", data.load_test_inputs(), config.TEST_EMB_PATH),
    ):
        if path.exists() and len(np.load(path, mmap_mode="r")) == len(df):
            print(f"[{name}] cached: {path}")
            continue
        emb = encode_chunked(name, df["subject"].tolist(), df["body"].tolist())
        assert len(emb) == len(df)
        np.save(path, emb)
        print(f"[{name}] {emb.shape} -> {path}", flush=True)
    shutil.rmtree(config.ARTIFACTS / "emb_chunks", ignore_errors=True)


if __name__ == "__main__":
    main()
