"""Retriever used by the app. Run directly to test it and pick a similarity threshold."""
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "intfloat/multilingual-e5-small"


class Retriever:
    def __init__(self):
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.model.max_seq_length = 256
        self.emb = np.load(os.path.join(HERE, "index", "kb_embeddings.npy"))
        self.meta = pd.read_parquet(os.path.join(HERE, "index", "kb_meta.parquet"))

    def embed_query(self, text: str) -> np.ndarray:
        # e5 models expect the "query: " prefix for searches
        return self.model.encode(["query: " + text], normalize_embeddings=True)[0].astype(np.float32)

    def search(self, text: str, k: int = 5) -> pd.DataFrame:
        """Return the top-k most similar historical tickets with a 'similarity' column."""
        q = self.embed_query(text)
        sims = self.emb @ q  # cosine similarity (vectors are normalized)
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        out = self.meta.iloc[top].copy()
        out["similarity"] = sims[top].round(4)
        return out.reset_index(drop=True)


if __name__ == "__main__":
    import time
    r = Retriever()

    samples = [
        "VPN down for the whole clinic. Telemedicine and EMR integration not reachable since this morning.",
        "VPN funktioniert nicht mehr, seit heute Morgen keine Verbindung zum Firmennetz.",
        "Can I get a refund for my last invoice? I was charged twice.",
        "What is the best recipe for chocolate cake?",  # out-of-scope -> should score low
    ]
    for s in samples:
        t0 = time.perf_counter()
        res = r.search(s, k=3)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\nQUERY: {s}\n  ({ms:.0f} ms)")
        for _, row in res.iterrows():
            print(f"  {row.ticket_id}  sim={row.similarity:.3f}  [{row.language}] {str(row.subject)[:70]}")

    # Similarity distribution on holdout INPUTS only (no labels/answers are read)
    test_inputs = pd.read_parquet(os.path.join(HERE, "data", "test_inputs.parquet")).sample(200, random_state=0)
    top1 = [r.search(t, k=1)["similarity"].iloc[0] for t in test_inputs["text"]]
    print("\nTop-1 similarity on 200 unseen holdout tickets:")
    for p in [5, 10, 25, 50, 75]:
        print(f"  {p:>2}th percentile: {np.percentile(top1, p):.3f}")
