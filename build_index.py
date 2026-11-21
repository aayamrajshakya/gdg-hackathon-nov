"""Step 3: embed the historical KB tickets (80% split only) for retrieval.

Outputs (in index/):
  kb_embeddings.npy  -> normalized multilingual embeddings, one row per KB ticket
  kb_meta.parquet    -> ticket_id, subject, body, answer, language, type, queue, priority
Only kb_train.parquet is indexed. The holdout split is never touched here.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("index", exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-small"  # multilingual -> cross-lingual retrieval
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device} ...")
model = SentenceTransformer(MODEL_NAME, device=device)
model.max_seq_length = 256  # tickets are short; this keeps encoding fast

kb = pd.read_parquet("data/kb_train.parquet")
kb = kb.dropna(subset=["answer"]).reset_index(drop=True)  # precedents need a resolution
print(f"Embedding {len(kb)} KB tickets ...")

t0 = time.time()
# e5 models expect the "passage: " prefix for documents
emb = model.encode(["passage: " + t for t in kb["text"].tolist()],
                   batch_size=64, normalize_embeddings=True, show_progress_bar=True)
emb = np.asarray(emb, dtype=np.float32)
print(f"Done in {time.time() - t0:.0f}s, shape {emb.shape}")

np.save("index/kb_embeddings.npy", emb)
meta_cols = ["ticket_id", "subject", "body", "answer", "language", "type", "queue", "priority"]
kb[meta_cols].to_parquet("index/kb_meta.parquet", index=False)
print("Saved index/kb_embeddings.npy and index/kb_meta.parquet")
