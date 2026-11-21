"""Save EmbeddingGemma into artifacts/embeddinggemma so containers run offline (no HF token)."""

import _bootstrap  # noqa: F401
from sentence_transformers import SentenceTransformer

from sentineldesk import config

out = config.ARTIFACTS / "embeddinggemma"
SentenceTransformer(config.EMBED_MODEL, device="cpu").save(str(out))
print(f"saved {config.EMBED_MODEL} -> {out}")
