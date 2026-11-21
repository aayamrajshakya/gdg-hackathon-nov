"""Central configuration. Override any value with an environment variable or a .env file."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: read a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("SENTINEL_ARTIFACT_DIR", ROOT / "artifacts"))

# ---------------------------------------------------------------- dataset
# We load ONE specific CSV from the official repo (English + German).
# Loading the whole repo mixes three files with different versions of the tickets.
HF_REPO = "Tobi-Bueck/customer-support-tickets"
DATA_FILE = os.getenv("SENTINEL_DATA_FILE", "aa_dataset-tickets-multi-lang-5-2-50-version.csv")
DATA_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{DATA_FILE}"
LANGS = ("en", "de")

RANDOM_STATE = 42
TEST_FRAC = 0.20  # holdout, never indexed
VAL_FRAC = 0.10   # carved out of the 80% knowledge base only, for threshold tuning

TARGETS = ("type", "queue", "priority")

# ---------------------------------------------------------------- retrieval
RETRIEVAL_BACKEND = os.getenv("SENTINEL_RETRIEVAL", "auto")  # auto | embeddings | tfidf
EMBED_MODEL = os.getenv(
    "SENTINEL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
USE_FAISS = os.getenv("SENTINEL_USE_FAISS", "0") == "1"  # opt-in; NumPy exact search by default
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # avoid tokenizer/fork conflicts
TOP_K = 5
CANDIDATES = 30      # retrieve this many, then re-rank and keep TOP_K
QUEUE_BONUS = 0.03   # small re-ranking boost when a precedent shares the predicted queue
TYPE_BONUS = 0.01

# ---------------------------------------------------------------- escalation gate
MIN_TRIAGE_CONFIDENCE = float(os.getenv("SENTINEL_MIN_CONF", "0.40"))
# Similarity scales differ by backend, so each gets its own threshold.
SIM_THRESHOLD = {"embeddings": 0.55, "tfidf": 0.25}  # fallbacks; train.py calibrates from data
SIM_BOUNDS = {"embeddings": (0.35, 0.70), "tfidf": (0.15, 0.45)}  # limits for the calibrated value
CRITICAL_TARGET_RECALL = 0.90  # tune the critical-incident detector for this recall

# ---------------------------------------------------------------- outage burst detection
BURST_SIM = {"embeddings": 0.62, "tfidf": 0.30}
BURST_WINDOW_MIN = 30
BURST_MIN_TICKETS = 3

# ---------------------------------------------------------------- LLM drafting (local only)
# Tickets contain PII, so drafting runs on a local open-source model via Ollama.
# No API keys are used, and the generator refuses any non-local endpoint.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | template
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # Qwen 2.5 7B Instruct, Apache 2.0
OLLAMA_NUM_CTX = 8192
LLM_MAX_TOKENS = 600
LLM_TIMEOUT_S = 180  # the first call loads the model into memory