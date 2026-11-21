"""Central configuration. Every tunable is overridable via environment variables (or a .env file)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _envf(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ---------------------------------------------------------------- data / artifacts
DATA_DIR = ROOT / "data"
DATASET_REPO = "Tobi-Bueck/customer-support-tickets"
DATASET_FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
RAW_CSV = DATA_DIR / "raw" / DATASET_FILE

ARTIFACTS = Path(_env("SD_ARTIFACTS", str(ROOT / "artifacts")))
KB_PATH = ARTIFACTS / "kb.parquet"                    # 80% split, answers allowed
TEST_INPUTS_PATH = ARTIFACTS / "test_inputs.parquet"  # 20% split: ticket_id/subject/body ONLY
TEST_LABELS_PATH = ARTIFACTS / "test_labels.parquet"  # 20% split labels: read only by evaluate.py
KB_EMB_PATH = ARTIFACTS / "kb_emb.npy"
TEST_EMB_PATH = ARTIFACTS / "test_emb.npy"
CLASSIFIER_PATH = ARTIFACTS / "triage_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"

SEED = 42
TEST_FRAC = 0.2
VAL_FRAC_OF_KB = 0.1  # carved from the KB split for calibration / threshold tuning

# ---------------------------------------------------------------- models
EMBED_MODEL = _env("SD_EMBED_MODEL", "google/embeddinggemma-300m")
EMBED_MAX_TOKENS = int(_env("SD_EMBED_MAX_TOKENS", "512"))
# The KB index is always built in float32. Live queries on Apple GPUs use bfloat16 (half the
# memory, leaves room for local Gemma on 8 GB Macs). Measured vs float32: cosine >= 0.998,
# 100% top-1 / 99.7% top-5 retrieval agreement. float16 is NOT supported by EmbeddingGemma.
EMBED_QUERY_DTYPE = _env("SD_EMBED_QUERY_DTYPE", "bfloat16")
MLX_CACHE_LIMIT_MB = int(_env("SD_MLX_CACHE_LIMIT_MB", "512"))

# LLM backend: "auto" tries gemini (Gemma hosted on Google AI, needs GEMINI_API_KEY), then mlx
# (local Gemma 4 on Apple Silicon), then ollama. Or force one: gemini | mlx | ollama.
LLM_BACKEND = _env("SD_LLM_BACKEND", "auto")
MLX_MODEL = _env("SD_MLX_MODEL", "mlx-community/gemma-4-e4b-it-4bit")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_GEMMA_MODEL = _env("SD_GEMINI_MODEL", "gemma-3-27b-it")
# Set GOOGLE_GENAI_USE_VERTEXAI=true for Vertex AI (express-mode) keys instead of AI Studio keys.
GEMINI_USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _env("SD_OLLAMA_MODEL", "gemma3:4b")
LLM_TIMEOUT_S = _envf("SD_LLM_TIMEOUT_S", 90)

# ---------------------------------------------------------------- retrieval
TOP_K = int(_env("SD_TOP_K", "5"))

# ---------------------------------------------------------------- escalation gate
# Defaults below are overwritten by values tuned on the KB validation split (stored in the
# classifier artifact); env vars still win if set explicitly.
GATE_MIN_CONFIDENCE = os.getenv("SD_GATE_MIN_CONFIDENCE")      # min joint triage confidence
GATE_MIN_SIMILARITY = os.getenv("SD_GATE_MIN_SIMILARITY")      # min top-1 retrieval cosine
GATE_CRITICAL_RISK = os.getenv("SD_GATE_CRITICAL_RISK")        # P(high & Incident) threshold

# ---------------------------------------------------------------- burst detection
BURST_WINDOW_S = _envf("SD_BURST_WINDOW_S", 15 * 60)
BURST_MIN_TICKETS = int(_env("SD_BURST_MIN_TICKETS", "3"))
BURST_SIMILARITY = _envf("SD_BURST_SIMILARITY", 0.80)
