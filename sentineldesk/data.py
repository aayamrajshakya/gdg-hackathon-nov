"""Dataset loading, the leakage-safe 80/20 split, and ticket text normalisation."""

from __future__ import annotations

import re

import pandas as pd

from . import config

LABELS = ("type", "queue", "priority")
TAG_COLS = [f"tag_{i}" for i in range(1, 9)]


def clean(text: object) -> str:
    """Null-safe whitespace normalisation (subjects may be NaN/None/empty)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).replace("\\n", "\n").replace("\r", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def ticket_text(subject: object, body: object) -> str:
    """Single string used by the classifier and the retriever. Missing subjects are fine."""
    subj, bod = clean(subject), clean(body)
    return f"{subj}\n{bod}" if subj else bod


def download_raw() -> None:
    if config.RAW_CSV.exists():
        return
    from huggingface_hub import hf_hub_download

    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        config.DATASET_REPO,
        config.DATASET_FILE,
        repo_type="dataset",
        local_dir=str(config.RAW_CSV.parent),
    )


def load_raw() -> pd.DataFrame:
    download_raw()
    return pd.read_csv(config.RAW_CSV)


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly the split from the challenge brief: shuffle(seed=42) -> TKT ids -> first 80% = KB."""
    df = df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
    df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    split_idx = int(len(df) * (1 - config.TEST_FRAC))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def prepare_splits() -> None:
    """Materialise the split to disk.

    The holdout is written as two files so inference code can never touch labels:
      * test_inputs.parquet  -> ticket_id, subject, body  (what a live ticket looks like)
      * test_labels.parquet  -> ticket_id, type, queue, priority, language (eval only)
    The holdout `answer` column is dropped and never persisted.
    """
    kb, test = make_split(load_raw())
    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    kb["subject"] = kb["subject"].map(clean)
    kb["body"] = kb["body"].map(clean)
    kb["answer"] = kb["answer"].map(clean)
    kb["text"] = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    kb["tags"] = kb[TAG_COLS].apply(lambda r: [t for t in r if isinstance(t, str) and t], axis=1)
    kb = kb[["ticket_id", "subject", "body", "text", "answer", *LABELS, "language", "tags"]]
    kb.to_parquet(config.KB_PATH, index=False)

    test[["ticket_id", "subject", "body"]].to_parquet(config.TEST_INPUTS_PATH, index=False)
    test[["ticket_id", *LABELS, "language"]].to_parquet(config.TEST_LABELS_PATH, index=False)


def load_kb() -> pd.DataFrame:
    return pd.read_parquet(config.KB_PATH)


def load_test_inputs() -> pd.DataFrame:
    return pd.read_parquet(config.TEST_INPUTS_PATH)


def load_test_labels() -> pd.DataFrame:
    """Only scripts/evaluate.py should call this."""
    return pd.read_parquet(config.TEST_LABELS_PATH)
