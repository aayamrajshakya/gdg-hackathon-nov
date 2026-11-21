"""Load the official dataset, clean it, and create the leakage-safe 80/20 split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

TEXT_COLS = ("subject", "body", "answer")
LABEL_COLS = ("type", "queue", "priority", "language")

# Some dataset versions use other priority names; map everything onto high/medium/low.
PRIORITY_MAP = {
    "critical": "high", "urgent": "high", "very_high": "high", "3": "high", "high": "high",
    "medium": "medium", "normal": "medium", "2": "medium",
    "low": "low", "very_low": "low", "1": "low",
}


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read a local CSV, or download the official file once and cache it in data/."""
    if path:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No dataset at {path}")
        print(f"[data] reading {path}")
        return pd.read_csv(path)

    cached = config.DATA_DIR / config.DATA_FILE
    if cached.exists():
        print(f"[data] reading cached {cached}")
        return pd.read_csv(cached)

    print(f"[data] downloading {config.DATA_URL}")
    df = pd.read_csv(config.DATA_URL)
    cached.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cached, index=False)
    return df


def clean(df: pd.DataFrame, langs=config.LANGS) -> pd.DataFrame:
    """Normalise text and labels, drop unusable rows and exact duplicates."""
    missing = [c for c in ("body", "type", "queue", "priority") if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    for col in TEXT_COLS + LABEL_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["priority"] = df["priority"].str.lower().map(lambda p: PRIORITY_MAP.get(p, p))
    df["language"] = df["language"].str.lower()

    usable = df["body"].ne("") & df["type"].ne("") & df["queue"].ne("") & df["priority"].ne("")
    df = df[usable]
    if langs and df["language"].ne("").any():
        df = df[df["language"].isin(langs)]

    # Remove exact duplicates BEFORE splitting so one ticket can't land in both KB and test.
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "body"])
    print(f"[data] {len(df)} usable tickets ({before - len(df)} exact duplicates removed)")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle with a fixed seed, assign ticket IDs, and cut 80% KB / 20% holdout."""
    df = df.sample(frac=1.0, random_state=config.RANDOM_STATE).reset_index(drop=True)
    width = max(5, len(str(len(df))))
    df["ticket_id"] = [f"TKT-{i:0{width}d}" for i in range(len(df))]
    cut = int(len(df) * (1 - config.TEST_FRAC))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def save_split(kb: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write the KB in full, and the holdout as separate inputs and labels.

    The holdout `answer` column is never written to disk, so it cannot leak.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    kb.to_csv(config.DATA_DIR / "kb_train.csv", index=False)
    test[["ticket_id", "subject", "body"]].to_csv(config.DATA_DIR / "test_inputs.csv", index=False)
    test[["ticket_id", "type", "queue", "priority", "language"]].to_csv(
        config.DATA_DIR / "test_labels.csv", index=False
    )


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read one of our CSVs keeping empty subjects as '' rather than NaN."""
    return pd.read_csv(path, keep_default_na=False, dtype=str)