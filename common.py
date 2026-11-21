"""Shared helpers used by training, evaluation and the app."""
import pandas as pd

TARGETS = ["type", "queue", "priority"]

# Some dataset versions use extra priority levels; fold them into high/medium/low.
PRIORITY_MAP = {"critical": "high", "very_high": "high", "high": "high",
                "medium": "medium", "low": "low", "very_low": "low"}


def build_text(subject, body) -> str:
    """Combine subject + body. Handles missing/NaN subjects without crashing."""
    subject = "" if subject is None or pd.isna(subject) else str(subject).strip()
    body = "" if body is None or pd.isna(body) else str(body).strip()
    return f"{subject}\n{body}".strip()
