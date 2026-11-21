"""Automatic language detection from raw ticket text (never trusts the dataset's language column)."""

from __future__ import annotations

from functools import lru_cache

SUPPORTED = ("en", "de")


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    # A few extra languages so that e.g. Dutch/French text isn't forced into en/de.
    langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH,
             Language.DUTCH, Language.ITALIAN, Language.PORTUGUESE]
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


def detect(text: str) -> tuple[str, float]:
    """Return (iso_code, confidence). Falls back to English for empty/undetectable text."""
    text = (text or "").strip()
    if not text:
        return "en", 0.0
    values = _detector().compute_language_confidence_values(text)
    if not values:
        return "en", 0.0
    top = values[0]
    return top.language.iso_code_639_1.name.lower(), float(top.value)


def reply_language(text: str) -> tuple[str, str, float]:
    """(detected_code, reply_code, confidence). Replies are produced in en or de."""
    code, conf = detect(text)
    return code, (code if code in SUPPORTED else "en"), conf
