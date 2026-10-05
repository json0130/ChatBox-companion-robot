"""Pronunciation respellings for the TTS, te reo word -> NZ English respelling.

PLACEHOLDERS: validate every entry with Māori language advisers or speakers before any real use.
"""
from typing import Optional

LEXICON = {
    "whānau": "FAH-no",
    "hui": "HOO-ee",
    "kia ora": "kee-ah OR-ah",
}


def lookup(word: str) -> Optional[str]:
    """Respelling for `word`, case-insensitive, or None if unknown."""
    return LEXICON.get(word.strip().casefold())
