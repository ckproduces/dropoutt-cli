"""Text normalization and shingling.

Shared by deduplication, contamination scanning and several hygiene checks, so
the definitions live in one place and every check agrees on what "the same text"
means.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

# Characters that indicate a UTF-8 stream was decoded as latin-1 somewhere.
MOJIBAKE_MARKERS = ("Ã", "â€", "Â", "ð\x9f", "Ä±", "Ã¼", "Ã§", "ÅŸ", "ÄŸ")

# Control characters that should never appear in training text. Tab, newline and
# carriage return are excluded because they are legitimate.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Turkish has two distinct letter I pairs. Lowercasing "I" with the default
# locale produces "i", which is wrong in Turkish, and uppercasing "i" produces
# "I" rather than "İ". Corpora that have been through a naive .lower() carry
# this damage, and it is invisible to language identification.
TR_LOWER = str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")

# The dotless and dotted forms, used to spot corpora that were ASCII-folded.
TR_SPECIFIC = "ıİğĞşŞçÇöÖüÜ"
TR_ASCII_FOLD = str.maketrans("ıİğĞşŞçÇöÖüÜ", "iIgGsScCoOuU")


def normalize_ws(text: str) -> str:
    """Collapse whitespace. Used for whitespace-insensitive duplicate detection."""
    return _WS.sub(" ", text).strip()


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for near-duplicate comparison.

    Turkish-aware lowercasing, punctuation stripped, whitespace collapsed. This
    matches what the delta-max platform already does, so the CLI and the
    platform agree on what counts as a duplicate.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(TR_LOWER).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def word_shingles(text: str, n: int = 5) -> list[str]:
    """Word n-grams, the FineWeb MinHash unit."""
    words = normalize_for_dedup(text).split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def hashed_shingles(text: str, n: int = 5) -> list[int]:
    """Word n-grams hashed to 64-bit integers, deduplicated."""
    seen: set[int] = set()
    for sh in word_shingles(text, n):
        seen.add(int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big"))
    return sorted(seen)


def token_ngrams(tokens: list[str], n: int = 8) -> list[str]:
    """Token n-grams, the contamination-detection unit (Tulu 3 uses 8)."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def has_mojibake(text: str) -> bool:
    return any(m in text for m in MOJIBAKE_MARKERS)


def control_chars(text: str) -> list[str]:
    return _CONTROL.findall(text)


def needs_nfc(text: str) -> bool:
    """True when the text is not in NFC form.

    Matters for Turkish and other diacritic-heavy languages: the same visible
    word can tokenize into different ids depending on composition form, so a
    mixed-form corpus fragments its own vocabulary.
    """
    return unicodedata.normalize("NFC", text) != text


def is_ascii_folded_turkish(text: str) -> bool:
    """Detect Turkish text that has lost its diacritics.

    Language identification will confidently call "degil mi" Turkish, and it is,
    but it is damaged Turkish. A model trained on it learns the wrong
    orthography. Heuristic: the text scores as Turkish-shaped by common function
    words, yet contains none of the Turkish-specific characters.
    """
    lowered = text.lower()
    if any(ch in text for ch in TR_SPECIFIC):
        return False
    # Function words that are spelled identically with and without diacritics,
    # so their presence indicates Turkish without presupposing the diacritics.
    markers = (" bir ", " ve ", " bu ", " icin ", " degil ", " ile ", " daha ", " gibi ")
    hits = sum(1 for m in markers if m in f" {lowered} ")
    return hits >= 2


def repetition_ratio(text: str, n: int = 10) -> float:
    """Fraction of n-gram positions that repeat a previously seen n-gram.

    High values indicate a degenerate record: a stuck generation loop, or a page
    of repeated boilerplate.
    """
    words = text.split()
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def excerpt(text: str, limit: int = 220) -> str:
    """A short, single-line preview for evidence display."""
    flat = _WS.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
