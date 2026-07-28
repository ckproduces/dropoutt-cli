"""Language identification with an honest confidence signal.

Two things this module refuses to do.

It does not report a language without a confidence. Identification is least
reliable exactly where it matters most for this product: short text, and closely
related languages. Turkish against Azerbaijani against Turkmen is the canonical
hard case, and a bag-of-n-grams model will confuse them on a five-word snippet
no matter which implementation is used.

It does not call the fallback backend equivalent to the real one. When
``fasttext-langdetect`` is missing, a small character-profile detector is used
instead; it covers a handful of languages, is materially less accurate, and
every result it produces is marked low-trust so the report can say so.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .compat import HAVE_FASTTEXT_LID

#: Below this, we emit "unknown" rather than a language.
DEFAULT_FLOOR = 0.55

#: Languages that are routinely confused with one another. When the detector
#: picks one of these with middling confidence, the finding says so explicitly
#: instead of pretending the answer is settled.
CONFUSABLE_GROUPS = [
    {"tr", "az", "tk", "ug", "uz", "kk", "ky"},
    {"es", "gl", "pt", "ca"},
    {"hr", "sr", "bs", "sl"},
    {"da", "no", "nb", "nn", "sv"},
    {"hi", "mr", "ne", "bh"},
    {"id", "ms"},
    {"cs", "sk"},
]

SCRIPT_RANGES = [
    ("Latin", 0x0041, 0x024F),
    ("Greek", 0x0370, 0x03FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Hebrew", 0x0590, 0x05FF),
    ("Arabic", 0x0600, 0x06FF),
    ("Devanagari", 0x0900, 0x097F),
    ("Hangul", 0xAC00, 0xD7AF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Han", 0x4E00, 0x9FFF),
]

#: Which scripts each language is normally written in. A mismatch is a finding:
#: Turkish in Arabic script is Ottoman, not a typo, and the user should know.
EXPECTED_SCRIPT = {
    "tr": {"Latin"}, "az": {"Latin", "Cyrillic", "Arabic"}, "tk": {"Latin", "Cyrillic"},
    "en": {"Latin"}, "de": {"Latin"}, "fr": {"Latin"}, "es": {"Latin"}, "it": {"Latin"},
    "ru": {"Cyrillic"}, "uk": {"Cyrillic"}, "bg": {"Cyrillic"},
    "ar": {"Arabic"}, "fa": {"Arabic"}, "ur": {"Arabic"},
    "he": {"Hebrew"}, "el": {"Greek"}, "hi": {"Devanagari"},
    "ja": {"Hiragana", "Katakana", "Han"}, "ko": {"Hangul"}, "zh": {"Han"},
}


@dataclass(slots=True)
class LangResult:
    lang: str
    confidence: float
    script: str
    #: True when produced by the reduced-accuracy fallback backend.
    low_trust: bool = False

    @property
    def is_unknown(self) -> bool:
        return self.lang == "unknown"

    @property
    def confusable_with(self) -> set[str]:
        for group in CONFUSABLE_GROUPS:
            if self.lang in group:
                return group - {self.lang}
        return set()


def dominant_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text[:2000]:
        cp = ord(ch)
        if not ch.isalpha():
            continue
        for name, lo, hi in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["Other"] = counts.get("Other", 0) + 1
    if not counts:
        return "None"
    return max(counts, key=lambda k: counts[k])


class LanguageDetector:
    """Wraps whichever backend is available."""

    def __init__(self, floor: float | None = None) -> None:
        if HAVE_FASTTEXT_LID:
            from ftlangdetect import detect  # noqa: PLC0415

            self._detect = detect
            self.backend = "fasttext-lid.176"
            self.low_trust = False
            self.floor = DEFAULT_FLOOR if floor is None else floor
        else:
            self._detect = None
            self.backend = "builtin-fallback"
            self.low_trust = True
            # The fallback is capped at 0.75 by construction and scores lower
            # than fastText on the same text, so applying the same floor would
            # reject everything and report a corpus as entirely unidentifiable.
            self.floor = 0.30 if floor is None else floor

    def detect(self, text: str) -> LangResult:
        stripped = text.strip()
        script = dominant_script(stripped)
        if len(stripped) < 12:
            # Too short to be worth an answer. Saying "unknown" here is more
            # useful than a coin flip presented as a result.
            return LangResult("unknown", 0.0, script, self.low_trust)

        if self._detect is not None:
            try:
                # fastText chokes on embedded newlines.
                res = self._detect(stripped.replace("\n", " ")[:2000], low_memory=True)
                lang, score = res["lang"], float(res["score"])
            except Exception:
                return LangResult("unknown", 0.0, script, self.low_trust)
        else:
            lang, score = _fallback_detect(stripped, script)

        if score < self.floor:
            return LangResult("unknown", score, script, self.low_trust)
        return LangResult(lang, score, script, self.low_trust)

    def script_mismatch(self, result: LangResult) -> bool:
        expected = EXPECTED_SCRIPT.get(result.lang)
        if not expected or result.script in ("None", "Other"):
            return False
        return result.script not in expected


# --------------------------------------------------------------------------
# Fallback detector.
#
# Character-profile scoring over a handful of languages. This exists so the tool
# still runs on a cluster where no extra wheels can be installed. It is not a
# replacement, and everything it returns is flagged low_trust.
# --------------------------------------------------------------------------

_PROFILES: dict[str, tuple[str, tuple[str, ...]]] = {
    # language: (distinctive characters, common function words)
    "tr": ("ıİğĞşŞçÇöÖüÜ", (" bir ", " ve ", " bu ", " için ", " değil ", " ile ", " daha ")),
    "en": ("", (" the ", " and ", " of ", " to ", " is ", " that ", " it ")),
    "de": ("äöüß", (" der ", " die ", " und ", " das ", " ist ", " nicht ", " ein ")),
    "fr": ("àâçéèêëîïôùûü", (" le ", " la ", " et ", " les ", " des ", " est ", " une ")),
    "es": ("áéíóúñ¿¡", (" el ", " la ", " de ", " que ", " y ", " los ", " en ")),
    "it": ("àèéìòù", (" il ", " di ", " che ", " la ", " per ", " non ", " una ")),
    "pt": ("ãõáéíóúç", (" de ", " que ", " não ", " uma ", " com ", " para ")),
    "ru": ("абвгдежзийклмнопрстуфхцчшщъыьэюя", (" и ", " в ", " не ", " на ", " что ")),
    "ar": ("ابتثجحخدذرزسشصضطظعغفقكلمنهوي", (" في ", " من ", " على ", " الى ")),
}


def _fallback_detect(text: str, script: str) -> tuple[str, float]:
    lowered = unicodedata.normalize("NFC", text.lower())
    padded = f" {lowered} "
    scores: dict[str, float] = {}
    for lang, (chars, words) in _PROFILES.items():
        char_hits = sum(1 for c in chars if c in lowered) / (len(chars) or 1)
        word_hits = sum(1 for w in words if w in padded) / len(words)
        scores[lang] = 0.4 * char_hits + 0.6 * word_hits
    if not scores:
        return "unknown", 0.0
    best = max(scores, key=lambda k: scores[k])
    score = scores[best]
    # The fallback is deliberately not allowed to express high confidence.
    return (best, min(score, 0.75)) if score > 0.15 else ("unknown", score)
