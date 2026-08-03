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

import numpy as np

from .compat import HAVE_FASTTEXT_LID
from .textutil import codepoints

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


#: Script names in the order their ids appear in the lookup table below. Index 0
#: is "not a letter, ignore" and the last entry is the catch-all the original
#: per-character loop called "Other".
_SCRIPT_NAMES = ["", *[name for name, _lo, _hi in SCRIPT_RANGES], "Other"]
_OTHER_ID = len(_SCRIPT_NAMES) - 1

_SCRIPT_TABLE: np.ndarray | None = None


def _script_table() -> np.ndarray:
    """Codepoint -> script id, for the Basic Multilingual Plane.

    A table of 65,536 bytes built once, in place of a Python loop that called
    ``ord``, ``isalpha`` and then walked ten ranges for every character of every
    record. Only characters that are letters get a script id, which is what the
    per-character version did by testing ``isalpha`` before the ranges.
    """
    global _SCRIPT_TABLE
    if _SCRIPT_TABLE is None:
        table = np.zeros(0x10000, dtype=np.uint8)
        for cp in range(0x10000):
            if chr(cp).isalpha():
                table[cp] = _OTHER_ID
        for idx, (_name, lo, hi) in enumerate(SCRIPT_RANGES, start=1):
            span = table[lo : hi + 1]
            span[span == _OTHER_ID] = idx
        _SCRIPT_TABLE = table
    return _SCRIPT_TABLE


def dominant_script(text: str) -> str:
    """Which writing system most of the letters in this text belong to.

    Characters outside the Basic Multilingual Plane are ignored rather than
    counted as "Other". Nothing downstream distinguishes the two: the only
    consumer is :meth:`LanguageDetector.script_mismatch`, which treats "Other"
    and "None" identically.
    """
    codes = codepoints(text[:2000])
    if codes.size == 0:
        return "None"
    ids = _script_table()[codes[codes < 0x10000]]
    ids = ids[ids != 0]
    if ids.size == 0:
        return "None"
    return _SCRIPT_NAMES[int(np.bincount(ids, minlength=len(_SCRIPT_NAMES)).argmax())]


class LanguageDetector:
    """Wraps whichever backend is available."""

    #: How much of a record is fed to identification. fastText is a bag of
    #: character n-grams, so the tail of a long document adds nothing the head
    #: has not already said, and slicing before ``strip`` keeps a megabyte-long
    #: record from being copied twice.
    HEAD_CHARS = 2000

    def __init__(self, floor: float | None = None) -> None:
        self._model = None
        if HAVE_FASTTEXT_LID:
            self.backend = "fasttext-lid.176"
            self.low_trust = False
            self.floor = DEFAULT_FLOOR if floor is None else floor
        else:
            self.backend = "builtin-fallback"
            self.low_trust = True
            # The fallback is capped at 0.75 by construction and scores lower
            # than fastText on the same text, so applying the same floor would
            # reject everything and report a corpus as entirely unidentifiable.
            self.floor = 0.30 if floor is None else floor

    def _predict(self, normalized: str) -> tuple[str, float]:
        """One call into fastText's pybind layer.

        ``ftlangdetect.detect`` is not used at scan time. It re-collapses
        whitespace with a regex over text this class has already normalised,
        takes a lock to look up a model in a dict, and allocates a TypedDict per
        call — about seventy microseconds of pure overhead on top of a
        prediction that costs less than that. The model object and the
        normalisation contract are the same either way.
        """
        if self._model is None:
            from ftlangdetect.detect import get_or_load_model

            self._model = get_or_load_model(True)
        predictions = self._model.f.predict(normalized + "\n", 1, 0.0, "strict")
        if not predictions:
            raise RuntimeError("fastText returned no prediction")
        probability, label = predictions[0]
        return label.replace("__label__", ""), min(float(probability), 1.0)

    def detect(self, text: str) -> LangResult:
        head = text[: self.HEAD_CHARS + 64].strip() if len(text) > self.HEAD_CHARS else text.strip()
        script = dominant_script(head)
        if len(head) < 12:
            # Too short to be worth an answer. Saying "unknown" here is more
            # useful than a coin flip presented as a result.
            return LangResult("unknown", 0.0, script, self.low_trust)

        if not self.low_trust:
            try:
                # fastText treats a newline as a record separator and rejects
                # any string containing one, so whitespace is collapsed first.
                lang, score = self._predict(" ".join(head[: self.HEAD_CHARS].split()))
            except Exception:
                return LangResult("unknown", 0.0, script, self.low_trust)
        else:
            lang, score = _fallback_detect(head, script)

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
