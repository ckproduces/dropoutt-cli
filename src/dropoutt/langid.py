"""Language identification with an honest confidence signal.

Two things this module refuses to do.

It does not report a language without a confidence. Identification is least
reliable exactly where it matters most for this product: short text, and closely
related languages. Turkish against Azerbaijani against Turkmen is the canonical
hard case, and a bag-of-n-grams model will confuse them on a five-word snippet
no matter which implementation is used.

It does not call the fallback backend equivalent to the real one. When
``py3langid`` is missing, a small character-profile detector is used instead; it
covers a handful of languages, is materially less accurate, and every result it
produces is marked low-trust so the report can say so.

**Why not fastText.** ``fasttext-langdetect`` was the backend through 1.0, and
it is a better classifier on short text than what replaced it. It is also a
compiled extension whose publisher does not build a wheel for every supported
interpreter, and a missing wheel is not a degraded install — pip falls through
to compiling it, so a Windows user on CPython 3.14 running ``pip install
dropoutt`` was told to install Microsoft Visual C++ Build Tools. ``py3langid`` is
pure Python over numpy, covers 97 languages, ships its model inside the wheel,
and costs about sixty microseconds a call. Trading a few points of accuracy on
sub-twenty-character strings for an install that cannot fail is the right trade
for a tool whose first impression is the install.

The accuracy that was traded away is bounded and stated rather than hidden: see
:data:`SHORT_TEXT_CHARS` for the length below which this backend's confidence is
discounted, because that is exactly where the difference lives.
"""

from __future__ import annotations

import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

from .compat import HAVE_PY3LANGID
from .textutil import codepoints

#: Below this, we emit "unknown" rather than a language.
DEFAULT_FLOOR = 0.55

#: Text shorter than this is where a bag-of-n-grams classifier is least
#: reliable and most confident, which is the worst combination. py3langid
#: returns normalised posteriors, and on a single word those are routinely above
#: 0.9 for the wrong language ("Merhaba" scores 0.87 German). Confidence is
#: scaled down linearly below this length so the floor does its job.
SHORT_TEXT_CHARS = 60

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


#: One identifier per process, shared by every detector instance. Building it
#: unpickles about a megabyte of numpy arrays; a scan constructs a
#: ``LanguageDetector`` in the parent and again in each worker, and paying that
#: once per worker instead of once per instance is the whole point of the cache.
#: Forked workers inherit it already built, copy-on-write.
_IDENTIFIER: Any = None
_IDENTIFIER_LOCK = threading.Lock()


def _identifier() -> Any:
    """The py3langid model, loaded once. None when the backend is unavailable."""
    global _IDENTIFIER
    if _IDENTIFIER is None:
        with _IDENTIFIER_LOCK:
            if _IDENTIFIER is None:
                from py3langid.langid import MODEL_FILE, LanguageIdentifier

                # norm_probs=True turns the raw log-likelihoods into posteriors
                # that sum to one. Without it the "confidence" is an unbounded
                # negative number, and every threshold in this module — the
                # floor, the confusable gate, the short-text discount — is
                # written against a probability.
                _IDENTIFIER = LanguageIdentifier.from_pickled_model(
                    MODEL_FILE, norm_probs=True
                )
    return _IDENTIFIER


class LanguageDetector:
    """Wraps whichever backend is available."""

    #: How much of a record is fed to identification. The classifier is a bag of
    #: character n-grams, so the tail of a long document adds nothing the head
    #: has not already said, and slicing before ``strip`` keeps a megabyte-long
    #: record from being copied twice.
    HEAD_CHARS = 2000

    def __init__(self, floor: float | None = None) -> None:
        self._model: Any = None
        if HAVE_PY3LANGID:
            self.backend = "py3langid-97"
            self.low_trust = False
            self.floor = DEFAULT_FLOOR if floor is None else floor
        else:
            self.backend = "builtin-fallback"
            self.low_trust = True
            # The fallback is capped at 0.75 by construction and scores lower
            # than the real backend on the same text, so applying the same floor
            # would reject everything and report a corpus as unidentifiable.
            self.floor = 0.30 if floor is None else floor

    def _predict(self, normalized: str) -> tuple[str, float]:
        """One classification, with the short-text discount applied.

        py3langid returns a normalised posterior, which on a handful of
        characters is confident and wrong often enough to matter — it is a
        multinomial naive Bayes over byte n-grams, and a short string simply
        does not carry enough of them. Rather than let the floor be crossed by
        noise, the score is scaled by how much text the decision was actually
        made on. Above :data:`SHORT_TEXT_CHARS` the scale is 1 and this is the
        model's own number.
        """
        if self._model is None:
            self._model = _identifier()
        lang, probability = self._model.classify(normalized)
        score = min(float(probability), 1.0)
        if len(normalized) < SHORT_TEXT_CHARS:
            score *= len(normalized) / SHORT_TEXT_CHARS
        return str(lang), score

    def detect(self, text: str) -> LangResult:
        head = text[: self.HEAD_CHARS + 64].strip() if len(text) > self.HEAD_CHARS else text.strip()
        script = dominant_script(head)
        if len(head) < 12:
            # Too short to be worth an answer. Saying "unknown" here is more
            # useful than a coin flip presented as a result.
            return LangResult("unknown", 0.0, script, self.low_trust)

        if not self.low_trust:
            try:
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
