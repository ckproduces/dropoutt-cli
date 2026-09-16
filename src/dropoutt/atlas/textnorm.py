"""What the atlas encoder reads, and how much each token it reads counts.

The encoder is a static table: a record's vector is the SIF-weighted mean of
its token rows. Nothing in that mean knows what a word *means* in context, so
anything that fills a record with distinctive tokens without saying what the
record is about decides where the record lands. Audited on atlas-v3 (13 Sep
2026) by placing members back after removing one surface feature at a time,
four such features had built cells of their own:

* **ALL-CAPS text.** ``INFORMATIONEN ÜBER DATENSCHUTZERKLÄRUNG`` tokenises into
  thirteen capital-letter fragments and no word; its sentence-cased spelling is
  four real words. Sentence-casing a cell of shouted web copy moved every one
  of its members to a subject cell.
* **Table rules.** Pipe-delimited infoboxes and ``---`` separators. Removing the
  pipes moved 85% of a "tables about anything" cell elsewhere.
* **Single-letter and two-letter pieces.** Initials, abbreviations and the
  leading fragment of rare names (``▁T``, ``▁P``, ``TT``) pulled records into
  districts that shared nothing but a first letter.
* **Instruction templates.** One translated prompt -- "generate a more complex
  version of this sentence" -- filled ten districts, split by the language of
  the template rather than by what the sentences were about.

:class:`EncoderInput` is the frozen answer to those four. Two parts change the
text before tokenisation (fold runs of ALL-CAPS words, drop table rules); the
rest scale token weights inside the pool (short cased pieces, symbol-only and
digit-only tokens, and every token covered by a phrase that repeats across a
large share of one source's documents). Scaling rather than deleting keeps a
record made only of such tokens placeable: pooling divides by the weight sum,
so its direction is unchanged when there is nothing better to read.

The policy is part of the coordinate system. An artifact records the policy it
was built with, and :class:`~dropoutt.atlas.apply.Atlas` binds the same policy
to the encoder at scan time. Artifacts that record none were built from raw
text and keep reading raw text: version 0 is byte-identical to the encoder
before this module existed.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np

#: Newest policy this code can apply. An artifact declaring a later one was
#: built by a newer dropoutt and cannot be read faithfully here.
LATEST_VERSION = 1

#: Phrase length, in tokens, for template detection.
PHRASE_N = 4
#: A phrase is template text when it occurs in at least this share of one
#: source's sampled documents. Measured on a 5M-record draw of the atlas-v3
#: corpus: the translated rewriting template sits at 37-83% of its source,
#: while the most repeated ordinary phrases of prose stay under 15%
#: ("References" in English Wikipedia is 14.6%, ". It is" 11.9%) and a
#: subject phrase such as "is a village in" at 4.6%.
PHRASE_SHARE = 0.10
#: ...and in at least this many of them, so a small sample cannot promote a
#: phrase on two coincidences.
PHRASE_MIN_DOCS = 20
#: Sources with fewer sampled documents than this contribute no phrases.
PHRASE_MIN_SOURCE_DOCS = 200
#: Documents per source a build samples to find template phrases.
PHRASE_DOCS_PER_SOURCE = 4_000

#: Word boundary helpers: a "letter" is any word character that is not a digit
#: or underscore, which is how Python's ``re`` spells a Unicode letter.
_LETTER = r"[^\W\d_]"
#: Separators allowed inside a run of capitalised words.
_GAP = r"[\W\d_]+"
#: Horizontal table rules and runs of decoration characters.
_RULES = re.compile(r"[¦│┃]+|[-=_*#~+.:]{3,}|[─━═]{2,}")
#: Two rule characters side by side, the cheapest sign a text has any. A scan
#: with no hit skips the substitution, which is two thirds of its cost.
_RULE_HINT = re.compile(r"[¦│┃]|[-=_*#~+.:─━═]{2}")


@cache
def _upper_class() -> str:
    """A character class of every uppercase letter in the Basic Multilingual Plane.

    Python's ``re`` has no ``\\p{Lu}``, and an ASCII ``[A-Z]`` would miss the
    Cyrillic, Greek and accented capitals that half of this corpus is written
    in. Built once, lazily, from ``str.isupper``: about twenty milliseconds, and
    only on the first record that is actually normalised.
    """
    ranges: list[tuple[int, int]] = []
    for code in range(0x10000):
        char = chr(code)
        if char.isupper() and not char.islower():
            if ranges and ranges[-1][1] == code - 1:
                ranges[-1] = (ranges[-1][0], code)
            else:
                ranges.append((code, code))
    parts = []
    for start, stop in ranges:
        if start == stop:
            parts.append(re.escape(chr(start)))
        else:
            parts.append(f"{re.escape(chr(start))}-{re.escape(chr(stop))}")
    return "[" + "".join(parts) + "]"


@cache
def _caps_patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    upper = _upper_class()
    word = rf"(?<!{_LETTER}){upper}{{2,}}(?!{_LETTER})"
    return re.compile(word), re.compile(rf"{word}(?:{_GAP}{word}){{2,}}")


def fold_caps_runs(text: str) -> str:
    """Sentence-case every run of three or more ALL-CAPS words.

    Three rather than one so acronyms survive: ``NASA``, ``EU`` and ``HTML`` in
    ordinary prose are words the vocabulary knows in capitals, while a heading
    or a page of shouted copy is a run. Each word keeps its first letter, so
    proper nouns in a shouted heading still read as proper nouns.
    """
    word, run = _caps_patterns()
    if not run.search(text):
        return text
    return run.sub(
        lambda match: word.sub(lambda w: w.group(0).capitalize(), match.group(0)),
        text,
    )


def strip_rules(text: str) -> str:
    """Replace table pipes and decoration runs with a space.

    The tokenizer collapses repeated spaces, so a pipe becoming a space and a
    run of pipes becoming one are the same token sequence.
    """
    if "|" in text:
        text = text.replace("|", " ")
    if not _RULE_HINT.search(text):
        return text
    return _RULES.sub(" ", text)


@dataclass(frozen=True)
class EncoderInput:
    """A frozen policy for what the encoder reads. Version 0 reads raw text."""

    version: int = 0
    fold_caps_runs: bool = False
    strip_rules: bool = False
    #: Weight multiplier for cased tokens of one or two characters.
    short_cased_weight: float = 1.0
    #: ...for tokens with no letter or digit in them.
    symbol_weight: float = 1.0
    #: ...for tokens made only of digits.
    digit_weight: float = 1.0
    #: ...for every token inside a known template phrase.
    phrase_weight: float = 1.0
    #: Sorted 64-bit hashes of :data:`PHRASE_N`-token template phrases.
    phrase_hashes: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def active(self) -> bool:
        return self.version > 0

    @property
    def damps_tokens(self) -> bool:
        return self.active and (
            self.short_cased_weight != 1.0
            or self.symbol_weight != 1.0
            or self.digit_weight != 1.0
        )

    @property
    def damps_phrases(self) -> bool:
        return (
            self.active
            and self.phrase_weight != 1.0
            and self.phrase_hashes is not None
            and len(self.phrase_hashes) > 0
        )

    def with_phrases(self, hashes: np.ndarray) -> EncoderInput:
        table = np.unique(np.asarray(hashes, dtype=np.uint64))
        return replace(self, phrase_hashes=table)

    def prepare(self, text: str) -> str:
        """The text the tokenizer is given."""
        if not self.active:
            return text
        if self.strip_rules:
            text = strip_rules(text)
        if self.fold_caps_runs:
            text = fold_caps_runs(text)
        return text

    def declaration(self) -> dict:
        """What an artifact records. The phrase table ships as an array beside it."""
        hashes = self.phrase_hashes
        count = 0 if hashes is None else len(hashes)
        digest = ""
        if count:
            digest = hashlib.blake2b(
                np.asarray(hashes, dtype=np.uint64).tobytes(), digest_size=16
            ).hexdigest()
        return {
            "version": self.version,
            "fold_caps_runs": self.fold_caps_runs,
            "strip_rules": self.strip_rules,
            "short_cased_weight": self.short_cased_weight,
            "symbol_weight": self.symbol_weight,
            "digit_weight": self.digit_weight,
            "phrase_weight": self.phrase_weight,
            "phrase_n": PHRASE_N,
            "phrase_count": count,
            "phrase_digest": digest,
        }

    @classmethod
    def from_artifact(
        cls, declared: dict | None, hashes: np.ndarray | None = None
    ) -> EncoderInput:
        """The policy an artifact was built with; raw text when it names none."""
        if not declared:
            return RAW_INPUT
        version = int(declared.get("version", 0))
        if version > LATEST_VERSION:
            raise ValueError(
                f"this atlas was built with encoder input version {version}; "
                f"this dropoutt reads up to {LATEST_VERSION}. Upgrade dropoutt."
            )
        if version == 0:
            return RAW_INPUT
        if int(declared.get("phrase_n", PHRASE_N)) != PHRASE_N:
            raise ValueError("atlas phrase table uses a phrase length this dropoutt cannot read")
        policy = cls(
            version=version,
            fold_caps_runs=bool(declared.get("fold_caps_runs", False)),
            strip_rules=bool(declared.get("strip_rules", False)),
            short_cased_weight=float(declared.get("short_cased_weight", 1.0)),
            symbol_weight=float(declared.get("symbol_weight", 1.0)),
            digit_weight=float(declared.get("digit_weight", 1.0)),
            phrase_weight=float(declared.get("phrase_weight", 1.0)),
        )
        expected = int(declared.get("phrase_count", 0))
        if expected:
            if hashes is None or len(hashes) != expected:
                raise ValueError("atlas declares template phrases but ships no phrase table")
            policy = policy.with_phrases(hashes)
            if policy.declaration()["phrase_digest"] != declared.get("phrase_digest"):
                raise ValueError("atlas phrase table does not match its declared digest")
        return policy


RAW_INPUT = EncoderInput()

#: The policy atlas-v3 is built and read with.
ENCODER_INPUT_V1 = EncoderInput(
    version=1,
    fold_caps_runs=True,
    strip_rules=True,
    short_cased_weight=0.2,
    symbol_weight=0.2,
    digit_weight=0.2,
    phrase_weight=0.1,
)


def _piece_class(piece: str) -> str:
    """``short``, ``symbol``, ``digit`` or ``word`` for one vocabulary entry."""
    surface = piece.lstrip("▁")
    if not surface:
        return "word"
    if not any(ch.isalnum() for ch in surface):
        return "symbol"
    if all(ch.isdigit() for ch in surface):
        return "digit"
    if len(surface) <= 2 and any(ch.isalpha() and ch.lower() != ch.upper() for ch in surface):
        return "short"
    return "word"


_FACTOR_CACHE: dict[tuple[int, int, float, float, float], np.ndarray] = {}


def token_factors(tokenizer, n_rows: int, policy: EncoderInput) -> np.ndarray:
    """Per-token-id weight multipliers for ``policy`` over one vocabulary.

    Only scripts with letter case are touched by the short-piece rule. Chinese,
    Japanese, Arabic, Hebrew, Thai and the Indic scripts spell whole words in one
    or two characters, and damping those would damp their content.
    """
    key = (
        id(tokenizer), int(n_rows),
        policy.short_cased_weight, policy.symbol_weight, policy.digit_weight,
    )
    cached = _FACTOR_CACHE.get(key)
    if cached is not None:
        return cached
    factors = np.ones(int(n_rows), dtype=np.float32)
    weight = {
        "short": policy.short_cased_weight,
        "symbol": policy.symbol_weight,
        "digit": policy.digit_weight,
    }
    for piece, token_id in tokenizer.get_vocab().items():
        if 0 <= token_id < n_rows:
            kind = _piece_class(piece)
            if kind != "word":
                factors[token_id] = weight[kind]
    _FACTOR_CACHE[key] = factors
    return factors


_M0 = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xC2B2AE3D27D4EB4F)
_M2 = np.uint64(0x165667B19E3779F9)


def phrase_hashes(token_ids: np.ndarray) -> np.ndarray:
    """64-bit hash of every :data:`PHRASE_N`-token window of one flat id array.

    ``out[i]`` covers ``token_ids[i:i + 4]``. The caller masks windows that
    cross a document boundary.
    """
    ids = np.asarray(token_ids, dtype=np.uint64)
    if len(ids) < PHRASE_N:
        return np.zeros(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return ((ids[:-3] * _M0 + ids[1:-2]) * _M1 + ids[2:-1]) * _M2 + ids[3:]


def phrase_mask(token_ids: np.ndarray, indptr: np.ndarray, table: np.ndarray) -> np.ndarray:
    """True for every token that sits inside a phrase listed in ``table``."""
    n = len(token_ids)
    covered = np.zeros(n, dtype=bool)
    if n < PHRASE_N or table is None or not len(table):
        return covered
    hashes = phrase_hashes(token_ids)
    lengths = np.diff(np.asarray(indptr, dtype=np.int64))
    doc = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    within = doc[: n - PHRASE_N + 1] == doc[PHRASE_N - 1 :]
    slot = np.searchsorted(table, hashes)
    slot[slot == len(table)] = 0
    hit = within & (table[slot] == hashes)
    if not hit.any():
        return covered
    for offset in range(PHRASE_N):
        covered[offset : offset + len(hit)] |= hit
    return covered


def source_phrases(
    documents: Iterable[np.ndarray],
    *,
    share: float = PHRASE_SHARE,
    min_docs: int = PHRASE_MIN_DOCS,
    min_source_docs: int = PHRASE_MIN_SOURCE_DOCS,
) -> np.ndarray:
    """Template phrases of one source: windows in at least ``share`` of its documents."""
    per_doc = []
    for ids in documents:
        hashes = phrase_hashes(ids)
        if len(hashes):
            per_doc.append(np.unique(hashes))
    if len(per_doc) < min_source_docs:
        return np.zeros(0, dtype=np.uint64)
    values, counts = np.unique(np.concatenate(per_doc), return_counts=True)
    floor = max(int(min_docs), int(np.ceil(share * len(per_doc))))
    return values[counts >= floor]


def fit_phrases(per_source: Iterable[Iterable[np.ndarray]], **kwargs) -> np.ndarray:
    """Union of every source's template phrases, sorted for lookup."""
    found = [source_phrases(docs, **kwargs) for docs in per_source]
    found = [part for part in found if len(part)]
    if not found:
        return np.zeros(0, dtype=np.uint64)
    return np.unique(np.concatenate(found))
