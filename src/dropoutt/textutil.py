"""Text normalization and shingling.

Shared by deduplication, contamination scanning and several hygiene checks, so
the definitions live in one place and every check agrees on what "the same text"
means.

Two n-gram hash families live here and they are deliberately different.

``eval_ngram_hashes`` must stay bit-identical forever: the shipped contamination
indices are files full of these hashes and nothing can re-derive them, because an
index stores hashes and never the benchmark text. So it keeps BLAKE2b over the
joined n-gram string, and the only speed available is to encode each word once
instead of once per n-gram it appears in.

``shingle_hashes`` feeds MinHash and never leaves the process, so it is free to
use whatever is fastest that is deterministic across machines. It hashes each
*word* once through a bounded cache — word frequency is Zipfian, so almost every
lookup is a hit — and then combines words into n-gram hashes with a vectorised
polynomial over uint64, which removes one Python-level join, hash and set
insertion per shingle.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
#: ``_PUNCT`` then ``_WS`` in one pass. Any run of characters that is neither a
#: word character nor whitespace collapses to a single space, and so does any run
#: of whitespace, so a single ``[^\w]+`` substitution produces the same string as
#: applying both in sequence.
_NONWORD = re.compile(r"[^\w]+", re.UNICODE)

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
#: ``str.translate`` walks a mapping per character and costs about twenty
#: nanoseconds each. This test costs under one, and on a corpus with no Turkish
#: uppercase — which is most of them, and all of the English ones — the
#: translation is a no-op anyway.
_TR_UPPER = re.compile("[IİÇĞÖŞÜ]")


def _tr_lower(text: str) -> str:
    return text.translate(TR_LOWER) if _TR_UPPER.search(text) else text



def normalize_ws(text: str) -> str:
    """Collapse whitespace. Used for whitespace-insensitive duplicate detection.

    ``str.split`` rather than ``_WS.sub``: the two agree on every codepoint in
    Unicode — ``re``'s ``\\s`` and ``str.isspace`` classify identically — and
    split-then-join is about four times faster because it never leaves C. This
    runs three times per record in the duplicate checks.
    """
    return " ".join(text.split())


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization for near-duplicate comparison.

    Turkish-aware lowercasing, punctuation stripped, whitespace collapsed. This
    matches what the delta-max platform already does, so the CLI and the
    platform agree on what counts as a duplicate.
    """
    text = unicodedata.normalize("NFC", text)
    return _NONWORD.sub(" ", _tr_lower(text).lower()).strip()


def dedup_words(text: str) -> list[str]:
    """The normalized word sequence both shingling families are built from.

    One function so the normalization runs once per record rather than once per
    consumer. Near-duplicate detection and contamination scanning used to each
    call :func:`normalize_for_dedup` on the same string.
    """
    return _NONWORD.sub(
        " ", _tr_lower(unicodedata.normalize("NFC", text)).lower()
    ).split()


def word_shingles(text: str, n: int = 5) -> list[str]:
    """Word n-grams, the FineWeb MinHash unit."""
    words = dedup_words(text)
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

_blake = hashlib.blake2b


def hash64(s: str) -> int:
    return int.from_bytes(_blake(s.encode("utf-8"), digest_size=8).digest(), "big")


#: Word hashes are memoised because word frequency is Zipfian: a 50k-record
#: corpus makes millions of lookups against a vocabulary of tens of thousands, so
#: nearly every one is a dict hit rather than a hash. Bounded so a corpus of
#: random tokens cannot grow it without limit; when the cap is reached the cache
#: simply stops taking new entries rather than evicting, since the entries
#: already in it are by construction the frequent ones.
_WORD_HASH: dict[str, int] = {}
_WORD_HASH_CAP = 1 << 21

_U64 = np.uint64
_MASK64 = (1 << 64) - 1

#: Odd 64-bit multiplier used to fold word hashes into an n-gram hash.
_POLY = 0x9E3779B97F4A7C15


def _word_hash(word: str) -> int:
    h = _WORD_HASH.get(word)
    if h is None:
        h = int.from_bytes(_blake(word.encode("utf-8"), digest_size=8).digest(), "big")
        if len(_WORD_HASH) < _WORD_HASH_CAP:
            _WORD_HASH[word] = h
    return h


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Finaliser that breaks the linearity of the polynomial combination.

    Without it, two n-grams differing by a swap of adjacent words could collide
    far more often than 64-bit hashing implies, which would show up as inflated
    Jaccard estimates.
    """
    x = x ^ (x >> _U64(30))
    x = x * _U64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> _U64(27))
    x = x * _U64(0x94D049BB133111EB)
    return x ^ (x >> _U64(31))


def shingle_hashes(words: list[str], n: int = 5) -> np.ndarray:
    """Distinct 64-bit hashes of the word n-grams, sorted.

    Internal to MinHash. The values are deterministic across machines and
    versions of Python but are not an interchange format, so this is free to be
    fast rather than to match any file on disk.
    """
    count = len(words)
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    if count < n:
        return np.array([hash64(" ".join(words))], dtype=np.uint64)

    get = _WORD_HASH.get
    wh = np.fromiter(
        (h if (h := get(w)) is not None else _word_hash(w) for w in words),
        dtype=np.uint64,
        count=count,
    )
    with np.errstate(over="ignore"):
        acc = wh[: count - n + 1].copy()
        mult = _U64(_POLY)
        for k in range(1, n):
            acc = acc * mult + wh[k : count - n + 1 + k]
        return np.unique(_splitmix64(acc))


def hashed_shingles(text: str, n: int = 5) -> list[int]:
    """Word n-grams hashed to 64-bit integers, deduplicated."""
    return [int(x) for x in shingle_hashes(dedup_words(text), n)]


def token_ngrams(tokens: list[str], n: int = 8) -> list[str]:
    """Token n-grams, the contamination-detection unit (Tulu 3 uses 8)."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def eval_ngram_hashes(words: list[str], n: int = 8) -> np.ndarray:
    """BLAKE2b-64 over each word n-gram, sorted and deduplicated.

    Frozen by the shipped contamination indices: every ``.idx`` file is a table
    of exactly these numbers and there is no text left anywhere to recompute
    them from. So the *values* cannot change; only how they are produced can.

    The whole word list is joined and encoded once, and each n-gram is a
    ``memoryview`` slice of that one buffer — because the joined form of words
    ``i..i+n`` is literally a substring of the joined form of all of them. That
    replaces one list slice, one join and n string encodes per n-gram, on the
    single most-called function in a scan. Digests are concatenated and read
    back with ``frombuffer`` rather than converted one at a time.

    A word containing whitespace would break the offsets, so the separator
    count is checked against the word count and the slow path handles the
    mismatch. :func:`dedup_words` cannot produce one; a caller passing its own
    list can.
    """
    count = len(words)
    if count == 0:
        return np.empty(0, dtype=np.uint64)
    blake = _blake
    joined = " ".join(words)
    buf = joined.encode("utf-8")
    if count < n:
        return np.array(
            [int.from_bytes(blake(buf, digest_size=8).digest(), "big")], dtype=np.uint64
        )

    sep = np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 0x20)
    if sep.size != count - 1:
        return _eval_ngram_hashes_slow(words, n)

    starts = np.empty(count, dtype=np.int64)
    starts[0] = 0
    starts[1:] = sep + 1
    ends = np.empty(count, dtype=np.int64)
    ends[:-1] = sep
    ends[-1] = len(buf)

    view = memoryview(buf)
    # Both slices hold `count - n + 1` entries by construction — one per n-gram
    # position — so checking that per record would be pure overhead.
    starts_of_grams = starts[: count - n + 1].tolist()
    ends_of_grams = ends[n - 1 :].tolist()
    digests = b"".join([
        blake(view[start:end], digest_size=8).digest()
        for start, end in zip(starts_of_grams, ends_of_grams, strict=False)
    ])
    return np.unique(np.frombuffer(digests, dtype=">u8").astype(np.uint64))


def _eval_ngram_hashes_slow(words: list[str], n: int) -> np.ndarray:
    """The definition, for word lists the buffer trick cannot index.

    Encoding each word once and joining bytes is identical to encoding the
    joined string, because UTF-8 encoding distributes over concatenation.
    """
    parts = [w.encode("utf-8") for w in words]
    join = b" ".join
    blake = _blake
    frombytes = int.from_bytes
    out = [
        frombytes(blake(join(parts[i : i + n]), digest_size=8).digest(), "big")
        for i in range(len(words) - n + 1)
    ]
    return np.unique(np.array(out, dtype=np.uint64))


def has_mojibake(text: str) -> bool:
    return any(m in text for m in MOJIBAKE_MARKERS)


def control_chars(text: str) -> list[str]:
    return _CONTROL.findall(text)


def has_control_chars(text: str) -> bool:
    """Whether any control character is present, without materialising the list.

    Callers that only branch on presence used to build a list of every match on
    every record.
    """
    return _CONTROL.search(text) is not None


def needs_nfc(text: str) -> bool:
    """True when the text is not in NFC form.

    Matters for Turkish and other diacritic-heavy languages: the same visible
    word can tokenize into different ids depending on composition form, so a
    mixed-form corpus fragments its own vocabulary.
    """
    return not unicodedata.is_normalized("NFC", text)


def repetition_ratio(text: str, n: int = 10) -> float:
    """Fraction of n-gram positions that repeat a previously seen n-gram.

    High values indicate a degenerate record: a stuck generation loop, or a page
    of repeated boilerplate.

    The n-grams are tuples of word references rather than joined strings. Joining
    built one throwaway string per position, which on a corpus of long responses
    is the single most allocated object in the scan.
    """
    words = text.split()
    total = len(words) - n + 1
    if len(words) < n * 2:
        return 0.0
    # The slices are staggered on purpose and each is one word shorter than the
    # last; zip stopping at the shortest is what makes these n-grams.
    grams = zip(*(words[i:] for i in range(n)), strict=False)
    return 1.0 - (len(set(grams)) / total)


def excerpt(text: str, limit: int = 220) -> str:
    """A short, single-line preview for evidence display.

    Whitespace is collapsed over a slice rather than the whole record, because
    on a corpus of long documents this ran over megabytes to produce two hundred
    characters. The slice is widened if it collapsed to less than the limit,
    which is the only case where the shortcut could lose text.
    """
    head = text[: limit * 8]
    flat = " ".join(head.split())
    if len(flat) < limit and len(head) < len(text):
        flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Surface shape
# --------------------------------------------------------------------------

_SURFACE_TABLE: np.ndarray | None = None
_SPACE = 1
_ALPHA = 2


def _surface_table() -> np.ndarray:
    """Codepoint -> whitespace / letter / other, for the Basic Multilingual Plane.

    Built once, in about fifteen milliseconds, and then reused for every record.
    A per-character Python loop over a two-thousand-character record was the
    third most expensive thing in the atlas path.
    """
    global _SURFACE_TABLE
    if _SURFACE_TABLE is None:
        table = np.zeros(0x10000, dtype=np.uint8)
        for cp in range(0x10000):
            ch = chr(cp)
            if ch.isspace():
                table[cp] = _SPACE
            elif ch.isalpha():
                table[cp] = _ALPHA
        _SURFACE_TABLE = table
    return _SURFACE_TABLE


def codepoints(text: str) -> np.ndarray:
    """Text as a uint32 array of codepoints, or an empty array if undecodable."""
    if not text:
        return np.empty(0, dtype=np.uint32)
    try:
        raw = text.encode("utf-32-le", "surrogatepass")
    except UnicodeEncodeError:  # pragma: no cover - defensive
        return np.empty(0, dtype=np.uint32)
    return np.frombuffer(raw, dtype=np.uint32)


def non_whitespace_prefix(text: str) -> np.ndarray:
    """``out[i]`` is how many non-whitespace characters sit in ``text[:i]``.

    One array answers "how much of this text do these ranges cover", for any
    number of ranges, with two lookups each. The alternative is a generator per
    range over a slice of the text, which is what the file sniffer used to do
    over half a megabyte at a time.
    """
    codes = codepoints(text)
    space = np.zeros(codes.size, dtype=bool)
    if codes.size:
        bmp = codes < 0x10000
        space[bmp] = _surface_table()[codes[bmp]] == _SPACE
    out = np.empty(codes.size + 1, dtype=np.int64)
    out[0] = 0
    np.cumsum(~space, out=out[1:])
    return out


def surface_shares(text: str) -> tuple[float, float]:
    """Whitespace share and non-letter share.

    Non-letter counts everything that is neither whitespace nor a letter, which
    is what separates markup, base64 and log lines from prose.
    """
    if not text:
        return 0.0, 0.0
    codes = codepoints(text)
    n = int(codes.size)
    if n == 0:
        return 0.0, 0.0
    # Astral characters are neither whitespace nor letters in every case this
    # measure is used to separate, so they fall into the "other" bucket, which
    # is where the per-character version put emoji and symbols too.
    bmp = codes[codes < 0x10000]
    kinds = _surface_table()[bmp]
    ws = int(np.count_nonzero(kinds == _SPACE))
    alpha = int(np.count_nonzero(kinds == _ALPHA))
    return ws / n, (n - ws - alpha) / n
