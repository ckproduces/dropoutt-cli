"""Process-pool workers for the per-record stages of an Atlas build.

Kept apart from ``build_atlas_v2`` on purpose: the pool starts children with
``spawn``, which imports the module that owns the worker function, and the
builder's module is heavy to import and must never run twice.

Two entry points. :func:`detect_chunk` detects languages only; it serves the
IDF pass and any caller that has already extracted its text. :func:`prepare_chunk`
does every per-record stage of ingest that does not depend on other records --
extraction, the dedup digest, language detection, the encoder input policy and
tokenization -- so the builder's main process keeps only the order-dependent
work: the dedup set, pooling, the within-batch semantic dedup and the append.
Profiled on the atlas-v3 corpus, those per-record stages were four fifths of a
single-process ingest.
"""

from __future__ import annotations

import hashlib

_DETECTOR = None
_TOKENIZER = None
_INPUT = None
_MAX_CHARS = 2_000
_MAX_TOKENS = 512
_MIN_CHARS = 80


def init(
    tokenizer_path: str | None = None,
    encoder_input: dict | None = None,
    max_chars: int = 2_000,
    max_tokens: int = 512,
    min_chars: int = 80,
) -> None:
    global _DETECTOR, _TOKENIZER, _INPUT, _MAX_CHARS, _MAX_TOKENS, _MIN_CHARS
    from dropoutt.langid import LanguageDetector

    _DETECTOR = LanguageDetector()
    if tokenizer_path is not None:
        from tokenizers import Tokenizer

        from dropoutt.atlas.textnorm import EncoderInput

        _TOKENIZER = Tokenizer.from_file(tokenizer_path)
        # The phrase table only matters inside the pool, which stays in the
        # main process; text preparation needs the flags alone.
        declared = dict(encoder_input or {})
        declared["phrase_count"] = 0
        _INPUT = EncoderInput.from_artifact(declared)
    _MAX_CHARS, _MAX_TOKENS, _MIN_CHARS = int(max_chars), int(max_tokens), int(min_chars)


def detect_chunk(texts: list[str]) -> list[tuple[str, float]]:
    assert _DETECTOR is not None, "pool worker was not initialised"
    return [(r.lang, float(r.confidence)) for r in _DETECTOR.detect_many(texts)]


def prepare_chunk(rows: list[tuple[str, bool]]):
    """Per-record ingest stages for one batch of ``(raw text, is_code)`` rows.

    Returns, row for row, ``texts`` (the extracted text, or ``None`` for a row
    the length gate drops), ``digests`` (the dedup hash, 0 where dropped), and,
    for the surviving rows in order, ``detections`` as ``(language,
    confidence)`` and ``token_ids`` as int32 arrays already windowed to
    ``max_tokens``. Every value is what the single-process builder computed.
    """
    import numpy as np

    from dropoutt.atlas.embed import select_token_windows
    from dropoutt.atlas.extract import extract_text

    assert _DETECTOR is not None and _TOKENIZER is not None and _INPUT is not None, (
        "prepare worker was not initialised with a tokenizer"
    )
    texts: list[str | None] = []
    digests: list[int] = []
    for raw, is_code in rows:
        text, _ = extract_text(raw, detected_format="code" if is_code else None)
        text = text[:_MAX_CHARS]
        if len(text) < _MIN_CHARS:
            texts.append(None)
            digests.append(0)
            continue
        digest = hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()
        texts.append(text)
        digests.append(int.from_bytes(digest, "little"))
    kept = [text for text in texts if text is not None]
    detections = (
        [(r.lang, float(r.confidence)) for r in _DETECTOR.detect_many(kept)] if kept else []
    )
    prepared = [_INPUT.prepare(text) for text in kept] if _INPUT.active else kept
    encode = getattr(_TOKENIZER, "encode_batch_fast", None) or _TOKENIZER.encode_batch
    encodings = encode(prepared, add_special_tokens=False) if prepared else []
    token_ids = [
        np.asarray(select_token_windows(encoding.ids, _MAX_TOKENS), dtype=np.int32)
        for encoding in encodings
    ]
    return texts, digests, detections, token_ids
