"""Default paragraph-scale chunker for atlas embedding.

Corpus records are usually already sequence-length chunks and pass through.
The chunker exists for users who bring raw documents. Target is ~200–500
tokens; we approximate tokens as whitespace words so the client needs no
tokenizer for this step.
"""

from __future__ import annotations

#: Versioned with the pipeline. Changing this changes ``pipeline_hash``.
CHUNKER_VERSION = "para-v1"
DEFAULT_TARGET_WORDS = 350
DEFAULT_MAX_WORDS = 500
DEFAULT_MIN_WORDS = 40


def chunk_text(
    text: str,
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
    min_words: int = DEFAULT_MIN_WORDS,
) -> list[str]:
    """Split ``text`` into paragraph-scale chunks.

    Short inputs return a single chunk (or empty if below ``min_words`` after
    the caller's extraction gate). Long inputs split on blank lines first,
    then on sentences / hard word caps.
    """
    text = text.strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text] if len(words) >= min_words or len(text) >= 80 else []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = _split_sentences(text)

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    for para in paragraphs:
        n = len(para.split())
        if buf and buf_words + n > max_words:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
        if n > max_words:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            words_p = para.split()
            for i in range(0, len(words_p), target_words):
                piece = " ".join(words_p[i : i + target_words])
                if piece:
                    chunks.append(piece)
            continue
        buf.append(para)
        buf_words += n
        if buf_words >= target_words:
            chunks.append(" ".join(buf))
            buf, buf_words = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.split()) >= min_words or len(c) >= 80]


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for raw_token in text.replace("?", ".").replace("!", ".").split("."):
        token = raw_token.strip()
        if not token:
            continue
        buf.append(token)
        if len(" ".join(buf).split()) >= 40:
            parts.append(". ".join(buf) + ".")
            buf = []
    if buf:
        parts.append(". ".join(buf) + ".")
    return parts or [text]
