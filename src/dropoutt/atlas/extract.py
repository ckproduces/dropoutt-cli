"""Format detection and content extraction for atlas embeddings.

Static embeddings average over tokens. In JSON, braces and key names outnumber
content, so the vector encodes "this is JSON" rather than the subject. Same for
CSV, HTML, markdown, and code. Extraction emits natural-language content only;
``detected_format`` travels as separate metadata and never enters the vector.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from ..compat import json_loads

#: Formats the detector recognises. Anything else is ``unknown``.
FORMATS = ("json", "jsonl", "csv", "tsv", "markdown", "html", "code", "plain", "unknown")

#: Drop extracted text shorter than this — embeddings of stubs are noise.
MIN_EXTRACTED_CHARS = 80

#: JSON/JSONL string values must clear this to count as content.
MIN_STRING_CHARS = 24
MIN_STRING_WORDS = 4

_HTML_BLOCK = re.compile(
    r"(?is)<(script|style|nav|footer|header|noscript)[^>]*>.*?</\1>"
)
_HTML_TAG = re.compile(r"(?s)<[^>]+>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MD_EMPH = re.compile(r"[*_`~]{1,3}")
_MD_HEADING = re.compile(r"(?m)^#{1,6}\s+")
_CODE_COMMENT_LINE = re.compile(r"(?m)^\s*(#|//|--)\s?(.*)$")
# Bounded, non-backtracking-ish forms. Nested ``.*?`` over multi-KB LaTeX was
# burning minutes of CPU after braces mis-triggered the code detector.
_CODE_BLOCK_COMMENT = re.compile(
    r"/\*[^*]{0,2000}\*/|'''[^']{0,2000}'''|\"\"\"[^\"]{0,2000}\"\"\""
)
_CODE_STRING = re.compile(
    r"'(?:\\.|[^'\\]){0,400}'|\"(?:\\.|[^\"\\]){0,400}\"|`(?:\\.|[^`\\]){0,400}`"
)
_ID_LIKE = re.compile(r"^[A-Za-z0-9_\-]{8,}$")
_MOSTLY_NUMERIC = re.compile(r"^[\d\s.,+\-eE/$%]+$")


def detect_format(raw: str | bytes, *, filename: str = "") -> str:
    """Return a format label, or ``unknown`` rather than guessing."""
    name = filename.lower()
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        return "jsonl"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".tsv"):
        return "tsv"
    if name.endswith((".md", ".markdown")):
        return "markdown"
    if name.endswith((".html", ".htm")):
        return "html"
    if name.endswith((
        ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp",
        ".c", ".h", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".r",
    )):
        return "code"

    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    sample = text[:8000].strip()
    if not sample:
        return "unknown"

    if sample[0] in "{[":
        try:
            json_loads(sample if sample[-1] in "}]" else sample.split("\n", 1)[0])
            return "json"
        except Exception:
            first = sample.split("\n", 1)[0]
            try:
                json_loads(first)
                return "jsonl"
            except Exception:
                pass

    lines = [ln for ln in sample.splitlines() if ln.strip()][:20]
    if len(lines) >= 3 and sum("\t" in ln for ln in lines) >= len(lines) * 0.8:
        return "tsv"
    if len(lines) >= 3 and sum(ln.count(",") >= 2 for ln in lines) >= len(lines) * 0.8:
        return "csv"
    if "<html" in sample[:500].lower() or sample.lstrip().lower().startswith("<!doctype html"):
        return "html"
    if sample.lstrip().startswith("#") and ("\n## " in sample or "\n# " in sample):
        return "markdown"
    if _looks_like_code(sample):
        return "code"
    if _mostly_prose(sample):
        return "plain"
    return "unknown"


def extract_text(
    raw: str | bytes,
    *,
    detected_format: str | None = None,
    filename: str = "",
) -> tuple[str, str]:
    """Return ``(extracted_text, detected_format)``.

    Empty string means the record should be dropped before embedding.
    """
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    fmt = detected_format or detect_format(text, filename=filename)
    if fmt == "json":
        out = _extract_json(text)
    elif fmt == "jsonl":
        out = _extract_jsonl(text)
    elif fmt in ("csv", "tsv"):
        out = _extract_delimited(text, "\t" if fmt == "tsv" else ",")
    elif fmt == "html":
        out = _extract_html(text)
    elif fmt == "markdown":
        out = _extract_markdown(text)
    elif fmt == "code":
        out = _extract_code(text)
    elif fmt == "plain":
        out = text.strip()
    else:
        out = text.strip()
    out = _collapse(out)
    if len(out) < MIN_EXTRACTED_CHARS:
        return "", fmt
    return out, fmt


def extract_from_fields(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str]:
    """Extract natural-language content from named dataset fields.

    Used by the atlas build when the source is already structured records.
    Detects format per field value when a value looks like JSON/HTML/code.
    """
    parts: list[str] = []
    formats: list[str] = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        if isinstance(v, str):
            text, fmt = extract_text(v)
            if text:
                parts.append(text)
                formats.append(fmt)
        elif isinstance(v, (dict, list)):
            text = _strings_from_obj(v)
            if text:
                parts.append(text)
                formats.append("json")
    joined = _collapse("\n".join(parts))
    if len(joined) < MIN_EXTRACTED_CHARS:
        return "", "unknown"
    # Majority format; ties fall to plain.
    fmt = max(set(formats), key=formats.count) if formats else "plain"
    return joined[:4000], fmt


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    try:
        obj = json_loads(text)
    except Exception:
        return ""
    return _strings_from_obj(obj)


def _extract_jsonl(text: str) -> str:
    parts: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json_loads(line)
        except Exception:
            continue
        piece = _strings_from_obj(obj)
        if piece:
            parts.append(piece)
    return "\n".join(parts)


def _strings_from_obj(obj: Any) -> str:
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            s = x.strip()
            if (
                len(s) >= MIN_STRING_CHARS
                and len(s.split()) >= MIN_STRING_WORDS
                and not _MOSTLY_NUMERIC.match(s)
                and not _ID_LIKE.match(s)
            ):
                out.append(s)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x[:50]:
                walk(v)

    walk(obj)
    return "\n".join(out)


def _extract_delimited(text: str, delimiter: str) -> str:
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        rows = list(reader)
    except Exception:
        return ""
    if len(rows) < 2:
        return ""
    header, body = rows[0], rows[1:]
    if not body:
        return ""
    keep: list[int] = []
    for i, _name in enumerate(header):
        col = [r[i] for r in body if i < len(r)]
        if not col:
            continue
        textish = 0
        for raw_cell in col[:40]:
            cell = (raw_cell or "").strip()
            if (
                len(cell) >= 8
                and not _MOSTLY_NUMERIC.match(cell)
                and not _ID_LIKE.match(cell)
                and any(ch.isalpha() for ch in cell)
            ):
                textish += 1
        if textish >= max(1, min(5, len(col[:40]) // 3)):
            keep.append(i)
    if not keep:
        return ""
    parts: list[str] = []
    for row in body[:200]:
        for i in keep:
            if i < len(row) and row[i].strip():
                parts.append(row[i].strip())
    return "\n".join(parts)


def _extract_html(text: str) -> str:
    cleaned = _HTML_BLOCK.sub(" ", text)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    return cleaned


def _extract_markdown(text: str) -> str:
    t = _MD_IMAGE.sub(r"\1", text)
    t = _MD_LINK.sub(r"\1", t)
    t = _MD_HEADING.sub("", t)
    t = _MD_EMPH.sub("", t)
    return t


def _extract_code(text: str) -> str:
    parts: list[str] = []
    for m in _CODE_BLOCK_COMMENT.finditer(text):
        body = m.group(0)
        body = body.strip("#'\"/ \n")
        if len(body) >= MIN_STRING_CHARS:
            parts.append(body)
    for m in _CODE_COMMENT_LINE.finditer(text):
        body = (m.group(2) or "").strip()
        if len(body) >= 12:
            parts.append(body)
    for m in _CODE_STRING.finditer(text):
        body = m.group(0)[1:-1].strip()
        if len(body) >= MIN_STRING_CHARS and len(body.split()) >= MIN_STRING_WORDS:
            parts.append(body)
    return "\n".join(parts)


def _looks_like_code(sample: str) -> bool:
    # Do not treat bare ``{`` as a code signal — LaTeX and JSON both use it and
    # were being routed into the code extractor.
    markers = (
        "def ", "function ", "class ", "import ", "package ",
        "=>", "};", "public ", "const ", "let ", "var ", "fn ",
    )
    hits = sum(1 for m in markers if m in sample)
    return hits >= 3 and sample.count("\n") >= 3


def _mostly_prose(sample: str) -> bool:
    if not sample:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in sample)
    return letters / len(sample) > 0.75


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
