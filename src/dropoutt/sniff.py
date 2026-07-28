"""Working out whether a plain-text file is actually holding records.

A ``.txt`` file is not evidence of anything. Generation pipelines write JSON to
whatever path they were handed, and the extension survives long after the
contents stop matching it. Read literally, a folder of such files becomes one
corpus document per file: the record count is off by two orders of magnitude,
the profile is inferred as ``corpus``, and every structural check for
conversational data is skipped without ever saying it was skipped, because as
far as the scanner is concerned there was no conversational data.

That is the worst failure shape this tool can have. It is silent, it is
confident, and everything downstream of it is wrong in a way that looks fine.

So text files are sniffed rather than assumed. One scanner covers every framing
seen in practice, because they differ only in what sits *between* the records:

===============  ==========================================================
line-delimited   ``{"messages": ...}`` one per line, JSONL under another name
blank-separated  pretty-printed objects with blank lines between them
fenced           objects wrapped in ```json fences, as an LLM emits them
prefixed         any of the above with generator scaffolding around them
===============  ==========================================================

Rather than special-casing four framings, :func:`iter_json_spans` walks the text
tracking brace depth outside of string literals and yields every balanced
top-level ``{...}`` span. The framings collapse into one algorithm, and the text
lying *between* spans falls out for free — which is what catches scaffolding the
generator was never supposed to emit.

Two rules keep this from firing on real prose.

**A span must parse, and most of them must.** Balanced braces are not evidence;
prose about code contains them. The gate is on how many spans parse as JSON
objects, not on how many were found.

**The spans must account for most of the file.** A blog post quoting one JSON
snippet is prose, and its one span covers 2% of the bytes. A record file's spans
cover nearly all of them. That ratio is the difference, and it is why the
scaffolding measurement and the framing decision come from the same number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

#: How much of the head of a file to read when deciding. Large enough that a
#: few-hundred-KB file is decided on its entirety, small enough that a 4 GB
#: mislabelled dump does not become a memory problem.
PROBE_BYTES = 512 * 1024

#: Of the balanced spans found in the probe, this share must parse as JSON
#: objects. Set below the obvious 0.9 on purpose: malformed records are exactly
#: what this tool exists to report, and a file that is 70% valid records is
#: still a record file with a parse-rate problem, not a prose file.
MIN_PARSE_RATE = 0.6

#: The spans must cover this share of the probe's non-whitespace bytes. This is
#: the test that separates a record file from prose that quotes JSON.
MIN_COVERAGE = 0.5

#: Below this many spans there is nothing to generalise from, and a single
#: object is better described as a JSON file that was misnamed.
MIN_SPANS = 2

#: Text between records long enough to be worth reporting. Shorter runs are
#: fence markers, commas and stray punctuation, which say nothing.
SCAFFOLD_MIN_CHARS = 12


@dataclass(slots=True)
class Framing:
    """How a text file turned out to be holding its records."""

    #: ``"prose"`` when the file is what its extension claims.
    kind: str
    spans: int = 0
    parsed: int = 0
    #: Share of non-whitespace probe bytes covered by the spans.
    coverage: float = 0.0
    #: Distinct runs of text sitting between or before the records. This is
    #: generator scaffolding: control tags, commentary, fence markers.
    scaffolding: list[str] = field(default_factory=list)
    #: True when the probe hit its byte limit, so the numbers describe a head
    #: sample rather than the file.
    truncated: bool = False

    @property
    def is_records(self) -> bool:
        return self.kind != "prose"

    @property
    def parse_rate(self) -> float:
        return self.parsed / self.spans if self.spans else 0.0


def iter_json_spans(text: str) -> list[tuple[int, int]]:
    """Every balanced top-level ``{...}`` span, as ``(start, end)`` offsets.

    Brace depth is tracked outside string literals only, so a ``}`` inside a
    content string does not close a record. Escapes are honoured, because
    ``"\\""`` inside a JSON string would otherwise end the string early and let
    the rest of the record's braces corrupt the depth count.

    Unbalanced tails are dropped rather than guessed at: a record cut off by the
    probe limit is not a record.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1))
                    start = -1
    return spans


def sniff_text(
    text: str, *, truncated: bool = False, max_scaffolding: int = 5
) -> Framing:
    """Decide whether ``text`` is prose or a container of JSON records."""
    spans = iter_json_spans(text)
    if len(spans) < MIN_SPANS:
        return Framing(kind="prose", spans=len(spans), truncated=truncated)

    parsed = 0
    for start, end in spans:
        try:
            if isinstance(json.loads(text[start:end]), dict):
                parsed += 1
        except Exception:
            continue

    non_ws_total = sum(1 for c in text if not c.isspace())
    covered = sum(
        sum(1 for c in text[s:e] if not c.isspace()) for s, e in spans
    )
    coverage = covered / non_ws_total if non_ws_total else 0.0

    framing = Framing(
        kind="prose", spans=len(spans), parsed=parsed,
        coverage=round(coverage, 4), truncated=truncated,
    )
    if parsed / len(spans) < MIN_PARSE_RATE or coverage < MIN_COVERAGE:
        return framing

    framing.kind = _framing_kind(text, spans)
    framing.scaffolding = _between(text, spans, limit=max_scaffolding)
    return framing


def _framing_kind(text: str, spans: list[tuple[int, int]]) -> str:
    """Name the framing, for the report only.

    Reading does not branch on this: the span offsets already say where every
    record is. It exists so the finding can tell the user what their files look
    like in words they will recognise.
    """
    if "```" in text:
        return "fenced"
    one_per_line = sum(
        1 for s, e in spans if "\n" not in text[s:e]
    )
    if one_per_line / len(spans) >= 0.9:
        return "line-delimited"
    return "blank-separated"


def _between(text: str, spans: list[tuple[int, int]], *, limit: int) -> list[str]:
    """Distinct runs of text sitting outside the records.

    Fence markers are stripped first. What survives is content the generator
    emitted around its output — a leaked control tag, a preamble it was told not
    to write — and the user almost certainly does not know it is in there.
    """
    seen: list[str] = []
    edges = [0] + [x for s, e in spans for x in (s, e)] + [len(text)]
    for i in range(0, len(edges) - 1, 2):
        chunk = text[edges[i]:edges[i + 1]]
        for marker in ("```json", "```"):
            chunk = chunk.replace(marker, " ")
        chunk = " ".join(chunk.split())
        if len(chunk) < SCAFFOLD_MIN_CHARS:
            continue
        snippet = chunk[:120]
        if snippet not in seen:
            seen.append(snippet)
        if len(seen) >= limit:
            break
    return seen


def sniff_file(path: str, *, compressed: bool = False) -> Framing:
    """Sniff a file on disk. Never raises; an unreadable file is prose."""
    from .readers import open_maybe_compressed  # noqa: PLC0415

    try:
        with open_maybe_compressed(path, compressed) as fh:
            head = fh.read(PROBE_BYTES + 1)
    except Exception:
        return Framing(kind="prose")
    truncated = len(head) > PROBE_BYTES
    return sniff_text(head[:PROBE_BYTES], truncated=truncated)
