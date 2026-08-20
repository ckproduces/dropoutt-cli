"""File readers.

Every reader yields ``RawRecord`` and never raises on bad input. A malformed
line becomes a record carrying its parse error, which downstream turns into a
finding. Erroring out on line 4,000,001 of a 28 GB scan would be the wrong
behaviour: the point of the tool is to report what is wrong with the data.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import lzma
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compat import HAVE_PYARROW, HAVE_ZSTANDARD, json_loads


@dataclass(frozen=True, slots=True)
class ReadSpan:
    """A contiguous piece of one file, and the record index it begins at.

    Exists so the streaming pass can be split across processes without the
    record numbering changing. ``source_index`` appears in evidence, in
    duplicate keys and in the fingerprint, so a shard has to number its records
    exactly as a serial read would, which means knowing how many records precede
    its first byte.

    For line-delimited formats ``start`` and ``end`` are byte offsets and
    ``start`` always sits at the beginning of a line. For columnar formats they
    are row-group indices.
    """

    start: int = 0
    #: Exclusive. -1 means "to the end of the file".
    end: int = -1
    first_index: int = 0

    @property
    def is_whole_file(self) -> bool:
        return self.start == 0 and self.end < 0


WHOLE_FILE = ReadSpan()


@dataclass(slots=True)
class RawRecord:
    payload: Any
    source_file: str
    source_index: int
    #: Set when the record could not be parsed. ``payload`` then holds the raw text.
    error: str | None = None
    #: Raw text as read, kept for text-profile reading and for error evidence.
    raw_text: str = ""
    #: Set when the record is real but shorter than what is on disk, because a
    #: cap was hit. Deliberately not ``error``: a truncated record still parses,
    #: still has a layout and still counts, and routing it through the parse
    #: error path would drop its text and report it as malformed. What it needs
    #: is for the scan to say the file was longer, which the runner does.
    truncated: str | None = None


def open_maybe_compressed(path: str, compressed: bool) -> io.TextIOBase:
    if compressed and path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if compressed and path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if compressed and path.endswith(".xz"):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="replace")
    if compressed and path.endswith(".zst"):
        if not HAVE_ZSTANDARD:
            raise RuntimeError("zstd support not installed; reinstall dropoutt")
        import zstandard

        raw = open(path, "rb")  # noqa: SIM115 - closed with the wrapper this returns
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def is_json_array_file(path: str, compressed: bool = False) -> bool:
    """True when a line-delimited extension is really one JSON array.

    The extension is a claim about the file, not a fact about it. A ``.jsonl``
    whose first character is ``[`` is a pretty-printed array, and reading it a
    line at a time yields one parse error per line: on a real corpus that was
    1,256,362 failures out of 1,371,772 records. The damage does not stop at the
    parse count. Every ``{``, ``},`` and ``"license": "CC BY-SA 3.0",`` becomes a
    record in its own right, so the same fragment repeats a hundred thousand
    times and the duplicate check reports a cluster of 157,045 copies, while the
    fragments are too short to identify and the language breakdown fills up with
    "unknown". One misread file produced three wrong findings.

    Only the head is read, so this costs a single buffered read.
    """
    try:
        with open_maybe_compressed(path, compressed) as fh:
            head = fh.read(4096)
    except Exception:
        return False
    return head.lstrip()[:1] == "["


def read_jsonl(
    path: str,
    *,
    compressed: bool = False,
    limit: int | None = None,
    span: ReadSpan = WHOLE_FILE,
) -> Iterator[RawRecord]:
    if not span.is_whole_file:
        yield from _read_jsonl_span(path, span)
        return
    if is_json_array_file(path, compressed):
        # The incremental span scanner already handles this shape and does not
        # hold the file in memory, which a whole-document json.loads would.
        yield from _read_embedded_records(path, compressed=compressed, limit=limit)
        return
    with open_maybe_compressed(path, compressed) as fh:
        for idx, line in enumerate(fh):
            if limit is not None and idx >= limit:
                return
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield RawRecord(json_loads(stripped), path, idx, raw_text=stripped)
            except Exception as exc:
                yield RawRecord(None, path, idx, error=f"{type(exc).__name__}: {exc}",
                                raw_text=stripped[:2000])


def _read_jsonl_span(path: str, span: ReadSpan) -> Iterator[RawRecord]:
    """Read one byte range, numbering records from ``span.first_index``.

    Decoding is done here rather than by a TextIOWrapper so the byte position
    stays exact: a wrapper buffers ahead and ``tell`` on it is not a cheap
    integer.
    """
    idx = span.first_index
    end = span.end
    with open(path, "rb") as fh:
        fh.seek(span.start)
        pos = span.start
        for raw in fh:
            pos += len(raw)
            stripped = raw.decode("utf-8", "replace").strip()
            if stripped:
                try:
                    yield RawRecord(json_loads(stripped), path, idx, raw_text=stripped)
                except Exception as exc:
                    yield RawRecord(None, path, idx, error=f"{type(exc).__name__}: {exc}",
                                    raw_text=stripped[:2000])
            idx += 1
            if 0 <= end <= pos:
                return


#: Blocks used when locating split points. Large enough that the scan is bound
#: by the disk rather than by Python.
_SPLIT_BLOCK = 1 << 23


def plan_line_splits(path: str, parts: int) -> list[ReadSpan]:
    """Cut a line-delimited file into ``parts`` spans aligned to line starts.

    Costs one sequential pass at ``memchr`` speed, which buys the record index
    at each cut. Without that index a shard could not number its records the way
    a serial read does, and every duplicate key, evidence location and
    fingerprint would depend on how many cores the machine had.
    """
    size = Path(path).stat().st_size
    if parts <= 1 or size <= 0:
        return [WHOLE_FILE]
    targets = [size * k // parts for k in range(1, parts)]
    cuts: list[tuple[int, int]] = []
    with open(path, "rb") as fh:
        pos = 0
        lines = 0
        t = 0
        while t < len(targets):
            block = fh.read(_SPLIT_BLOCK)
            if not block:
                break
            block_end = pos + len(block)
            while t < len(targets) and targets[t] < block_end:
                nl = block.find(b"\n", max(0, targets[t] - pos))
                if nl < 0:
                    break
                cut = pos + nl + 1
                if not cuts or cut > cuts[-1][0]:
                    cuts.append((cut, lines + block.count(b"\n", 0, nl + 1)))
                t += 1
            lines += block.count(b"\n")
            pos = block_end

    spans: list[ReadSpan] = []
    prev_offset, prev_index = 0, 0
    for cut, index in cuts:
        spans.append(ReadSpan(prev_offset, cut, prev_index))
        prev_offset, prev_index = cut, index
    spans.append(ReadSpan(prev_offset, -1, prev_index))
    return spans


def plan_row_group_splits(path: str, parts: int) -> list[ReadSpan]:
    """Cut a Parquet file into spans of whole row groups.

    The row counts come from the footer, so this reads kilobytes rather than the
    file.
    """
    if parts <= 1 or not HAVE_PYARROW:
        return [WHOLE_FILE]
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(path).metadata
        groups = meta.num_row_groups
        if groups < 2:
            return [WHOLE_FILE]
        rows = [meta.row_group(i).num_rows for i in range(groups)]
    except Exception:
        return [WHOLE_FILE]

    per = max(1, groups // parts)
    spans: list[ReadSpan] = []
    index = 0
    start = 0
    while start < groups:
        stop = min(groups, start + per) if len(spans) < parts - 1 else groups
        spans.append(ReadSpan(start, stop, index))
        index += sum(rows[start:stop])
        start = stop
    return spans


def read_json(
    path: str, *, compressed: bool = False, limit: int | None = None
) -> Iterator[RawRecord]:
    """A whole-file JSON document.

    Handles both a top-level array of records and a single object. Falls back to
    JSONL if the file turns out to be line-delimited despite its extension,
    which is common enough to be worth handling silently.
    """
    with open_maybe_compressed(path, compressed) as fh:
        text = fh.read()
    stripped = text.strip()
    if not stripped:
        return
    try:
        doc = json_loads(stripped)
    except Exception:
        yield from read_jsonl(path, compressed=compressed, limit=limit)
        return

    if isinstance(doc, list):
        for idx, item in enumerate(doc):
            if limit is not None and idx >= limit:
                return
            yield RawRecord(item, path, idx)
    elif isinstance(doc, dict):
        # A dict of splits, as `dataset_dict.json`-style exports produce.
        listy = {k: v for k, v in doc.items() if isinstance(v, list)}
        if listy:
            idx = 0
            for split, items in listy.items():
                for item in items:
                    if limit is not None and idx >= limit:
                        return
                    record = {**item, "_split": split} if isinstance(item, dict) else item
                    yield RawRecord(record, path, idx)
                    idx += 1
        else:
            yield RawRecord(doc, path, 0)


#: Most of one plain-text file that is read as a single record.
#:
#: A ``.txt`` in a training corpus is a document. A ``.txt`` in a home directory
#: is a log, a database export or a core dump, and ``fh.read()`` on one of those
#: is how a scan of a folder someone pointed at by accident becomes an
#: out-of-memory kill. Sixteen megabytes is two orders of magnitude above any
#: document a person wrote and two orders below the files that cause the
#: problem. The truncation is recorded on the record so the report can say the
#: file was longer than this rather than silently shortening a corpus.
MAX_TEXT_RECORD_BYTES = 16 << 20


def read_text(path: str, *, compressed: bool = False, limit: int | None = None,
              paragraph_split: bool = False) -> Iterator[RawRecord]:
    """Plain text, unless it turns out not to be.

    A ``.txt`` file holding JSON records is read as those records. Taking the
    extension at its word instead is the quietest way this tool can be wrong:
    every record collapses into one corpus document, the profile is inferred
    from that, and the conversational checks are not skipped so much as never
    considered. See :mod:`dropoutt.sniff` for how the decision is made.
    """
    from .sniff import sniff_file

    framing = sniff_file(path, compressed=compressed)
    if framing.is_records:
        yield from _read_embedded_records(path, compressed=compressed, limit=limit)
        return

    with open_maybe_compressed(path, compressed) as fh:
        content = fh.read(MAX_TEXT_RECORD_BYTES + 1)
        truncated = len(content) > MAX_TEXT_RECORD_BYTES
    if truncated:
        content = content[:MAX_TEXT_RECORD_BYTES]
    if not content.strip():
        return
    note = (
        f"{path}: larger than {MAX_TEXT_RECORD_BYTES >> 20} MB, read up to that point"
        if truncated else None
    )
    if not paragraph_split:
        yield RawRecord({"text": content}, path, 0, raw_text=content[:2000], truncated=note)
        return
    for idx, block in enumerate(b for b in content.split("\n\n") if b.strip()):
        if limit is not None and idx >= limit:
            return
        yield RawRecord({"text": block}, path, idx, raw_text=block[:2000])


#: Read size for the incremental span scanner. The buffer holds at most this
#: plus one record, so a mislabelled multi-gigabyte dump costs bounded memory.
_SPAN_CHUNK = 1 << 20


def _read_embedded_records(
    path: str, *, compressed: bool = False, limit: int | None = None
) -> Iterator[RawRecord]:
    """Yield the JSON records embedded in a text file.

    Scans incrementally so the whole file is never held at once. A span that
    fails to parse is yielded as an error record rather than dropped: those are
    the malformed records, and reporting them is the point.
    """
    from .sniff import scan_json_spans

    idx = 0
    buffer = ""
    with open_maybe_compressed(path, compressed) as fh:
        while True:
            chunk = fh.read(_SPAN_CHUNK)
            if not chunk:
                break
            buffer += chunk
            spans, resume = scan_json_spans(buffer)
            for start, end in spans:
                if limit is not None and idx >= limit:
                    return
                raw = buffer[start:end]
                # A malformed record is data, so it becomes an error record.
                # Anything that is not a decode failure is a bug in this file
                # and must not be disguised as one.
                try:
                    yield RawRecord(json_loads(raw), path, idx, raw_text=raw[:2000])
                except (ValueError, TypeError) as exc:
                    yield RawRecord(None, path, idx, error=f"{type(exc).__name__}: {exc}",
                                    raw_text=raw[:2000])
                idx += 1
            # Drop everything the scanner is finished with, including when it
            # found nothing: a long records-free stretch would otherwise be
            # re-scanned from offset zero on every read.
            buffer = buffer[max(resume, spans[-1][1] if spans else 0):]

    for start, end in scan_json_spans(buffer)[0]:
        if limit is not None and idx >= limit:
            return
        raw = buffer[start:end]
        try:
            yield RawRecord(json_loads(raw), path, idx, raw_text=raw[:2000])
        except (ValueError, TypeError) as exc:
            yield RawRecord(None, path, idx, error=f"{type(exc).__name__}: {exc}",
                            raw_text=raw[:2000])
        idx += 1


#: Delimiters worth considering when the extension does not settle it. A
#: semicolon-separated export read as comma-separated parses as a single column
#: holding the whole row, which then reads as unstructured raw text.
_CSV_DELIMITERS = (",", ";", "\t", "|")


def _csv_delimiter(path: str, compressed: bool, suffix: str) -> str:
    """Delimiter from the header row, falling back to the extension's default."""
    if suffix == ".tsv":
        return "\t"
    try:
        with open_maybe_compressed(path, compressed) as fh:
            head = fh.read(8192)
    except Exception:
        return ","
    line = next((ln for ln in head.splitlines() if ln.strip()), "")
    counts = {d: line.count(d) for d in _CSV_DELIMITERS}
    best = max(counts, key=lambda d: counts[d]) if line else ","
    return best if counts.get(best, 0) > 0 else ","


def read_csv(
    path: str, *, compressed: bool = False, limit: int | None = None
) -> Iterator[RawRecord]:
    effective = Path(path)
    if effective.suffix.lower() in {".gz", ".bz2", ".xz", ".zst"}:
        effective = effective.with_suffix("")
    delimiter = _csv_delimiter(path, compressed, effective.suffix.lower())
    with open_maybe_compressed(path, compressed) as fh:
        try:
            from .schema_induction import canonical_tabular_keys

            reader = csv.DictReader(fh, delimiter=delimiter)
            mapping: dict[str, str] | None = None
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    return
                if mapping is None:
                    mapping = canonical_tabular_keys(list(row.keys()))
                yield RawRecord(
                    {mapping.get(k, k): v for k, v in row.items()}, path, idx
                )
        except Exception as exc:
            yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")


def read_parquet(
    path: str, *, limit: int | None = None, span: ReadSpan = WHOLE_FILE
) -> Iterator[RawRecord]:
    if not HAVE_PYARROW:
        yield RawRecord(
            None, path, 0,
            error="parquet support not installed; reinstall dropoutt",
        )
        return
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")
        return

    if span.is_whole_file:
        batches = pf.iter_batches(batch_size=4096)
        idx = 0
    else:
        stop = span.end if span.end >= 0 else pf.num_row_groups
        batches = pf.iter_batches(
            batch_size=4096, row_groups=list(range(span.start, stop))
        )
        idx = span.first_index

    for batch in batches:
        rows = batch.to_pylist()
        for row in rows:
            if limit is not None and idx >= limit:
                return
            yield RawRecord(row, path, idx)
            idx += 1


def read_arrow(path: str, *, limit: int | None = None) -> Iterator[RawRecord]:
    """Arrow IPC file/stream or Feather table."""
    if not HAVE_PYARROW:
        yield RawRecord(
            None, path, 0,
            error="Arrow/Feather support not installed; reinstall dropoutt",
        )
        return

    try:
        import pyarrow as pa
        from pyarrow import ipc

        source = pa.memory_map(path, "r")
        try:
            reader = ipc.open_file(source)
            batches = (
                reader.get_batch(i) for i in range(reader.num_record_batches)
            )
        except Exception:
            source.seek(0)
            try:
                batches = ipc.open_stream(source)
            except Exception:
                # Feather V1 predates the Arrow IPC file layout.
                from pyarrow import feather

                batches = feather.read_table(path).to_batches(max_chunksize=4096)

        idx = 0
        try:
            for batch in batches:
                for row in batch.to_pylist():
                    if limit is not None and idx >= limit:
                        return
                    yield RawRecord(row, path, idx)
                    idx += 1
        finally:
            source.close()
    except Exception as exc:
        yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")


def read_orc(path: str, *, limit: int | None = None) -> Iterator[RawRecord]:
    """Apache ORC, using the same optional pyarrow dependency as Parquet."""
    if not HAVE_PYARROW:
        yield RawRecord(
            None, path, 0,
            error="ORC support not installed; reinstall dropoutt",
        )
        return
    from pyarrow import orc

    try:
        source = orc.ORCFile(path)
        idx = 0
        for stripe in range(source.nstripes):
            for row in source.read_stripe(stripe).to_pylist():
                if limit is not None and idx >= limit:
                    return
                yield RawRecord(row, path, idx)
                idx += 1
    except Exception as exc:
        yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")


def read_tar(path: str, *, limit: int | None = None) -> Iterator[RawRecord]:
    """WebDataset shards: consecutive members sharing a basename are one sample."""
    from .containers import read_tar as _read

    for idx, (payload, error, raw) in enumerate(_read(path, limit=limit)):
        yield RawRecord(payload, path, idx, error=error, raw_text=raw)


def read_mds(path: str, *, limit: int | None = None) -> Iterator[RawRecord]:
    """MosaicML Streaming shards, decoded against the sibling ``index.json``."""
    from .containers import read_mds as _read

    for idx, (payload, error, raw) in enumerate(_read(path, limit=limit)):
        yield RawRecord(payload, path, idx, error=error, raw_text=raw)


#: Formats whose records can be located without reading everything before them.
#:
#: `.mds` looks like it belongs here — it carries a byte offset per sample in
#: its header, which is exactly what a splitter needs. It is left out because
#: the win is small and the risk is not: a shard is one file among hundreds in a
#: Streaming split, so the corpus already divides at file granularity, and a
#: wrong offset in a binary format yields plausible garbage rather than a parse
#: error. Revisit when someone scans a single multi-gigabyte shard.
SPLITTABLE = (".jsonl", ".ndjson", ".parquet")


def plan_splits(path: str, suffix: str, parts: int, *, compressed: bool = False) -> list[ReadSpan]:
    """Spans for one file, or a single whole-file span if it cannot be split.

    Compressed streams are never split: finding a record boundary in one means
    decompressing everything before it, which is the work the split was supposed
    to divide.
    """
    if parts <= 1 or compressed:
        return [WHOLE_FILE]
    if suffix in (".jsonl", ".ndjson"):
        if is_json_array_file(path, compressed):
            # Records span many lines, so a line offset is not a record
            # boundary. Splitting here would hand each shard a fragment.
            return [WHOLE_FILE]
        return plan_line_splits(path, parts)
    if suffix == ".parquet":
        return plan_row_group_splits(path, parts)
    return [WHOLE_FILE]


def read_file(path: str, suffix: str, *, compressed: bool = False,
              limit: int | None = None, span: ReadSpan = WHOLE_FILE) -> Iterator[RawRecord]:
    """Dispatch on suffix."""
    if not span.is_whole_file:
        try:
            if suffix in (".jsonl", ".ndjson"):
                yield from read_jsonl(path, compressed=compressed, limit=limit, span=span)
            elif suffix == ".parquet":
                yield from read_parquet(path, limit=limit, span=span)
            else:  # pragma: no cover - the planner never produces these
                yield from read_file(path, suffix, compressed=compressed, limit=limit)
        except Exception as exc:
            yield RawRecord(None, path, span.first_index, error=f"{type(exc).__name__}: {exc}")
        return
    try:
        if compressed and suffix in {".parquet", ".arrow", ".feather", ".orc", ".mds"}:
            yield RawRecord(
                None,
                path,
                0,
                error=(
                    f"external compression is not supported for {suffix}; "
                    "decompress the file first"
                ),
            )
            return
        if suffix in (".jsonl", ".ndjson"):
            yield from read_jsonl(path, compressed=compressed, limit=limit)
        elif suffix == ".json":
            yield from read_json(path, compressed=compressed, limit=limit)
        elif suffix == ".parquet":
            yield from read_parquet(path, limit=limit)
        elif suffix in (".arrow", ".feather"):
            yield from read_arrow(path, limit=limit)
        elif suffix == ".orc":
            yield from read_orc(path, limit=limit)
        elif suffix in (".csv", ".tsv"):
            yield from read_csv(path, compressed=compressed, limit=limit)
        elif suffix in (".txt", ".md"):
            yield from read_text(path, compressed=compressed, limit=limit)
        elif suffix == ".tar":
            # `compressed` is not forwarded: tarfile's stream mode detects and
            # decodes gzip, bzip2 and xz itself, and a `.tar.gz` reaches here
            # with its outer suffix already stripped by the classifier.
            yield from read_tar(path, limit=limit)
        elif suffix == ".mds":
            yield from read_mds(path, limit=limit)
    except Exception as exc:
        yield RawRecord(
            None,
            path,
            0,
            error=f"{type(exc).__name__}: {exc}",
        )
