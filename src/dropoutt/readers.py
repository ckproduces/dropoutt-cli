"""File readers.

Every reader yields ``RawRecord`` and never raises on bad input. A malformed
line becomes a record carrying its parse error, which downstream turns into a
finding. Erroring out on line 4,000,001 of a 28 GB scan would be the wrong
behaviour: the point of the tool is to report what is wrong with the data.
"""

from __future__ import annotations

import csv
import gzip
import io
from dataclasses import dataclass
from typing import Any, Iterator

from .compat import HAVE_PYARROW, json_loads


@dataclass(slots=True)
class RawRecord:
    payload: Any
    source_file: str
    source_index: int
    #: Set when the record could not be parsed. ``payload`` then holds the raw text.
    error: str | None = None
    #: Raw text as read, kept for text-profile reading and for error evidence.
    raw_text: str = ""


def open_maybe_compressed(path: str, compressed: bool) -> io.TextIOBase:
    if compressed and path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_jsonl(path: str, *, compressed: bool = False, limit: int | None = None) -> Iterator[RawRecord]:
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


def read_json(path: str, *, compressed: bool = False, limit: int | None = None) -> Iterator[RawRecord]:
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
                    if isinstance(item, dict):
                        item = {**item, "_split": split}
                    yield RawRecord(item, path, idx)
                    idx += 1
        else:
            yield RawRecord(doc, path, 0)


def read_text(path: str, *, compressed: bool = False, limit: int | None = None,
              paragraph_split: bool = False) -> Iterator[RawRecord]:
    """Plain text. One document per file, or one per blank-line-separated block."""
    with open_maybe_compressed(path, compressed) as fh:
        content = fh.read()
    if not content.strip():
        return
    if not paragraph_split:
        yield RawRecord({"text": content}, path, 0, raw_text=content[:2000])
        return
    for idx, block in enumerate(b for b in content.split("\n\n") if b.strip()):
        if limit is not None and idx >= limit:
            return
        yield RawRecord({"text": block}, path, idx, raw_text=block[:2000])


def read_csv(path: str, *, compressed: bool = False, limit: int | None = None) -> Iterator[RawRecord]:
    delimiter = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
    with open_maybe_compressed(path, compressed) as fh:
        try:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for idx, row in enumerate(reader):
                if limit is not None and idx >= limit:
                    return
                yield RawRecord(dict(row), path, idx)
        except Exception as exc:
            yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")


def read_parquet(path: str, *, limit: int | None = None) -> Iterator[RawRecord]:
    if not HAVE_PYARROW:
        yield RawRecord(
            None, path, 0,
            error="parquet support not installed; pip install 'dropoutt[parquet]'",
        )
        return
    import pyarrow.parquet as pq  # noqa: PLC0415

    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        yield RawRecord(None, path, 0, error=f"{type(exc).__name__}: {exc}")
        return

    idx = 0
    for batch in pf.iter_batches(batch_size=4096):
        rows = batch.to_pylist()
        for row in rows:
            if limit is not None and idx >= limit:
                return
            yield RawRecord(row, path, idx)
            idx += 1


def read_file(path: str, suffix: str, *, compressed: bool = False,
              limit: int | None = None) -> Iterator[RawRecord]:
    """Dispatch on suffix."""
    if suffix in (".jsonl", ".ndjson"):
        yield from read_jsonl(path, compressed=compressed, limit=limit)
    elif suffix == ".json":
        yield from read_json(path, compressed=compressed, limit=limit)
    elif suffix == ".parquet":
        yield from read_parquet(path, limit=limit)
    elif suffix in (".csv", ".tsv"):
        yield from read_csv(path, compressed=compressed, limit=limit)
    elif suffix in (".txt", ".md"):
        yield from read_text(path, compressed=compressed, limit=limit)
    else:
        return
