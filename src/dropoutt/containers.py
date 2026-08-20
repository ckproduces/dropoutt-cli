"""Container formats: one file holding many records, in a layout of its own.

Everything in :mod:`dropoutt.readers` is either line-delimited or columnar, and
both are self-describing. The two formats here are neither. They are what large
corpora actually ship as once they outgrow a folder of JSONL:

``.mds``
    MosaicML Streaming shards. A flat binary file of length-prefixed samples,
    with the column names and their encodings kept *outside* the shard in a
    sibling ``index.json``. Reading one without that file is guesswork, so a
    shard whose index is missing is reported as unreadable rather than decoded
    hopefully.

``.tar``
    WebDataset shards. A plain tar where consecutive members sharing a basename
    are one sample: ``000017.json``, ``000017.txt``, ``000017.jpg``. The
    grouping is a convention rather than a header, and it is the convention the
    whole ecosystem uses.

Both are read as streams. A WebDataset shard is commonly several gigabytes and a
tar index is built by walking the archive, so :func:`read_tar` opens in
``r|*`` mode — sequential, no seeking, no member table held in memory — and
every member is size-capped before it is read into a string. That is the
difference between scanning a 40 GB shard set and being killed by the OOM
reaper.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .compat import json_loads

#: Largest single archive member read into memory. A WebDataset sample's text
#: side is kilobytes; anything past this is an image, an audio file or a nested
#: archive, and none of those are training text this tool can read.
MAX_MEMBER_BYTES = 8 << 20

#: Members whose payload is text worth scanning. Everything else in a
#: WebDataset shard is a modality this tool does not read, and is counted rather
#: than decoded.
TEXT_MEMBER_SUFFIXES = {
    ".txt", ".text", ".json", ".jsonl", ".ndjson", ".cls", ".caption",
    ".md", ".csv", ".tsv", ".transcript", ".label", ".tags",
}


def _record_from_members(members: dict[str, str]) -> Any:
    """Turn one WebDataset sample's members into a record.

    A ``.json`` member *is* the record when it holds an object — that is how
    text-only WebDataset shards are written, and unwrapping it means the layout
    matcher sees the same keys it would in JSONL. Otherwise the extensions
    become the field names, which is what a multi-modal sample looks like.
    """
    for key in ("json", "jsonl", "ndjson"):
        if key in members:
            try:
                doc = json_loads(members[key])
            except (ValueError, TypeError):
                continue
            if isinstance(doc, dict):
                return {**{k: v for k, v in members.items() if k != key}, **doc}
    return dict(members)


def read_tar(
    path: str, *, limit: int | None = None, compressed: bool = False
) -> Iterator[Any]:
    """Yield one record per WebDataset sample, as ``(payload, error, raw)``.

    Yields tuples rather than ``RawRecord`` so this module does not import
    :mod:`dropoutt.readers`, which imports this one.
    """
    # "r|*" is the streaming reader: it decodes gzip, bzip2 and xz transparently
    # and never builds the member index that "r:*" does. `compressed` is ignored
    # on purpose — a `.tar.gz` is handled by the archive layer, not by the outer
    # decompression wrapper, and opening the wrapper first would hand tarfile a
    # stream it cannot seek within either way.
    del compressed
    index = 0
    current_key: str | None = None
    members: dict[str, str] = {}
    try:
        with tarfile.open(path, "r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                key, _, suffix = _split_member(member.name)
                if key != current_key and current_key is not None:
                    if members:
                        if limit is not None and index >= limit:
                            return
                        yield _record_from_members(members), None, ""
                        index += 1
                    members = {}
                current_key = key
                if suffix not in TEXT_MEMBER_SUFFIXES:
                    # Counted, not decoded: knowing a sample carried an image is
                    # part of describing the corpus, and decoding one is not.
                    members.setdefault(suffix.lstrip("."), "")
                    continue
                if member.size > MAX_MEMBER_BYTES:
                    members[suffix.lstrip(".")] = ""
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                members[suffix.lstrip(".")] = handle.read().decode("utf-8", "replace")
            if members and (limit is None or index < limit):
                yield _record_from_members(members), None, ""
    except (tarfile.TarError, OSError, EOFError) as exc:
        yield None, f"{type(exc).__name__}: {exc}", ""


def _split_member(name: str) -> tuple[str, str, str]:
    """``("shard/000017", "000017", ".json")`` from ``shard/000017.json``.

    WebDataset keys keep every dot but the last, so ``000017.left.jpg`` and
    ``000017.left.json`` are two members of the sample ``000017.left``.
    """
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return name, name, ""
    return stem, stem.rpartition("/")[2], "." + suffix.lower()


# --------------------------------------------------------------------------
# MosaicML Streaming (.mds)
# --------------------------------------------------------------------------

#: MDS encodings this reader decodes to a Python value. The rest of the
#: catalogue is image and tensor codecs, which are recorded as their byte length
#: rather than decoded — a scan of training *text* has nothing to say about a
#: JPEG, and pulling Pillow in to look at one would be a compiled dependency for
#: a field nothing reads.
_MDS_TEXT = {"str", "utf8"}
_MDS_JSON = {"json"}
_MDS_INT = {
    "int": 8, "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
}
_MDS_FLOAT = {"float16": 2, "float32": 4, "float64": 8, "float": 8}


def _mds_index(path: str) -> dict[str, Any] | None:
    """The shard's column metadata, from the ``index.json`` beside it.

    Streaming writes one index per split directory covering every shard in it,
    so the file is looked for next to the shard and then one level up — the
    layout ``train/index.json`` plus ``train/shard.00000.mds`` is the common
    one, and ``index.json`` at the dataset root with shards in subdirectories is
    the other.
    """
    here = Path(path).parent
    for candidate in (here / "index.json", here.parent / "index.json"):
        try:
            with open(candidate, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        shards = doc.get("shards")
        if not isinstance(shards, list) or not shards:
            continue
        name = os.path.basename(path)
        for shard in shards:
            raw = (shard or {}).get("raw_data") or {}
            if raw.get("basename") == name:
                return shard
        # Every shard in a split shares its schema, so any entry answers the
        # question this function was asked when the basename does not match —
        # which happens whenever shards were renamed or copied out of a split.
        return shards[0]
    return None


def read_mds(path: str, *, limit: int | None = None) -> Iterator[Any]:
    """Yield ``(payload, error, raw)`` per sample in one MosaicML shard.

    Layout, from ``MDSWriter.encode_joint_shard``: a ``uint32`` sample count,
    then ``count + 1`` ``uint32`` byte offsets into this same file, then the
    sample bodies back to back. Each body is the sizes of the variable-width
    columns as ``uint32`` each, then every column's bytes in declaration order.
    """
    import numpy as np

    shard = _mds_index(path)
    if shard is None:
        yield None, (
            "no index.json found beside this .mds shard; MosaicML Streaming "
            "keeps the column names and encodings there, and a shard cannot be "
            "decoded without them"
        ), ""
        return

    names = list(shard.get("column_names") or [])
    encodings = list(shard.get("column_encodings") or [])
    sizes = list(shard.get("column_sizes") or [])
    if not names or len(names) != len(encodings) or len(names) != len(sizes):
        yield None, "index.json does not describe this shard's columns", ""
        return

    try:
        with open(path, "rb") as fh:
            header = fh.read(4)
            if len(header) < 4:
                yield None, "shard is empty or truncated", ""
                return
            count = int(np.frombuffer(header, dtype=np.uint32, count=1)[0])
            if count <= 0:
                return
            offset_bytes = fh.read(4 * (count + 1))
            if len(offset_bytes) < 4 * (count + 1):
                yield None, "shard offset table is truncated", ""
                return
            offsets = np.frombuffer(offset_bytes, dtype=np.uint32).astype(np.int64)

            for index in range(count):
                if limit is not None and index >= limit:
                    return
                start, stop = int(offsets[index]), int(offsets[index + 1])
                if stop <= start:
                    continue
                fh.seek(start)
                body = fh.read(stop - start)
                if len(body) < stop - start:
                    yield None, f"sample {index} is truncated", ""
                    return
                try:
                    yield _decode_mds_sample(body, names, encodings, sizes), None, ""
                except (ValueError, IndexError, UnicodeError) as exc:
                    yield None, f"{type(exc).__name__}: {exc}", ""
    except OSError as exc:
        yield None, f"{type(exc).__name__}: {exc}", ""


def _decode_mds_sample(
    body: bytes, names: list[str], encodings: list[str], sizes: list[int | None]
) -> dict[str, Any]:
    import numpy as np

    variable = [i for i, size in enumerate(sizes) if size is None]
    head = 4 * len(variable)
    if len(body) < head:
        raise ValueError("sample header is shorter than its variable-column count")
    widths = (
        np.frombuffer(body[:head], dtype=np.uint32).astype(np.int64).tolist()
        if variable else []
    )
    resolved = list(sizes)
    for slot, width in zip(variable, widths, strict=True):
        resolved[slot] = int(width)

    record: dict[str, Any] = {}
    cursor = head
    for name, encoding, size in zip(names, encodings, resolved, strict=True):
        width = int(size or 0)
        chunk = body[cursor:cursor + width]
        cursor += width
        record[name] = _decode_mds_value(chunk, str(encoding))
    return record


def _decode_mds_value(chunk: bytes, encoding: str) -> Any:
    import numpy as np

    base = encoding.split(":", 1)[0].lower()
    if base in _MDS_TEXT:
        return chunk.decode("utf-8", "replace")
    if base in _MDS_JSON:
        try:
            return json_loads(chunk.decode("utf-8", "replace"))
        except (ValueError, TypeError):
            return chunk.decode("utf-8", "replace")
    if base in _MDS_INT:
        return int(np.frombuffer(chunk, dtype=_numpy_name(base), count=1)[0]) if chunk else 0
    if base in _MDS_FLOAT:
        return float(np.frombuffer(chunk, dtype=_numpy_name(base), count=1)[0]) if chunk else 0.0
    # An unreadable modality is described, not decoded. The layout matcher reads
    # this as a non-text field, which is exactly what it is.
    return f"<{base} {len(chunk)} bytes>"


def _numpy_name(base: str) -> str:
    if base == "int":
        return "int64"
    if base == "float":
        return "float64"
    return base
