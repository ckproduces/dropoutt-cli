"""Compressed inputs must be decompressed or reported, never parsed as gibberish."""

from __future__ import annotations

import bz2
import json
import lzma

import pytest

from dropoutt import readers


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
)
def test_standard_library_compression_formats_are_read(tmp_path, suffix, compress):
    path = tmp_path / f"data.jsonl{suffix}"
    path.write_bytes(compress((json.dumps({"text": "hello"}) + "\n").encode()))

    rows = list(readers.read_file(str(path), ".jsonl", compressed=True))

    assert len(rows) == 1
    assert rows[0].payload == {"text": "hello"}
    assert rows[0].error is None


@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
)
def test_compressed_tsv_keeps_tab_delimiter(tmp_path, suffix, compress):
    path = tmp_path / f"data.tsv{suffix}"
    path.write_bytes(compress(b"text\tscore\nhello\t3\n"))

    rows = list(readers.read_file(str(path), ".tsv", compressed=True))

    assert rows[0].payload == {"text": "hello", "score": "3"}


def test_missing_zstd_support_is_an_actionable_record_error(monkeypatch, tmp_path):
    path = tmp_path / "data.jsonl.zst"
    path.write_bytes(b"not-used")
    monkeypatch.setattr(readers, "HAVE_ZSTANDARD", False)

    rows = list(readers.read_file(str(path), ".jsonl", compressed=True))

    assert len(rows) == 1
    assert "dropoutt[zstd]" in rows[0].error


@pytest.mark.skipif(not readers.HAVE_PYARROW, reason="pyarrow not installed")
@pytest.mark.parametrize("suffix", [".arrow", ".feather", ".orc"])
def test_columnar_formats_are_read(suffix, tmp_path):
    import pyarrow as pa

    table = pa.Table.from_pylist([
        {"text": "first", "score": 1},
        {"text": "second", "score": 2},
    ])
    path = tmp_path / f"data{suffix}"
    if suffix in (".arrow", ".feather"):
        import pyarrow.ipc as ipc

        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    else:
        import pyarrow.orc as orc

        orc.write_table(table, path)

    rows = list(readers.read_file(str(path), suffix))

    assert [row.payload["text"] for row in rows] == ["first", "second"]
    assert all(row.error is None for row in rows)


def test_columnar_formats_are_discovered(tmp_path):
    from dropoutt.discovery import discover

    for suffix in (".arrow", ".feather", ".orc"):
        (tmp_path / f"data{suffix}").write_bytes(b"placeholder")

    result = discover(str(tmp_path))

    assert {file.suffix for file in result.files} == {".arrow", ".feather", ".orc"}


def test_externally_compressed_columnar_file_has_actionable_error(tmp_path):
    path = tmp_path / "data.parquet.gz"
    path.write_bytes(b"not-used")

    rows = list(readers.read_file(
        str(path), ".parquet", compressed=True
    ))

    assert len(rows) == 1
    assert "decompress the file first" in rows[0].error
