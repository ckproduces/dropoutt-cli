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
        from pyarrow import ipc

        with pa.OSFile(str(path), "wb") as sink, ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    else:
        from pyarrow import orc

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


def test_jsonl_holding_a_pretty_printed_json_array_is_read_as_records(tmp_path):
    """The extension is a claim about the file, not a fact about it.

    A real corpus shipped 1.26M records this way. Read line by line, every line
    is a parse error, the repeated `{` and `"license": ...` fragments become a
    157,045-copy duplicate cluster, and they are too short to identify so the
    language breakdown fills with "unknown". One misread file, three wrong
    findings.
    """
    path = tmp_path / "sorucevap.jsonl"
    path.write_text(json.dumps(
        [{"question": f"Soru {i}", "answer": f"Cevap {i}", "license": "CC BY-SA 3.0"}
         for i in range(5)],
        ensure_ascii=False, indent=2,
    ), encoding="utf-8")

    rows = list(readers.read_file(str(path), ".jsonl"))

    assert len(rows) == 5
    assert all(r.error is None for r in rows)
    assert rows[0].payload["question"] == "Soru 0"
    assert rows[4].payload["answer"] == "Cevap 4"


def test_json_array_jsonl_is_never_line_split(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text("[\n  {\"text\": \"one\"},\n  {\"text\": \"two\"}\n]\n", encoding="utf-8")

    assert readers.plan_splits(str(path), ".jsonl", 4) == [readers.WHOLE_FILE]


def test_ordinary_jsonl_still_splits(tmp_path):
    path = tmp_path / "b.jsonl"
    path.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(400)),
                    encoding="utf-8")

    assert len(readers.plan_splits(str(path), ".jsonl", 4)) > 1


def test_csv_headers_are_mapped_to_layout_keys(tmp_path):
    """`Question,Activation-Feed,Result` matched no layout and fell back to raw."""
    path = tmp_path / "t_dataset.csv"
    path.write_text(
        "Question,Activation-Feed,Result\n"
        "What is Angular?,Kodlama,A frontend web framework.\n",
        encoding="utf-8",
    )

    rows = list(readers.read_file(str(path), ".csv"))

    assert rows[0].payload["question"] == "What is Angular?"
    assert rows[0].payload["answer"] == "A frontend web framework."
    # A column no layout knows about is kept, not dropped.
    assert rows[0].payload["activation-feed"] == "Kodlama"


def test_semicolon_delimited_csv_is_not_read_as_one_column(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("question;answer\nNedir bu?;Bir cevap.\n", encoding="utf-8")

    rows = list(readers.read_file(str(path), ".csv"))

    assert rows[0].payload == {"question": "Nedir bu?", "answer": "Bir cevap."}
