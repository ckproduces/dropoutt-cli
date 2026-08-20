"""Container formats: one file holding many records in a layout of its own.

Both of these are what a large corpus actually ships as, and neither is
self-describing in the way JSONL and Parquet are. A WebDataset shard is a
convention about filenames inside a tar; a MosaicML shard is a binary blob whose
schema lives in a different file. Getting either subtly wrong produces records
rather than errors, so the tests below assert on the decoded values.
"""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest

from dropoutt.discovery import discover
from dropoutt.readers import read_file


def _webdataset(path, samples=4, *, with_image=True):
    with tarfile.open(path, "w") as archive:
        for i in range(samples):
            payload = json.dumps({
                "messages": [
                    {"role": "user", "content": f"question number {i}"},
                    {"role": "assistant", "content": f"answer number {i}"},
                ]
            }).encode()
            info = tarfile.TarInfo(f"{i:06d}.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            caption = f"caption for sample {i}".encode()
            info = tarfile.TarInfo(f"{i:06d}.txt")
            info.size = len(caption)
            archive.addfile(info, io.BytesIO(caption))
            if with_image:
                blob = b"\xff\xd8not-really-a-jpeg"
                info = tarfile.TarInfo(f"{i:06d}.jpg")
                info.size = len(blob)
                archive.addfile(info, io.BytesIO(blob))
    return path


def _mds(folder, samples=5):
    folder.mkdir(parents=True, exist_ok=True)
    encoded = []
    for i in range(samples):
        text = f"sample number {i} with enough words to be a record".encode()
        head = np.array([len(text)], np.uint32).tobytes()
        encoded.append(head + text + np.int32(i * 3).tobytes())
    count = np.uint32(len(encoded))
    offsets = np.array([0, *map(len, encoded)]).cumsum().astype(np.uint32)
    offsets += len(count.tobytes()) + len(offsets.tobytes())
    shard = folder / "shard.00000.mds"
    shard.write_bytes(b"".join([count.tobytes(), offsets.tobytes(), b"".join(encoded)]))
    (folder / "index.json").write_text(json.dumps({
        "version": 2,
        "shards": [{
            "column_names": ["text", "label"],
            "column_encodings": ["str", "int32"],
            "column_sizes": [None, 4],
            "raw_data": {"basename": "shard.00000.mds"},
            "samples": samples,
        }],
    }))
    return shard


def test_webdataset_members_sharing_a_basename_become_one_record(tmp_path):
    path = _webdataset(tmp_path / "shard-000000.tar")
    rows = list(read_file(str(path), ".tar"))

    assert len(rows) == 4
    assert all(row.error is None for row in rows)
    first = rows[0].payload
    # The .json member *is* the record: its keys are lifted, not nested under
    # "json", so the layout matcher sees what it would see in JSONL.
    assert first["messages"][0]["content"] == "question number 0"
    assert first["txt"] == "caption for sample 0"
    # A modality this tool cannot read is recorded as present and left undecoded.
    assert first["jpg"] == ""
    assert [row.source_index for row in rows] == [0, 1, 2, 3]


def test_webdataset_respects_a_record_limit(tmp_path):
    path = _webdataset(tmp_path / "shard.tar", samples=10)
    assert len(list(read_file(str(path), ".tar", limit=3))) == 3


def test_a_corrupt_tar_becomes_a_record_carrying_the_error(tmp_path):
    path = tmp_path / "broken.tar"
    path.write_bytes(b"this is not a tar archive at all, not even close" * 40)
    rows = list(read_file(str(path), ".tar"))
    assert len(rows) == 1
    assert rows[0].error


def test_mds_shard_decodes_against_its_sibling_index(tmp_path):
    shard = _mds(tmp_path / "train")
    rows = list(read_file(str(shard), ".mds"))

    assert len(rows) == 5
    assert all(row.error is None for row in rows)
    assert rows[2].payload == {
        "text": "sample number 2 with enough words to be a record", "label": 6
    }


def test_an_mds_shard_without_its_index_says_so_rather_than_guessing(tmp_path):
    shard = _mds(tmp_path / "train")
    (tmp_path / "train" / "index.json").unlink()
    rows = list(read_file(str(shard), ".mds"))
    assert len(rows) == 1
    assert "index.json" in (rows[0].error or "")


def test_both_container_formats_are_discovered(tmp_path):
    (tmp_path / "web").mkdir()
    _webdataset(tmp_path / "web" / "shard.tar")
    _mds(tmp_path / "mosaic")

    found = discover(str(tmp_path))
    suffixes = {ref.suffix for ref in found.files}
    assert ".tar" in suffixes
    assert ".mds" in suffixes
    # The MosaicML index is that shard's schema, not a record of its own.
    assert not any(ref.rel.endswith("index.json") for ref in found.files)


def test_an_index_json_away_from_any_shard_is_still_data(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps([{"text": "a real record"}]))
    found = discover(str(tmp_path))
    assert [ref.rel for ref in found.files] == ["index.json"]


@pytest.mark.parametrize("name,suffix,compressed", [
    ("train.jsonl", ".jsonl", False),
    ("train.JSONL", ".jsonl", False),
    ("train.jsonl.zst", ".jsonl", True),
    ("shard.tar", ".tar", False),
    ("shard.tar.gz", ".tar", True),
    ("notes", "", False),
    ("archive.gz", "", True),
])
def test_effective_suffix_is_decided_in_one_place(name, suffix, compressed):
    """Discovery, the shard planner and the reader must agree on what a file is.

    They each worked it out separately and disagreed on case: discovery matched
    `.JSONL` against a lowercase set and dropped the file, while the reader
    would have read it.
    """
    from dropoutt.discovery import effective_suffix

    assert effective_suffix(name) == (suffix, compressed)
