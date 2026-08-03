"""The sharded scan has to give the same answer as the serial one.

Two guarantees are tested here, and they are the two that would fail silently.

Every check has to say how it merges. A check that grows a new counter and does
not name it keeps counting inside each worker and then loses it, so a parallel
scan reports a smaller number than a serial one with nothing to indicate that
anything went wrong. ``test_every_check_declares_how_it_merges`` compares the
declarations against the attributes an instance actually has.

And the whole pipeline has to agree end to end, including the parts that are
order-sensitive: which examples are kept, which record wins a contamination
witness, how duplicates are attributed, and the fingerprint id.
"""

from __future__ import annotations

import json

import pytest

from dropoutt.checks.base import REGISTRY
from dropoutt.parallel import file_salt, sample_key
from dropoutt.runner import scan

MERGE_FIELDS = (
    "MERGE_SUM",
    "MERGE_COUNTS",
    "MERGE_NESTED",
    "MERGE_EVIDENCE",
    "MERGE_CONCAT",
    "MERGE_FIRST",
    "MERGE_CUSTOM",
    "MERGE_IGNORE",
)


def test_every_check_declares_how_it_merges():
    missing: list[str] = []
    for cls in REGISTRY.all():
        instance = cls()
        declared = {name for field in MERGE_FIELDS for name in getattr(cls, field)}
        for attribute in vars(instance):
            if attribute in declared:
                continue
            missing.append(f"{cls.check_id}.{attribute}")
    assert not missing, (
        "these check attributes are neither merged nor explicitly ignored, so a "
        "parallel scan would drop them: " + ", ".join(sorted(missing))
    )


def test_declared_merge_fields_exist():
    for cls in REGISTRY.all():
        instance = cls()
        for field in MERGE_FIELDS:
            for name in getattr(cls, field):
                assert hasattr(instance, name), f"{cls.check_id} declares missing {name}"


def test_sample_key_is_uniform_and_positional():
    salt = file_salt("/some/path/train.jsonl")
    keys = [sample_key(salt, i) for i in range(4000)]
    assert len(set(keys)) == len(keys)
    # A bottom-k sample of a uniform key is a uniform sample of the records.
    smallest = sorted(range(4000), key=lambda i: keys[i])[:400]
    assert 120 < sum(1 for i in smallest if i < 2000) < 280
    # Position, not content: two identical records get different keys.
    assert sample_key(salt, 7) != sample_key(salt, 8)


def _corpus(tmp_path, records=900):
    root = tmp_path / "data"
    root.mkdir()
    for shard in range(3):
        lines = []
        for i in range(records):
            n = shard * records + i
            body = (
                "The mechanism is easiest to follow one component at a time, and the "
                f"interesting case is number {n % 37}. " * (1 + n % 4)
            )
            if n % 11 == 0:
                body = "Duplicated boilerplate paragraph that appears many times over. " * 4
            lines.append(json.dumps({"messages": [
                {"role": "user", "content": f"Explain topic {n % 53} in detail."},
                {"role": "assistant", "content": body},
            ]}))
        (root / f"train-{shard}.jsonl").write_text("\n".join(lines) + "\n")
    return root


def _summary(result):
    return {
        f.check_id: (f.count, f.total_considered, f.detail,
                     [(e.source_file.rsplit("/", 1)[-1], e.source_index, e.excerpt)
                      for e in f.evidence])
        for f in result.findings
    }


@pytest.mark.parametrize("workers", [2, 5])
def test_sharded_scan_matches_serial(tmp_path, workers, monkeypatch):
    from dropoutt import parallel

    root = _corpus(tmp_path)
    # The corpus is far below the byte threshold that would trigger a pool in
    # normal use, so lower it rather than writing a hundred megabytes of test
    # data to prove the same thing.
    monkeypatch.setattr(parallel, "MIN_BYTES_FOR_PARALLEL", 1)

    serial = scan(str(root), workers=1)
    parallel_result = scan(str(root), workers=workers)

    assert parallel_result.records_scanned == serial.records_scanned
    assert parallel_result.ctx.stats["shards"] > 1
    assert parallel_result.ctx.stats["content_hash"] == serial.ctx.stats["content_hash"]
    assert parallel_result.ctx.stats["total_chars"] == serial.ctx.stats["total_chars"]
    assert _summary(parallel_result) == _summary(serial)


def test_split_files_preserve_record_numbering(tmp_path, monkeypatch):
    """A byte-range shard has to number its records the way a whole-file read does."""
    from dropoutt import parallel
    from dropoutt.readers import plan_line_splits, read_file

    root = _corpus(tmp_path, records=300)
    path = str(root / "train-0.jsonl")
    whole = [(r.source_index, r.payload) for r in read_file(path, ".jsonl")]
    spans = plan_line_splits(path, 4)
    assert len(spans) > 1
    pieces = [row for span in spans
              for row in ((r.source_index, r.payload) for r in read_file(path, ".jsonl", span=span))]
    assert pieces == whole
    assert parallel.MIN_BYTES_FOR_PARALLEL > 0
