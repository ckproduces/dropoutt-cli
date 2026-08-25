"""The scan pass reads records in batches; the answers must not know that.

Every check still sees every record, in order, exactly once. The shared
per-record features are computed by column now, and the sampling keys are an
array rather than a loop, so the risk is not that something breaks loudly — it
is that a batch boundary quietly changes a count.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dropoutt.checks.base import Check
from dropoutt.context import ScanContext
from dropoutt.models import Profile
from dropoutt.parallel import RECORD_BATCH, _batched, sample_key, sample_keys


@pytest.fixture
def scan_context():
    return ScanContext(root="/corpus", profile=Profile.CORPUS)


def test_sample_keys_match_the_scalar_mix():
    """The array form wraps where the scalar form masks; they must agree."""
    positions = np.arange(0, 4096, dtype=np.int64)
    for salt in (0, 1, 0x9E3779B97F4A7C15, 2**63 + 12345, 2**64 - 1):
        vectorised = sample_keys(salt, positions)
        one_at_a_time = np.array(
            [sample_key(salt, int(i)) for i in positions], dtype=np.uint64
        )
        assert np.array_equal(vectorised, one_at_a_time), salt


def _raw(value, text: str = ""):
    from types import SimpleNamespace

    return SimpleNamespace(value=value, raw_text=text)


def test_batched_preserves_order_and_covers_everything():
    items = [_raw(i) for i in range(1000)]
    batches = list(_batched(iter(items), 256))

    assert [len(b) for b in batches] == [256, 256, 256, 232]
    assert [x for batch in batches for x in batch] == items
    assert list(_batched(iter([]), 256)) == []


def test_a_batch_closes_early_on_long_records():
    """256 megabyte-long records must not be pinned as one batch."""
    items = [_raw(i, "x" * 1_000_000) for i in range(10)]

    batches = list(_batched(iter(items), 256, char_budget=4_000_000))

    assert [len(b) for b in batches] == [4, 4, 2]
    assert [x for batch in batches for x in batch] == items


def test_bottom_k_selection_matches_the_scalar_path():
    """The heap key must be the full-magnitude uint64, not an int64 cast.

    Casting wraps every key above 2^63 negative — half the keyspace — and the
    bottom-k sample silently becomes a different set of records than 1.1
    selected on the same corpus.
    """
    salt = 0x9E3779B97F4A7C15
    positions = np.arange(0, 2000, dtype=np.int64)
    keys = sample_keys(salt, positions)

    # What run_shard pushes: the negated Python int of the uint64 key. The
    # heap keeps the entries whose pushed key is largest.
    heap_pick = set(sorted(range(2000), key=lambda i: -int(keys[i]), reverse=True)[:50])
    # What 1.1 kept: the records with the smallest scalar sample key.
    scalar_pick = set(sorted(range(2000), key=lambda i: sample_key(salt, i))[:50])
    # What the int64 cast kept instead: a different set for half the keyspace.
    wrapped = keys.astype(np.int64)
    wrapped_pick = set(
        sorted(range(2000), key=lambda i: -int(wrapped[i]), reverse=True)[:50]
    )

    assert heap_pick == scalar_pick
    assert wrapped_pick != scalar_pick, "the cast bug would not even be visible here"


class _Counter(Check):
    check_id = "TEST-BATCH-001"
    MERGE_SUM = ("seen",)

    def __init__(self) -> None:
        self.seen = 0
        self.order: list[int] = []

    def observe(self, doc, ctx) -> None:
        self.seen += 1
        self.order.append(doc.source_index)


class _Fussy(_Counter):
    check_id = "TEST-BATCH-002"

    def observe(self, doc, ctx) -> None:
        if doc.source_index == 3:
            raise ValueError("this one record")
        super().observe(doc, ctx)


def _docs(count: int):
    from dropoutt.models import Document

    return [
        Document(
            doc_id=f"d{i}", text=f"record {i}", turns=[],
            source_file="f.jsonl", source_index=i, dataset="d",
        )
        for i in range(count)
    ]


def test_observe_batch_defaults_to_the_loop_it_replaced(scan_context):
    check = _Counter()
    docs = _docs(10)

    check.observe_batch(docs, scan_context)

    assert check.seen == 10
    assert check.order == list(range(10))


def test_one_bad_record_costs_that_record_and_not_the_batch(scan_context):
    """A check that raises used to lose one record. It still loses one."""
    check = _Fussy()

    check.observe_batch(_docs(10), scan_context)

    assert check.seen == 9
    assert 3 not in check.order
    assert any("TEST-BATCH-002" in note for note in scan_context.degradations)


def test_a_scan_split_across_batch_boundaries_counts_the_same(tmp_path):
    """Two corpora identical but for their size relative to the batch."""
    from dropoutt import runner

    rows = [{"text": f"Record number {i} says something about topic {i % 7}."}
            for i in range(RECORD_BATCH * 2 + 5)]
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "a.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )

    result = runner.scan(str(folder), workers=1, offline=True)

    assert result.records_scanned == len(rows)
    assert result.ctx.stats["total_chars"] == sum(len(r["text"]) for r in rows)


@pytest.mark.parametrize("count", [1, RECORD_BATCH - 1, RECORD_BATCH, RECORD_BATCH + 1])
def test_the_signature_store_ignores_a_batch_it_has_already_taken(count):
    """Both dedup checks share one store and are handed the same batch."""
    from dropoutt.checks.tier1_dedup import _SignatureStore

    store = _SignatureStore()
    docs = _docs(count)
    for doc in docs:
        doc.text = f"{doc.text} " + "padding words to clear the length floor " * 3

    store.add_batch(docs)
    first = store.counter
    store.add_batch(docs)

    assert first == count
    assert store.counter == count, "the second offer of the same batch was indexed"


def test_a_store_error_mid_batch_costs_one_record_not_the_rest(scan_context):
    """The dedup override must keep the default loop's error contract."""
    from dropoutt.checks.tier1_dedup import _SignatureStore

    store = _SignatureStore()
    docs = _docs(10)
    for doc in docs:
        doc.text = f"{doc.text} " + "padding words to clear the length floor " * 3

    real_add = store._add
    def flaky_add(doc):
        if doc.source_index == 4:
            raise ValueError("this one record")
        real_add(doc)
    store._add = flaky_add

    errors: list[int] = []
    store.add_batch(docs, on_error=lambda doc, exc: errors.append(doc.source_index))

    assert store.counter == 9, "records after the error were dropped"
    assert errors == [4]
    # The sibling check's immediate re-offer of the same batch still no-ops.
    store.add_batch(docs, on_error=lambda doc, exc: errors.append(doc.source_index))
    assert store.counter == 9
