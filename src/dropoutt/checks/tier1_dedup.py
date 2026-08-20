"""Tier 1 near-duplicate and cross-dataset overlap checks.

Both are two-phase. Signatures accumulate during the streaming pass; clusters
and the overlap matrix are resolved at the end.

The overlap matrix is **directional and must stay that way**. If a 10k dataset
sits entirely inside a 1M dataset, the containment is 100% one way and 1% the
other. A symmetric Jaccard reports something small and hides the fact worth
acting on.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..context import ScanContext, dedup_words_of
from ..minhash import DEFAULT_PRESET, PRESETS, LSHIndex, MinHasher, UnionFind
from ..models import (
    CostClass,
    Document,
    Evidence,
    Finding,
    Profile,
    Requirement,
    Severity,
)
from ..textutil import excerpt
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)


def _exact_key(text: str) -> int:
    """Whitespace- and case-insensitive identity of a record's text.

    Near-duplicate clusters are full of exact copies, and the exact-duplicate
    check has already counted those. Reporting them twice inflates the second
    number and fills its examples with pairs the reader was shown a section
    earlier — on one corpus the top six "near-copies" were the same four rows,
    each matched against itself at 100%.

    Returned as a 64-bit integer rather than the hex string it used to be, so
    the store can hold one per record in a numpy array. Same digest, same
    collision probability, an order of magnitude less memory.
    """
    import hashlib
    import re

    collapsed = re.sub(r"\s+", " ", text.lower()).strip()
    return int.from_bytes(
        hashlib.blake2b(collapsed.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
        "big",
    )


def _digest64(text: str) -> int:
    """A 64-bit content digest, as an integer.

    Not ``models.content_hash``: that returns a 32-character hex string, which
    costs 81 bytes as a Python object where the integer costs 32 and the
    information is the same. Two of these are held per distinct prompt, so on a
    million-prompt corpus the difference is a hundred megabytes.
    """
    import hashlib

    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(),
        "big",
    )


def _example(doc: Document, prompt: str) -> tuple[str, str, int, str]:
    """The location and a short quote for one record, for evidence."""
    return (doc.doc_id, doc.source_file, doc.source_index, excerpt(prompt, 120))


#: Bytes one tracked prompt costs: two digests, a count, and a location with a
#: 120-character quote. Used to turn a memory budget into a prompt ceiling.
BYTES_PER_TRACKED_PROMPT = 500


def _prompt_capacity() -> int:
    """Distinct prompts this machine will track for contradictions."""
    from ..hardware import plan

    budget = plan().memory_budget
    return max(
        ContradictorySupervision.MAX_TRACKED,
        (budget // 8) // BYTES_PER_TRACKED_PROMPT,
    )


class _SignatureStore:
    """Shared MinHash state, built once and used by both checks below.

    Held as parallel lists indexed by document key rather than as a dict of
    tuples. Keys are dense — they are positions in the accepted-record sequence
    — so the dict was a hash table whose keys were 0, 1, 2, …, costing about
    230 bytes per record in slots, boxed integers and tuple headers to store
    what a list stores in eight. On half a million records that is a hundred
    megabytes for nothing, on top of the index itself.

    Repeated strings are stored by reference. ``source_file`` and ``dataset``
    are the same object for every record in a file, so the lists hold pointers
    to one string rather than half a million copies.
    """

    #: Records this index will hold before it stops growing.
    #:
    #: A near-duplicate index is linear in the corpus and there is no way around
    #: that: every record has to be compared with every other one, so every
    #: record's signature has to be somewhere. At roughly a kilobyte apiece a
    #: ten-million-record corpus is ten gigabytes of index, which is a scan that
    #: does not finish on a laptop.
    #:
    #: So it stops, and says that it stopped. This is the same contract
    #: ``ContradictorySupervision`` has had since 1.0: index a prefix, set
    #: ``overflowed``, and have the finding state the count is a floor over the
    #: first N records rather than a rate over the corpus. A number with its
    #: coverage attached is useful; a number that quietly describes a tenth of
    #: the corpus is not.
    #:
    #: Sized from the machine at construction; this is the floor and the value
    #: used when nothing is known about available memory.
    MAX_INDEXED = 750_000

    def __init__(self, preset_name: str = DEFAULT_PRESET, *, capacity: int | None = None) -> None:
        self.preset_name = preset_name
        self.preset = PRESETS[preset_name]
        self.hasher = MinHasher(self.preset)
        self.index = LSHIndex(self.preset)
        self.capacity = capacity or self.MAX_INDEXED
        #: True once ``capacity`` was reached and records began to be skipped.
        self.overflowed = False
        #: Records offered to the index, including those the ceiling rejected.
        self.offered = 0
        self._doc_ids: list[str] = []
        self._files: list[str] = []
        self._indexes: list[int] = []
        self._excerpts: list[str] = []
        self._datasets: list[str] = []
        self._exact = np.zeros(0, dtype=np.uint64)
        self.counter = 0
        self.per_dataset: dict[str, int] = {}
        # Both checks below share this store, so the same record arrives twice
        # in a row. Without this guard every document is indexed under two keys
        # and each one becomes its own near-duplicate, which inflates the count
        # past the total number of records.
        #
        # One location rather than a set of every location seen. The two calls
        # for a record are consecutive — they happen inside that record's own
        # observe cycle, with no other record in between — so remembering the
        # last one is exactly as correct as remembering all of them, and does
        # not grow a 77 MB set alongside an index that already knows every
        # record it holds.
        self._last: tuple[str, int] | None = None

    def doc(self, key: int) -> tuple[str, str, int, str, str, int]:
        """``(doc_id, file, index, excerpt, dataset, exact key)`` for one key."""
        return (
            self._doc_ids[key], self._files[key], self._indexes[key],
            self._excerpts[key], self._datasets[key], int(self._exact[key]),
        )

    def exact_key(self, key: int) -> int:
        return int(self._exact[key])

    def _push_exact(self, digest: int) -> None:
        """Append one exact-text digest to the growable array.

        Held as a 64-bit integer rather than the sixteen-character hex string it
        used to be: same identity, 8 bytes instead of 73 once the string header
        and the list slot are counted.
        """
        if self.counter >= self._exact.size:
            grown = np.zeros(max(4096, self._exact.size * 2), dtype=np.uint64)
            grown[: self._exact.size] = self._exact
            self._exact = grown
        self._exact[self.counter] = digest

    @property
    def keys(self) -> range:
        return range(self.counter)

    # -- crossing a process boundary --------------------------------------

    def __getstate__(self) -> dict:
        """Ship signatures as one array and leave the band hashes behind.

        The band hashes are a derived index, reconstructible from the
        signatures they were computed over, so sending them would be several
        extra megabytes per shard for something the receiver can rebuild in
        microseconds.
        """
        return {
            "preset_name": self.preset_name,
            "capacity": self.capacity,
            "overflowed": self.overflowed,
            "offered": self.offered,
            "sigs": np.array(self.index._matrix(), copy=True),
            "doc_ids": self._doc_ids,
            "files": self._files,
            "indexes": self._indexes,
            "excerpts": self._excerpts,
            "datasets": self._datasets,
            "exact": self._exact[: self.counter],
            "per_dataset": self.per_dataset,
            "counter": self.counter,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["preset_name"], capacity=state["capacity"])  # type: ignore[misc]
        self.counter = state["counter"]
        self.overflowed = state["overflowed"]
        self.offered = state["offered"]
        self.per_dataset = state["per_dataset"]
        self._doc_ids = state["doc_ids"]
        self._files = state["files"]
        self._indexes = state["indexes"]
        self._excerpts = state["excerpts"]
        self._datasets = state["datasets"]
        self._exact = np.asarray(state["exact"], dtype=np.uint64)
        sigs = state["sigs"]
        for key in range(self.counter):
            self.index.add(key, sigs[key], self._datasets[key])

    def merge(self, other: _SignatureStore) -> None:
        """Append a later shard's signatures, renumbering its keys.

        Keys are positions in the accepted-record sequence, so a shard's local
        numbering becomes global by adding the count of everything before it.
        Shards are contiguous and merged in order, so the result is the exact
        numbering a serial pass would have produced.
        """
        base = self.counter
        self.offered += other.offered
        self.overflowed = self.overflowed or other.overflowed
        room = max(0, self.capacity - base)
        taken = min(other.counter, room)
        if taken < other.counter:
            self.overflowed = True
        for local in range(taken):
            self.index.add(base + local, other.index.signature(local),
                           other._datasets[local])
            self._push_exact_at(base + local, int(other._exact[local]))
        self._doc_ids.extend(other._doc_ids[:taken])
        self._files.extend(other._files[:taken])
        self._indexes.extend(other._indexes[:taken])
        self._excerpts.extend(other._excerpts[:taken])
        self._datasets.extend(other._datasets[:taken])
        self.counter = base + taken
        for dataset, count in other.per_dataset.items():
            self.per_dataset[dataset] = self.per_dataset.get(dataset, 0) + count

    def _push_exact_at(self, key: int, digest: int) -> None:
        while key >= self._exact.size:
            grown = np.zeros(max(4096, self._exact.size * 2), dtype=np.uint64)
            grown[: self._exact.size] = self._exact
            self._exact = grown
        self._exact[key] = digest

    def add(self, doc: Document) -> None:
        if len(doc.text) < 40:
            return
        location = (doc.source_file, doc.source_index)
        if location == self._last:
            return
        self._last = location
        self.offered += 1
        # The ceiling is checked before the signature is computed, so an
        # overflowing scan stops paying for MinHash as well as for storage.
        if self.counter >= self.capacity:
            self.overflowed = True
            return
        sig = self.hasher.signature_from_words(dedup_words_of(doc))
        if sig is None:
            return
        key = self.counter
        self._push_exact(_exact_key(doc.text))
        self.counter += 1
        self.index.add(key, sig, doc.dataset)
        self._doc_ids.append(doc.doc_id)
        self._files.append(doc.source_file)
        self._indexes.append(doc.source_index)
        self._excerpts.append(excerpt(doc.text, 160))
        self._datasets.append(doc.dataset)
        self.per_dataset[doc.dataset] = self.per_dataset.get(doc.dataset, 0) + 1


#: Bytes one indexed record costs: a 104-word signature, its band hashes, an
#: excerpt and the small strings around them. Measured, not guessed — see the
#: memory note in :class:`_SignatureStore`. Used only to turn a memory budget
#: into a record ceiling.
BYTES_PER_INDEXED_RECORD = 1100


def _index_capacity() -> int:
    """How many records this machine will let the near-duplicate index hold.

    A quarter of the scan's sample memory budget, floored at the class default.
    The near-duplicate index and the corpus sample are the two things that grow
    with the corpus, and they are both bounded so that a scan's peak memory is a
    function of the machine rather than of the dataset.
    """
    from ..hardware import plan

    budget = plan().memory_budget
    return max(_SignatureStore.MAX_INDEXED, (budget // 4) // BYTES_PER_INDEXED_RECORD)


def _get_store(ctx: ScanContext) -> _SignatureStore:
    store = ctx.stats.get("_minhash_store")
    if store is None:
        store = _SignatureStore(
            ctx.stats.get("minhash_preset", DEFAULT_PRESET),
            capacity=_index_capacity(),
        )
        ctx.stats["_minhash_store"] = store
    return store


@register
class NearDuplicates(Check):
    check_id = "T1-NDUP-001"
    title = "Records that are near-copies of each other"
    tier = 1
    profiles = ALL_PROFILES
    cost = CostClass.GLOBAL
    severity = Severity.WARNING
    fix = "Review the largest clusters first. Do not bulk-delete without measuring."
    rationale = (
        "Reported, not prescribed. FineWeb deduplicated Common Crawl across all snapshots, "
        "produced a corpus that scored below their baseline, and found on inspection that the "
        "data the filter had thrown away trained a better model than the data it kept. Their "
        "conclusion was that the benefit comes from removing very large clusters and that "
        "going further hurts. So this check gives you cluster sizes and lets you decide."
    )

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        _get_store(ctx).add(doc)

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        store = _get_store(ctx)
        if store.counter < 2:
            return []
        pairs = store.index.verified_pairs()
        if not pairs:
            return []

        uf = UnionFind()
        for a, b, _j in pairs:
            uf.union(a, b)
        clusters = uf.clusters()
        if not clusters:
            return []

        # Count distinct texts per cluster, not members. A cluster of 40 records
        # that are 39 exact copies of one text plus one near-copy contributes a
        # single near-duplicate here; the other 39 belong to T0-DUP-001 and are
        # already reported there.
        distinct_sizes: list[int] = []
        redundant = 0
        exact_absorbed = 0
        by_dataset: dict[str, int] = {}
        near_clusters = 0
        for members in clusters.values():
            seen_text: dict[int, int] = {}
            for m in members:
                seen_text.setdefault(store.exact_key(m), m)
            distinct = len(seen_text)
            exact_absorbed += len(members) - distinct
            if distinct < 2:
                continue
            near_clusters += 1
            distinct_sizes.append(distinct)
            redundant += distinct - 1
            for m in list(seen_text.values())[1:]:
                ds = store._datasets[m]
                by_dataset[ds] = by_dataset.get(ds, 0) + 1

        if not redundant:
            return []
        distinct_sizes.sort(reverse=True)

        # One example per record, and never a record paired with a copy of
        # itself: that pair is an exact duplicate, not a near one.
        evidence: list[Evidence] = []
        shown: set[str] = set()
        for a, b, j in sorted(pairs, key=lambda p: -p[2]):
            da, db = store.doc(a), store.doc(b)
            if da[5] == db[5] or da[0] in shown or db[0] in shown:
                continue
            shown.add(da[0])
            evidence.append(
                Evidence(da[0], da[1], da[2], da[3],
                         partner_doc_id=db[0], partner_excerpt=db[3], score=j)
            )
            if len(evidence) >= 6:
                break

        detail = (
            f"{redundant:,} near-copies across {near_clusters:,} clusters "
            f"at Jaccard >= {store.preset.threshold} "
            f"(largest cluster {distinct_sizes[0]:,} distinct texts); "
            f"preset {store.preset.name}: {store.preset.description}"
        )
        if exact_absorbed:
            detail += (
                f". A further {exact_absorbed:,} cluster members are exact copies "
                f"and are counted by T0-DUP-001 instead"
            )
        if store.overflowed:
            detail += (
                f". The index holds {store.counter:,} of the {store.offered:,} "
                f"eligible records — the ceiling this machine's memory allows — "
                f"so this count is a floor over that prefix rather than a rate "
                f"over the corpus"
            )
        return [
            make_finding(
                self, count=redundant, total=store.counter,
                detail=detail,
                evidence=evidence, by_dataset=by_dataset,
                data={
                    "preset": store.preset.name,
                    "threshold": store.preset.threshold,
                    "clusters": near_clusters,
                    "largest_cluster": distinct_sizes[0],
                    "cluster_sizes_top10": distinct_sizes[:10],
                    "exact_copies_excluded": exact_absorbed,
                    "indexed": store.counter,
                    "eligible": store.offered,
                    "index_truncated": store.overflowed,
                },
            )
        ]


@register
class CrossDatasetOverlap(Check):
    check_id = "T1-OVERLAP-001"
    title = "Datasets overlap with each other"
    tier = 1
    unit = "directed dataset pair"
    total_unit = "dataset"
    profiles = ALL_PROFILES
    requires = (Requirement.MULTIPLE_DATASETS,)
    cost = CostClass.GLOBAL
    severity = Severity.WARNING
    fix = "Where one dataset is largely contained in another, drop the redundant one."
    rationale = (
        "Nobody ships this, and everybody combining a folder of Hugging Face datasets needs "
        "it. The matrix is directional: containment of a small dataset inside a large one is "
        "the actionable case, and a symmetric similarity score hides it."
    )

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        _get_store(ctx).add(doc)

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        store = _get_store(ctx)
        if len(store.per_dataset) < 2:
            return []
        pairs = store.index.verified_pairs()
        if not pairs:
            return []

        # matched[a][b] = records in a that have a near-duplicate in b
        matched: dict[str, dict[str, set[int]]] = {}
        for key_a, key_b, _j in pairs:
            ds_a = store._datasets[key_a]
            ds_b = store._datasets[key_b]
            if ds_a == ds_b:
                continue
            matched.setdefault(ds_a, {}).setdefault(ds_b, set()).add(key_a)
            matched.setdefault(ds_b, {}).setdefault(ds_a, set()).add(key_b)

        if not matched:
            return []

        rows: list[dict[str, Any]] = []
        for src, targets in matched.items():
            denom = store.per_dataset.get(src, 0) or 1
            for tgt, keys in targets.items():
                rows.append({
                    "from": src,
                    "to": tgt,
                    "matched": len(keys),
                    "of": denom,
                    "fraction": len(keys) / denom,
                })
        rows.sort(key=lambda r: -r["fraction"])  # type: ignore[index,return-value]

        worst = rows[0]
        detail = (
            f"{worst['fraction']:.0%} of {worst['from']!r} records "
            f"({worst['matched']} of {worst['of']}) also appear in {worst['to']!r}"
        )
        if len(rows) > 1:
            detail += f"; {len(rows)} directed dataset pairs overlap in total"

        evidence: list[Evidence] = []
        for key_a, key_b, j in sorted(pairs, key=lambda p: -p[2]):
            da, db = store.doc(key_a), store.doc(key_b)
            if da[4] == db[4]:
                continue
            evidence.append(
                Evidence(da[0], da[1], da[2], f"[{da[4]}] {da[3]}",
                         partner_doc_id=db[0], partner_excerpt=f"[{db[4]}] {db[3]}", score=j)
            )
            if len(evidence) >= 5:
                break

        return [
            make_finding(
                self, count=len(rows), total=len(store.per_dataset), detail=detail,
                evidence=evidence,
                data={
                    "direction": "row appears in column",
                    "matrix": rows[:60],
                    "dataset_sizes": store.per_dataset,
                },
            )
        ]


@register
class ContradictorySupervision(Check):
    check_id = "T1-DUP-002"
    title = "The same prompt is answered two different ways"
    tier = 1
    unit = "prompt"
    total_unit = "record"
    profiles = (Profile.SFT,)
    cost = CostClass.GLOBAL
    severity = Severity.WARNING
    blocking_in = ()
    fix = "Keep one answer per prompt, or accept that the model is being taught to pick at random."
    rationale = (
        "Exact-duplicate detection finds records that are identical and reports them as "
        "redundant. The more damaging case is the opposite: the same prompt appearing several "
        "times with *different* answers. That is not redundancy, it is contradiction, and it "
        "is invisible to every duplicate check because the records are genuinely distinct. "
        "The gradient it produces points in two directions at once, and the usual cause is a "
        "merge of two datasets that overlap on prompts, or a generation run repeated with a "
        "different system prompt. Prompts are compared after whitespace normalisation only, "
        "so near-misses are not counted."
    )

    #: Prompts are held as digests rather than text, so a corpus with millions of
    #: records costs a digest per distinct prompt rather than the prompt itself.
    #:
    #: Even so this table is linear in distinct prompts, and at roughly 450 bytes
    #: an entry two million of them is most of a gigabyte. The ceiling is sized
    #: from the machine at construction, exactly as the near-duplicate index is;
    #: this is the floor and the value used when memory cannot be measured. The
    #: overflow contract is unchanged — the finding says the count is over the
    #: first N distinct prompts rather than over the corpus.
    MAX_TRACKED = 500_000

    MERGE_SUM = ("total",)
    #: Folded together in ``merge`` below: a prompt whose two answers land in
    #: different shards is exactly the case this check is about, and no
    #: per-attribute rule can see it.
    MERGE_CUSTOM = ("seen", "answers", "overflowed")
    #: The digest function and the ceiling, both fixed at construction.
    MERGE_IGNORE = ("_hash", "capacity")

    #: Set once MAX_TRACKED prompts are held, so the finding can say the count
    #: is a floor. Declared here because ``merge`` reads it before ``reset``
    #: runs on a fresh instance.
    overflowed: bool = False

    def merge(self, other: Check) -> None:
        super().merge(other)
        # Shards are merged by check id, so this is always the same class. The
        # signature stays the base one so the override does not narrow it.
        if not isinstance(other, ContradictorySupervision):
            return
        self.overflowed = self.overflowed or other.overflowed
        for pk, (first_ak, _n, first_record) in other.seen.items():
            mine = self.seen.get(pk)
            if mine is None:
                if len(self.seen) >= self.capacity:
                    self.overflowed = True
                    continue
                self.seen[pk] = (first_ak, _n, first_record)
                if pk in other.answers:
                    self.answers[pk] = set(other.answers[pk])
                continue
            # Both shards saw this prompt. The union of the answer digests is
            # the distinct-answer count, and the earlier shard keeps the example
            # because it holds the earlier record.
            bucket = self.answers.setdefault(pk, {mine[0]})
            bucket.add(first_ak)
            bucket.update(other.answers.get(pk, ()))
            self.seen[pk] = (mine[0], len(bucket), mine[2])

    def __init__(self, *, capacity: int | None = None) -> None:
        self._hash = _digest64
        self.capacity = capacity or _prompt_capacity()
        # prompt digest -> (first answer digest, distinct answers, an example)
        self.seen: dict[int, tuple[int, int, tuple[str, str, int, str]]] = {}
        self.answers: dict[int, set[int]] = {}
        self.total = 0
        self.overflowed = False

    def observe(self, doc: Document, ctx: ScanContext) -> None:
        if not doc.turns:
            return
        prompt = doc.prompt_text.strip()
        answer = doc.assistant_text.strip()
        if len(prompt) < 20 or not answer:
            return
        self.total += 1
        pk = self._hash(" ".join(prompt.split()))
        ak = self._hash(" ".join(answer.split()))
        existing = self.seen.get(pk)
        if existing is None:
            # New prompts stop being tracked at the ceiling; prompts already in
            # the table keep accumulating answers, because a contradiction found
            # among them is still a true contradiction.
            if len(self.seen) >= self.capacity:
                self.overflowed = True
                return
            self.seen[pk] = (ak, 1, _example(doc, prompt))
            return
        first_ak, _n, first_record = existing
        bucket = self.answers.setdefault(pk, {first_ak})
        bucket.add(ak)
        self.seen[pk] = (first_ak, len(bucket), first_record)

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        conflicts = {pk: n for pk, (_a, n, _r) in self.seen.items() if n > 1}
        if not conflicts:
            return []
        affected = sum(conflicts.values())
        evidence = [
            Evidence(r[0], r[1], r[2], f"{n} different answers to: {r[3]}")
            for pk, n in sorted(conflicts.items(), key=lambda kv: -kv[1])[:5]
            for r in (self.seen[pk][2],)
        ]
        detail = (
            f"{len(conflicts):,} prompt(s) appear more than once with differing answers, "
            f"covering {affected:,} of {self.total:,} records"
        )
        if self.overflowed:
            detail += f"; only the first {self.capacity:,} distinct prompts were tracked"
        return [
            make_finding(
                self, count=len(conflicts), total=self.total, detail=detail,
                evidence=evidence,
                data={"max_answers_for_one_prompt": max(conflicts.values())},
            )
        ]
