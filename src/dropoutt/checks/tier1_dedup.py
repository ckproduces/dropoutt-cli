"""Tier 1 near-duplicate and cross-dataset overlap checks.

Both are two-phase. Signatures accumulate during the streaming pass; clusters
and the overlap matrix are resolved at the end.

The overlap matrix is **directional and must stay that way**. If a 10k dataset
sits entirely inside a 1M dataset, the containment is 100% one way and 1% the
other. A symmetric Jaccard reports something small and hides the fact worth
acting on.
"""

from __future__ import annotations

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


def _exact_key(text: str) -> str:
    """Whitespace- and case-insensitive identity of a record's text.

    Near-duplicate clusters are full of exact copies, and the exact-duplicate
    check has already counted those. Reporting them twice inflates the second
    number and fills its examples with pairs the reader was shown a section
    earlier — on one corpus the top six "near-copies" were the same four rows,
    each matched against itself at 100%.
    """
    import hashlib
    import re

    collapsed = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.blake2b(collapsed.encode("utf-8", "surrogatepass"),
                           digest_size=8).hexdigest()


class _SignatureStore:
    """Shared MinHash state, built once and used by both checks below."""

    def __init__(self, preset_name: str = DEFAULT_PRESET) -> None:
        self.preset_name = preset_name
        self.preset = PRESETS[preset_name]
        self.hasher = MinHasher(self.preset)
        self.index = LSHIndex(self.preset)
        #: key → (doc_id, file, index, excerpt, dataset, exact-text key)
        self.docs: dict[int, tuple[str, str, int, str, str, str]] = {}
        self.counter = 0
        self.per_dataset: dict[str, int] = {}
        # Both checks below share this store, so the same record arrives twice.
        # Without this guard every document is indexed under two keys and each
        # one becomes its own near-duplicate, which inflates the count past the
        # total number of records.
        self._seen: set[tuple[str, int]] = set()

    # -- crossing a process boundary --------------------------------------

    def __getstate__(self) -> dict:
        """Ship signatures as one array and leave the buckets behind.

        The LSH buckets are a derived index: fourteen byte-string keys per
        document, all of them reconstructible. Pickling them would send several
        times the payload of the signatures they were built from, and pickling
        fifty thousand small arrays individually is slower than sending one.
        """
        keys = sorted(self.docs)
        sigs = self.index._sigs
        stacked = (
            np.stack([sigs[k] for k in keys]) if keys
            else np.empty((0, self.preset.num_perm), dtype=np.uint32)
        )
        return {
            "preset_name": self.preset_name,
            "keys": keys,
            "sigs": stacked,
            "docs": [self.docs[k] for k in keys],
            "per_dataset": self.per_dataset,
            "counter": self.counter,
        }

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["preset_name"])  # type: ignore[misc]
        self.counter = state["counter"]
        self.per_dataset = state["per_dataset"]
        sigs = state["sigs"]
        for row, (key, doc) in enumerate(zip(state["keys"], state["docs"], strict=True)):
            self.docs[key] = doc
            self.index.add(key, sigs[row], doc[4])
            self._seen.add((doc[1], doc[2]))

    def merge(self, other: _SignatureStore) -> None:
        """Append a later shard's signatures, renumbering its keys.

        Keys are positions in the accepted-record sequence, so a shard's local
        numbering becomes global by adding the count of everything before it.
        Shards are contiguous and merged in order, so the result is the exact
        numbering a serial pass would have produced.
        """
        base = self.counter
        for local in sorted(other.docs):
            doc = other.docs[local]
            key = base + local
            self.docs[key] = doc
            self.index.add(key, other.index._sigs[local], doc[4])
            self._seen.add((doc[1], doc[2]))
        self.counter = base + other.counter
        for dataset, count in other.per_dataset.items():
            self.per_dataset[dataset] = self.per_dataset.get(dataset, 0) + count

    def add(self, doc: Document) -> None:
        if len(doc.text) < 40:
            return
        location = (doc.source_file, doc.source_index)
        if location in self._seen:
            return
        self._seen.add(location)
        sig = self.hasher.signature_from_words(dedup_words_of(doc))
        if sig is None:
            return
        key = self.counter
        self.counter += 1
        self.index.add(key, sig, doc.dataset)
        self.docs[key] = (doc.doc_id, doc.source_file, doc.source_index,
                          excerpt(doc.text, 160), doc.dataset,
                          _exact_key(doc.text))
        self.per_dataset[doc.dataset] = self.per_dataset.get(doc.dataset, 0) + 1


def _get_store(ctx: ScanContext) -> _SignatureStore:
    store = ctx.stats.get("_minhash_store")
    if store is None:
        store = _SignatureStore(ctx.stats.get("minhash_preset", DEFAULT_PRESET))
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
            seen_text: dict[str, int] = {}
            for m in members:
                seen_text.setdefault(store.docs[m][5], m)
            distinct = len(seen_text)
            exact_absorbed += len(members) - distinct
            if distinct < 2:
                continue
            near_clusters += 1
            distinct_sizes.append(distinct)
            redundant += distinct - 1
            for m in list(seen_text.values())[1:]:
                ds = store.docs[m][4]
                by_dataset[ds] = by_dataset.get(ds, 0) + 1

        if not redundant:
            return []
        distinct_sizes.sort(reverse=True)

        # One example per record, and never a record paired with a copy of
        # itself: that pair is an exact duplicate, not a near one.
        evidence: list[Evidence] = []
        shown: set[str] = set()
        for a, b, j in sorted(pairs, key=lambda p: -p[2]):
            da, db = store.docs[a], store.docs[b]
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
            ds_a = store.docs[key_a][4]
            ds_b = store.docs[key_b][4]
            if ds_a == ds_b:
                continue
            matched.setdefault(ds_a, {}).setdefault(ds_b, set()).add(key_a)
            matched.setdefault(ds_b, {}).setdefault(ds_a, set()).add(key_b)

        if not matched:
            return []

        rows: list[dict[str, object]] = []
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
            da, db = store.docs[key_a], store.docs[key_b]
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
    #: records costs 16 bytes per distinct prompt rather than the prompt itself.
    MAX_TRACKED = 2_000_000

    MERGE_SUM = ("total",)
    #: Folded together in ``merge`` below: a prompt whose two answers land in
    #: different shards is exactly the case this check is about, and no
    #: per-attribute rule can see it.
    MERGE_CUSTOM = ("seen", "answers", "overflowed")
    #: The digest function, bound once at construction.
    MERGE_IGNORE = ("_hash",)

    def merge(self, other: ContradictorySupervision) -> None:
        super().merge(other)
        self.overflowed = self.overflowed or other.overflowed
        for pk, (first_ak, _n, first_record) in other.seen.items():
            mine = self.seen.get(pk)
            if mine is None:
                if len(self.seen) >= self.MAX_TRACKED:
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

    def __init__(self) -> None:
        from ..models import content_hash

        self._hash = content_hash
        # prompt digest -> (first answer digest, distinct answers, an example)
        self.seen: dict[str, tuple[str, int, tuple[str, str, int, str]]] = {}
        self.answers: dict[str, set[str]] = {}
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
        if len(self.seen) >= self.MAX_TRACKED:
            self.overflowed = True
            return
        pk = self._hash(" ".join(prompt.split()))
        ak = self._hash(" ".join(answer.split()))
        record = (doc.doc_id, doc.source_file, doc.source_index, excerpt(prompt, 120))
        if pk not in self.seen:
            self.seen[pk] = (ak, 1, record)
            return
        first_ak, _n, first_record = self.seen[pk]
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
            detail += f"; only the first {self.MAX_TRACKED:,} distinct prompts were tracked"
        return [
            make_finding(
                self, count=len(conflicts), total=self.total, detail=detail,
                evidence=evidence,
                data={"max_answers_for_one_prompt": max(conflicts.values())},
            )
        ]
