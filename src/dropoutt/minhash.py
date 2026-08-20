"""MinHash signatures, LSH banding and union-find clustering.

Two named presets ship, because a deduplication result is only arguable if the
parameters are stated:

``fineweb``
    Word 5-grams, 112 hash functions, 14 bands of 8. These are the parameters
    the FineWeb team used to produce a 20T-token corpus.

``hf-neardedup``
    256 permutations over 5-grams with a Jaccard threshold of 0.7, matching the
    Hugging Face near-deduplication configuration.

Signing is numpy, always, and deliberately so. A Rust MinHash would be faster,
but it would seed its permutations differently, which means a corpus scanned on
a machine with the accelerator installed would produce different signatures,
different LSH buckets and — on borderline pairs — a different duplicate count
from the same corpus scanned without it. The scan already guarantees that its
findings do not depend on how many cores it was given; making them depend on
which wheels happen to be present would give that away for a few percent of one
phase.

Nothing in this module recommends deleting anything. FineWeb's own result was
that deduplicating across all Common Crawl snapshots produced a *worse* corpus
than deduplicating within each snapshot, and that the data thrown away by the
aggressive pass trained a better model than the data it kept. So this module
measures and clusters; the decision belongs to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .textutil import dedup_words, shingle_hashes

_MERSENNE = (1 << 61) - 1
_MAX32 = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class MinHashPreset:
    name: str
    num_perm: int
    bands: int
    rows: int
    ngram: int
    threshold: float

    @property
    def description(self) -> str:
        return (
            f"{self.ngram}-gram, {self.num_perm} hashes, "
            f"{self.bands} bands of {self.rows}, Jaccard >= {self.threshold}"
        )


PRESETS = {
    # 8 bands of 13 puts the S-curve's inflection at (1/8)^(1/13) = 0.852, so
    # the banding agrees with the threshold instead of admitting a wide band of
    # candidates and rejecting most of them afterwards.
    "strict": MinHashPreset("strict", 104, 8, 13, 5, 0.85),
    "fineweb": MinHashPreset("fineweb", 112, 14, 8, 5, 0.75),
    "hf-neardedup": MinHashPreset("hf-neardedup", 256, 32, 8, 5, 0.70),
}
#: ``strict`` rather than ``fineweb``. At 0.75 the check fired on records that
#: share a template or a category column but say different things — two support
#: answers about different products, two rows of a QA export whose third column
#: repeats a category name. Those are similar; calling them duplicates and
#: putting a delete-shaped finding next to them is wrong. FineWeb's 0.75 is
#: right for its own job, filtering web crawl at trillion-token scale, and is
#: kept under its own name for anyone who wants it.
DEFAULT_PRESET = "strict"


class MinHasher:
    """Computes MinHash signatures under a preset."""

    def __init__(self, preset: MinHashPreset, seed: int = 42) -> None:
        self.preset = preset
        rng = np.random.default_rng(seed)
        # Random affine permutations h_i(x) = (a_i * x + b_i) mod p, mod 2^32.
        self._a = rng.integers(1, _MERSENNE, size=preset.num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MERSENNE, size=preset.num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray | None:
        """Return a uint32 signature of length ``num_perm``, or None if too short."""
        return self.signature_from_words(dedup_words(text))

    def signature_from_words(self, words: list[str]) -> np.ndarray | None:
        """Same, from an already-normalised word list.

        The scan normalises each record once and hands the words to both this
        and the contamination scanner, rather than each of them re-running the
        NFC pass, the Turkish-aware case fold and two substitutions.
        """
        shingles = shingle_hashes(words, self.preset.ngram)
        if shingles.size == 0:
            return None
        return self._sign(shingles)

    def _sign(self, sh: np.ndarray) -> np.ndarray:
        # Broadcast to (n_shingles, num_perm) then take the column minimum.
        # uint64 multiply overflows deliberately; we mask back to 32 bits, which
        # is the same construction rensa uses.
        #
        # Chunk long documents so the temporary (n_shingles, num_perm) matrix
        # stays cache-friendly. Column-wise min over chunks equals min over all.
        chunk = 4096
        if len(sh) <= chunk:
            vals = (sh[:, None] * self._a[None, :] + self._b[None, :]) & np.uint64(_MAX32)
            return vals.min(axis=0).astype(np.uint32)
        mins = np.full(self.preset.num_perm, np.iinfo(np.uint32).max, dtype=np.uint64)
        for i in range(0, len(sh), chunk):
            part = sh[i : i + chunk]
            vals = (part[:, None] * self._a[None, :] + self._b[None, :]) & np.uint64(_MAX32)
            mins = np.minimum(mins, vals.min(axis=0))
        return mins.astype(np.uint32)


#: How many signatures the index grows by when it runs out of room. Growth is
#: geometric above this, so a large corpus does not pay a copy per thousand
#: records, and a small one does not reserve megabytes it will never use.
_GROW_MIN = 4096


class LSHIndex:
    """Banded LSH over MinHash signatures.

    Candidate pairs come from documents sharing a band bucket. Every candidate
    is then verified against the estimated Jaccard, so band collisions do not
    become findings.

    **Everything here is an array, and that is the point.** The obvious
    implementation of banded LSH is one dictionary per band mapping a band's
    bytes to the documents that share it, and that is what this was. It costs
    roughly 270 bytes per document per band in Python object overhead — a bytes
    key, a list, a dict slot — so at fourteen bands it is about 3.7 KB of
    bookkeeping for a document whose signature is 416 bytes. On a corpus of half
    a million records that is a gigabyte and a half of dictionaries, which was
    the single largest thing a scan held in memory.

    A band is stored instead as one 64-bit hash in a dense ``(documents, bands)``
    array: 8 bytes per document per band, a factor of thirty smaller, and the
    grouping is recovered at the end by sorting a column and taking runs of
    equal values. Sorting five million integers is also faster than five million
    dictionary insertions, so the second phase got quicker as well as smaller.

    The one behavioural difference is that two different bands could in
    principle hash to the same 64-bit value and be grouped together. That is a
    candidate pair, not a finding: every candidate is verified against the
    estimated Jaccard before it counts, so a collision at 2^-64 costs one
    comparison and changes no number.
    """

    def __init__(self, preset: MinHashPreset) -> None:
        self.preset = preset
        self._sigs_array = np.zeros((0, preset.num_perm), dtype=np.uint32)
        #: One 64-bit hash per (document, band).
        self._bands = np.zeros((0, preset.bands), dtype=np.uint64)
        self._datasets: list[str] = []
        self._next_key = 0
        self._verified: list[tuple[int, int, float]] | None = None

    # -- construction ------------------------------------------------------

    def _reserve(self, needed: int) -> None:
        if needed <= self._sigs_array.shape[0]:
            return
        size = max(needed, _GROW_MIN, self._sigs_array.shape[0] * 2)
        sigs = np.zeros((size, self.preset.num_perm), dtype=np.uint32)
        sigs[: self._sigs_array.shape[0]] = self._sigs_array
        bands = np.zeros((size, self.preset.bands), dtype=np.uint64)
        bands[: self._bands.shape[0]] = self._bands
        self._sigs_array = sigs
        self._bands = bands

    def add(self, key: int, sig: np.ndarray, dataset: str = "") -> None:
        self._reserve(key + 1)
        self._sigs_array[key] = sig
        self._bands[key] = _band_hashes(sig, self.preset.bands, self.preset.rows)
        while len(self._datasets) <= key:
            self._datasets.append("")
        self._datasets[key] = dataset
        self._next_key = max(self._next_key, key + 1)
        self._verified = None

    #: A pathological bucket — thousands of identical boilerplate records —
    #: would explode quadratically. The cluster is already obvious well before
    #: then, so only this many members of one bucket are paired up.
    BUCKET_CAP = 200

    def _encoded_candidates(self) -> np.ndarray:
        """Candidate pairs as ``low * n + high``, sorted and deduplicated.

        Packed into one integer per pair and deduplicated with ``np.unique``
        rather than accumulated in a set of tuples. On a corpus with real
        duplicate clusters this is tens of millions of insertions, and a Python
        set of tuples was the most expensive single operation in the scan's
        second phase.
        """
        span = max(self._next_key, 1)
        n = self._next_key
        if n < 2:
            return np.empty(0, dtype=np.int64)
        chunks: list[np.ndarray] = []
        for band in range(self.preset.bands):
            column = self._bands[:n, band]
            order = np.argsort(column, kind="stable")
            sorted_hashes = column[order]
            # Run starts: where the value changes. Every run of length two or
            # more is a bucket, and this is the dictionary that used to be.
            edges = np.flatnonzero(
                np.concatenate(([True], sorted_hashes[1:] != sorted_hashes[:-1], [True]))
            )
            starts, stops = edges[:-1], edges[1:]
            sizes = stops - starts
            for start, size in zip(
                starts[sizes > 1].tolist(), sizes[sizes > 1].tolist(), strict=True
            ):
                members = order[start : start + min(size, self.BUCKET_CAP)]
                i, j = np.triu_indices(members.size, k=1)
                a = members[i].astype(np.int64)
                b = members[j].astype(np.int64)
                chunks.append(np.minimum(a, b) * span + np.maximum(a, b))
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(chunks))

    def candidate_pairs(self) -> set[tuple[int, int]]:
        span = max(self._next_key, 1)
        encoded = self._encoded_candidates()
        return {(int(p // span), int(p % span)) for p in encoded.tolist()}

    def estimated_jaccard(self, a: int, b: int) -> float:
        sa, sb = self._sigs_array[a], self._sigs_array[b]
        return float(np.count_nonzero(sa == sb) / len(sa))

    def verified_pairs(self) -> list[tuple[int, int, float]]:
        """Candidate pairs whose estimated Jaccard clears the preset threshold.

        The comparison is one matrix operation over the stacked signatures
        instead of a Python call per pair, and the result is kept: the
        near-duplicate check and the cross-dataset overlap check both need it,
        and it used to be computed from scratch for each.
        """
        if self._verified is not None:
            return self._verified
        self._verified = self._verify()
        return self._verified

    def _verify(self) -> list[tuple[int, int, float]]:
        encoded = self._encoded_candidates()
        if encoded.size == 0:
            return []
        span = max(self._next_key, 1)
        lo = (encoded // span).astype(np.int64)
        hi = (encoded % span).astype(np.int64)

        matrix = self._matrix()
        perms = matrix.shape[1]
        out: list[tuple[int, int, float]] = []
        # Chunked so the (pairs x permutations) comparison stays in cache even
        # when a corpus produces millions of candidates.
        step = max(1, (1 << 22) // max(perms, 1))
        for start in range(0, encoded.size, step):
            a = lo[start : start + step]
            b = hi[start : start + step]
            scores = np.count_nonzero(matrix[a] == matrix[b], axis=1) / perms
            keep = np.nonzero(scores >= self.preset.threshold)[0]
            for k in keep.tolist():
                out.append((int(a[k]), int(b[k]), float(scores[k])))
        return out

    def _matrix(self) -> np.ndarray:
        """Signatures stacked into one array, indexable by document key."""
        return self._sigs_array[: self._next_key]

    def signature(self, key: int) -> np.ndarray:
        return self._sigs_array[key]

    def dataset_of(self, key: int) -> str:
        return self._datasets[key] if key < len(self._datasets) else ""

    def __len__(self) -> int:
        return self._next_key


def _band_hashes(sig: np.ndarray, bands: int, rows: int) -> np.ndarray:
    """One 64-bit value per band, standing in for the band's raw bytes.

    FNV-1a over the band's 32-bit words, computed with numpy's wrapping
    unsigned arithmetic. The identity that matters is only that equal bands
    hash equally; the reverse holds to 2^-64, and a collision produces a
    candidate pair that the Jaccard verification then rejects.
    """
    words = sig[: bands * rows].reshape(bands, rows).astype(np.uint64)
    acc = np.full(bands, np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        for column in range(rows):
            acc = (acc ^ words[:, column]) * prime
    return acc


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self._parent.setdefault(x, x)
        while p != self._parent[p]:
            self._parent[p] = self._parent[self._parent[p]]
            p = self._parent[p]
        return p

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for node in self._parent:
            out.setdefault(self.find(node), []).append(node)
        return {root: members for root, members in out.items() if len(members) > 1}
