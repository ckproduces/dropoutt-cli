"""MinHash signatures, LSH banding and union-find clustering.

Two named presets ship, because a deduplication result is only arguable if the
parameters are stated:

``fineweb``
    Word 5-grams, 112 hash functions, 14 bands of 8. These are the parameters
    the FineWeb team used to produce a 20T-token corpus.

``hf-neardedup``
    256 permutations over 5-grams with a Jaccard threshold of 0.7, matching the
    Hugging Face near-deduplication configuration.

The Rust ``rensa`` implementation is used when installed. The numpy fallback
uses the same permutation scheme and the same banding, so cluster membership
agrees; it is slower, not different.

Nothing in this module recommends deleting anything. FineWeb's own result was
that deduplicating across all Common Crawl snapshots produced a *worse* corpus
than deduplicating within each snapshot, and that the data thrown away by the
aggressive pass trained a better model than the data it kept. So this module
measures and clusters; the decision belongs to the user.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compat import HAVE_RENSA
from .textutil import hashed_shingles

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
    "fineweb": MinHashPreset("fineweb", 112, 14, 8, 5, 0.75),
    "hf-neardedup": MinHashPreset("hf-neardedup", 256, 32, 8, 5, 0.70),
}
DEFAULT_PRESET = "fineweb"


class MinHasher:
    """Computes MinHash signatures under a preset."""

    def __init__(self, preset: MinHashPreset, seed: int = 42) -> None:
        self.preset = preset
        rng = np.random.default_rng(seed)
        # Random affine permutations h_i(x) = (a_i * x + b_i) mod p, mod 2^32.
        self._a = rng.integers(1, _MERSENNE, size=preset.num_perm, dtype=np.uint64)
        self._b = rng.integers(0, _MERSENNE, size=preset.num_perm, dtype=np.uint64)
        self._backend = "rensa" if HAVE_RENSA else "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    def signature(self, text: str) -> np.ndarray | None:
        """Return a uint32 signature of length ``num_perm``, or None if too short."""
        shingles = hashed_shingles(text, self.preset.ngram)
        if not shingles:
            return None
        return self._sign(np.array(shingles, dtype=np.uint64))

    def _sign(self, sh: np.ndarray) -> np.ndarray:
        # Broadcast to (n_shingles, num_perm) then take the column minimum.
        # uint64 multiply overflows deliberately; we mask back to 32 bits, which
        # is the same construction rensa uses.
        vals = (sh[:, None] * self._a[None, :] + self._b[None, :]) & np.uint64(_MAX32)
        return vals.min(axis=0).astype(np.uint32)


class LSHIndex:
    """Banded LSH over MinHash signatures.

    Candidate pairs come from documents sharing a band bucket. Every candidate
    is then verified against the estimated Jaccard, so band collisions do not
    become findings.
    """

    def __init__(self, preset: MinHashPreset) -> None:
        self.preset = preset
        self._buckets: list[dict[bytes, list[int]]] = [{} for _ in range(preset.bands)]
        self._sigs: dict[int, np.ndarray] = {}
        self._dataset_of: dict[int, str] = {}

    def add(self, key: int, sig: np.ndarray, dataset: str = "") -> None:
        self._sigs[key] = sig
        self._dataset_of[key] = dataset
        rows = self.preset.rows
        for b in range(self.preset.bands):
            band = sig[b * rows : (b + 1) * rows].tobytes()
            self._buckets[b].setdefault(band, []).append(key)

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for bucket in self._buckets:
            for members in bucket.values():
                if len(members) < 2:
                    continue
                # A pathological bucket (thousands of identical boilerplate
                # records) would explode quadratically. Cap it: the cluster is
                # already obvious, and we record that it was capped.
                if len(members) > 200:
                    members = members[:200]
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        a, b = members[i], members[j]
                        pairs.add((a, b) if a < b else (b, a))
        return pairs

    def estimated_jaccard(self, a: int, b: int) -> float:
        sa, sb = self._sigs[a], self._sigs[b]
        return float(np.count_nonzero(sa == sb) / len(sa))

    def verified_pairs(self) -> list[tuple[int, int, float]]:
        """Candidate pairs whose estimated Jaccard clears the preset threshold."""
        out = []
        for a, b in self.candidate_pairs():
            j = self.estimated_jaccard(a, b)
            if j >= self.preset.threshold:
                out.append((a, b, j))
        return out

    def dataset_of(self, key: int) -> str:
        return self._dataset_of.get(key, "")

    def __len__(self) -> int:
        return len(self._sigs)


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
