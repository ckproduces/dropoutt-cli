"""Benchmark contamination indices.

What ships is an **index, not the benchmark**. For each evaluation instance we
extract its word 8-grams, hash each to 64 bits, and store a mapping from hash to
the instance ids containing it. Raw text is not stored, but known candidate
phrases can be tested against the unkeyed hashes. Private indices therefore stay
inside the evaluation set's trust boundary. An index for a ten-thousand-instance
benchmark is a couple of megabytes.

Mapping to instance ids rather than to mere presence is required, because the
Tulu 3 rule is evaluated per evaluation instance:

    an eval instance is contaminated when more than 50% of its tokens are
    covered by 8-gram matches against a single training instance

    a training set is contaminated when more than 2% of any evaluation's
    instances match

Three kinds of benchmark, and only the first is solved by shipping an index:

1. Public test sets, which we prebuild and distribute.
2. The user's own held-out set, handled by ``dropoutt index-eval`` locally, so
   the evaluation that matters most never leaves their machine.
3. Third-party private benchmarks, which nobody can index but their maintainer.
   The format below is deliberately simple and documented so a maintainer can
   publish an index for a private split without revealing its questions.

At scan time every loaded index is merged into **one** sorted array of gram
hashes with a compressed posting list beside it, and a record is matched with a
single vectorised binary search. The dictionary-per-benchmark form it replaces
cost one probe per benchmark per 8-gram — sixty-eight million dictionary lookups
on a fifty-thousand-record corpus, the largest single line in the profile — and
about ten times the memory, which is what makes running the scan across several
processes affordable.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compat import json_loads
from .textutil import hash64, normalize_for_dedup

NGRAM = 8
MAGIC = b"DTTIDX01"

#: Tulu 3 thresholds. Echoed in the report so the number is never presented
#: without the rule that produced it.
TOKEN_FRACTION = 0.50
INSTANCE_FRACTION = 0.02


def eval_ngrams(text: str, n: int = NGRAM) -> list[int]:
    """Hashed word n-grams of one record.

    Used for index building and by callers outside the scan loop. The scan
    itself calls :func:`dropoutt.textutil.eval_ngram_hashes` on an already
    normalised word list, so normalisation runs once per record rather than once
    per consumer.
    """
    words = normalize_for_dedup(text).split()
    if len(words) < n:
        return [hash64(" ".join(words))] if words else []
    return [hash64(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)]


@dataclass(slots=True)
class BenchmarkIndex:
    """Hashed 8-grams for one benchmark.

    Two shapes, one class. While an index is being *built* the postings live in
    a dict, because instances arrive one at a time. Once *loaded* they live in
    arrays, because from then on the only operation is lookup.
    """

    name: str
    n_instances: int
    #: gram hash -> instance ids containing it. Build-time only; empty for an
    #: index read from disk.
    postings: dict[int, list[int]] = field(default_factory=dict)
    #: instance id -> number of distinct grams, the coverage denominator.
    instance_size: list[int] = field(default_factory=list)
    license: str | None = None
    source: str | None = None
    #: Loaded form: gram hashes in file order, CSR offsets into ``post_inst``.
    grams: np.ndarray | None = None
    post_start: np.ndarray | None = None
    post_inst: np.ndarray | None = None

    def add_instance(self, instance_id: int, text: str) -> None:
        grams = set(eval_ngrams(text))
        self.instance_size.append(len(grams))
        for g in grams:
            self.postings.setdefault(g, []).append(instance_id)

    def save(self, path: Path) -> None:
        """Write a compact binary index.

        Layout: magic, a JSON header, then the postings as packed integers. No
        benchmark text appears anywhere in the file.
        """
        header = {
            "name": self.name,
            "n_instances": self.n_instances,
            "license": self.license,
            "source": self.source,
            "ngram": NGRAM,
            "instance_size": self.instance_size,
        }
        import json

        head = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(MAGIC)
            fh.write(struct.pack("<I", len(head)))
            fh.write(head)
            fh.write(struct.pack("<I", len(self.postings)))
            for gram, ids in self.postings.items():
                fh.write(struct.pack("<QH", gram, min(len(ids), 65535)))
                for i in ids[:65535]:
                    fh.write(struct.pack("<I", i))

    @classmethod
    def load(cls, path: Path) -> BenchmarkIndex:
        with open(path, "rb") as fh:
            if fh.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path} is not a dropoutt contamination index")
            (head_len,) = struct.unpack("<I", fh.read(4))
            header = json_loads(fh.read(head_len))
            (n_grams,) = struct.unpack("<I", fh.read(4))
            body = fh.read()

        grams = np.empty(n_grams, dtype=np.uint64)
        starts = np.empty(n_grams + 1, dtype=np.int64)
        chunks: list[np.ndarray] = []
        unpack_head = struct.Struct("<QH").unpack_from
        offset = 0
        total = 0
        for i in range(n_grams):
            gram, count = unpack_head(body, offset)
            offset += 10
            grams[i] = gram
            starts[i] = total
            if count:
                chunks.append(np.frombuffer(body, dtype="<u4", count=count, offset=offset))
                offset += 4 * count
                total += count
        starts[n_grams] = total
        post_inst = (
            np.concatenate(chunks).astype(np.int32, copy=False)
            if chunks else np.empty(0, dtype=np.int32)
        )
        return cls(
            name=header["name"],
            n_instances=header["n_instances"],
            instance_size=header.get("instance_size", []),
            license=header.get("license"),
            source=header.get("source"),
            grams=grams,
            post_start=starts,
            post_inst=post_inst,
        )

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(gram hashes, CSR offsets, instance ids), building them if needed."""
        if self.grams is not None and self.post_start is not None:
            return self.grams, self.post_start, self.post_inst  # type: ignore[return-value]
        items = sorted(self.postings.items())
        grams = np.fromiter((g for g, _ in items), dtype=np.uint64, count=len(items))
        counts = np.fromiter((len(v) for _, v in items), dtype=np.int64, count=len(items))
        starts = np.empty(len(items) + 1, dtype=np.int64)
        starts[0] = 0
        np.cumsum(counts, out=starts[1:])
        flat = np.fromiter(
            (i for _, ids in items for i in ids), dtype=np.int32, count=int(starts[-1])
        )
        return grams, starts, flat


@dataclass
class ContaminationIndex:
    """All loaded benchmark indices, plus the accumulator for one scan."""

    benchmarks: dict[str, BenchmarkIndex] = field(default_factory=dict)
    #: benchmark name -> best covered-gram count per instance, from one record.
    _best: dict[str, np.ndarray] = field(default_factory=dict)
    #: (benchmark, instance id) -> the training record that produced it
    _witness: dict[tuple[str, int], tuple[str, str, int]] = field(default_factory=dict)
    #: Merged lookup, built on first use.
    _keys: np.ndarray | None = None
    _own_start: np.ndarray | None = None
    _own_bench: np.ndarray | None = None
    _own_inst: np.ndarray | None = None
    _names: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.benchmarks

    def add(self, index: BenchmarkIndex) -> None:
        self.benchmarks[index.name] = index
        self._keys = None  # invalidate the merged view

    # -- merged lookup ----------------------------------------------------

    def _build(self) -> None:
        """Fold every benchmark into one sorted key array with CSR postings."""
        self._names = list(self.benchmarks)
        gram_parts: list[np.ndarray] = []
        bench_parts: list[np.ndarray] = []
        inst_parts: list[np.ndarray] = []
        for bench_id, name in enumerate(self._names):
            grams, starts, inst = self.benchmarks[name]._arrays()
            if inst.size == 0:
                continue
            gram_parts.append(np.repeat(grams, np.diff(starts)))
            bench_parts.append(np.full(inst.size, bench_id, dtype=np.uint8))
            inst_parts.append(inst)
        if not gram_parts:
            self._keys = np.empty(0, dtype=np.uint64)
            self._own_start = np.zeros(1, dtype=np.int64)
            self._own_bench = np.empty(0, dtype=np.uint8)
            self._own_inst = np.empty(0, dtype=np.int32)
            return
        flat_grams = np.concatenate(gram_parts)
        order = np.argsort(flat_grams, kind="stable")
        flat_grams = flat_grams[order]
        self._own_bench = np.concatenate(bench_parts)[order]
        self._own_inst = np.concatenate(inst_parts)[order]
        keys, first = np.unique(flat_grams, return_index=True)
        self._keys = keys
        self._own_start = np.append(first, flat_grams.size).astype(np.int64)

    def _ensure(self) -> None:
        if self._keys is None:
            self._build()

    # -- accumulation -----------------------------------------------------

    def observe(self, text: str, doc_id: str, source_file: str, source_index: int) -> None:
        """Match one training record against every loaded benchmark."""
        from .textutil import dedup_words, eval_ngram_hashes

        self.observe_hashes(
            eval_ngram_hashes(dedup_words(text), NGRAM), doc_id, source_file, source_index
        )

    def observe_hashes(
        self, grams: np.ndarray, doc_id: str, source_file: str, source_index: int
    ) -> None:
        """Match one record's already-hashed 8-grams.

        The grams must be sorted and unique, which is what
        :func:`dropoutt.textutil.eval_ngram_hashes` returns.
        """
        self._ensure()
        keys = self._keys
        if keys is None or keys.size == 0 or grams.size == 0:
            return
        pos = np.searchsorted(keys, grams)
        np.clip(pos, 0, keys.size - 1, out=pos)
        hits = pos[keys[pos] == grams]
        if hits.size == 0:
            return

        start = self._own_start
        bench_of = self._own_bench
        inst_of = self._own_inst
        covered: dict[tuple[int, int], int] = {}
        for p in hits.tolist():
            for k in range(int(start[p]), int(start[p + 1])):
                key = (int(bench_of[k]), int(inst_of[k]))
                covered[key] = covered.get(key, 0) + 1

        names = self._names
        for (bench_id, inst), count in covered.items():
            name = names[bench_id]
            best = self._best.get(name)
            if best is None:
                best = np.zeros(self.benchmarks[name].n_instances, dtype=np.int32)
                self._best[name] = best
            if inst < best.size and count > best[inst]:
                best[inst] = count
                self._witness[(name, inst)] = (doc_id, source_file, source_index)

    def reset(self) -> None:
        """Drop what has been observed, keeping the loaded indices."""
        self._best = {}
        self._witness = {}

    def take_accumulator(self) -> tuple[dict[str, np.ndarray], dict[tuple[str, int], tuple]]:
        """Hand back only what one shard observed.

        A worker must not ship the merged key array home: it is tens of
        megabytes, identical in every process, and the parent already has it.
        """
        return self._best, self._witness

    def merge_accumulator(
        self,
        best_by_name: dict[str, np.ndarray],
        witnesses: dict[tuple[str, int], tuple],
    ) -> None:
        """Fold a shard's observations into this index.

        The rule is a maximum per evaluation instance, and a maximum is
        order-independent, so a sharded scan and a serial one agree exactly.
        Witnesses follow their maximum; a tie keeps the earlier shard, which
        holds the earlier record.
        """
        for name, other_best in best_by_name.items():
            best = self._best.get(name)
            if best is None:
                self._best[name] = other_best.copy()
                improved = np.nonzero(other_best)[0]
            else:
                improved = np.nonzero(other_best > best)[0]
                np.maximum(best, other_best, out=best)
            for inst in improved.tolist():
                witness = witnesses.get((name, inst))
                if witness is not None:
                    self._witness[(name, inst)] = witness

    # -- results ----------------------------------------------------------

    def results(self) -> dict[str, dict[str, object]]:
        """Apply the Tulu 3 rule and report per benchmark."""
        out: dict[str, dict[str, object]] = {}
        for name, bench in self.benchmarks.items():
            best = self._best.get(name)
            sizes = np.asarray(bench.instance_size, dtype=np.float64)
            contaminated: list[tuple[int, float]] = []
            if best is not None and sizes.size:
                n = min(best.size, sizes.size)
                with np.errstate(divide="ignore", invalid="ignore"):
                    frac = np.where(sizes[:n] > 0, best[:n] / sizes[:n], 0.0)
                for inst in np.nonzero(frac > TOKEN_FRACTION)[0].tolist():
                    contaminated.append((int(inst), float(frac[inst])))
            rate = len(contaminated) / bench.n_instances if bench.n_instances else 0.0
            out[name] = {
                "n_instances": bench.n_instances,
                "n_contaminated": len(contaminated),
                "rate": rate,
                "flagged": rate > INSTANCE_FRACTION,
                "witnesses": [
                    {
                        "instance": inst,
                        "coverage": frac,
                        "record": self._witness.get((name, inst)),
                    }
                    for inst, frac in sorted(contaminated, key=lambda kv: -kv[1])[:5]
                ],
            }
        return out


def load_indices(*directories: Path) -> ContaminationIndex:
    """Load every index found across the given directories.

    Takes several directories rather than one because the shipped benchmarks and
    a user's own `index-eval` output live in different places: the first inside
    the package, the second in the cache, which is the only writable location on
    a read-only install. Choosing one directory over the other would mean that
    building a private index silently switched off all ten bundled benchmarks.

    Earlier directories win on name collisions, so passing the cache before the
    package lets a user deliberately shadow a shipped index by giving theirs the
    same name.
    """
    idx = ContaminationIndex()
    seen: set[str] = set()
    for directory in directories:
        if directory is None or not directory.exists():
            continue
        for path in sorted(directory.glob("*.idx")):
            if path.stem in seen:
                continue
            try:
                bench = BenchmarkIndex.load(path)
            except Exception:
                continue
            seen.add(path.stem)
            idx.add(bench)
    return idx
