"""Benchmark contamination indices.

What ships is an **index, not the benchmark**. For each evaluation instance we
extract its word 8-grams, hash each to 64 bits, and store a mapping from hash to
the instance ids containing it. The benchmark text is not recoverable from that,
and the index for a ten-thousand-instance benchmark is a couple of megabytes.

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
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .compat import json_loads
from .textutil import hash64, normalize_for_dedup

NGRAM = 8
MAGIC = b"DTTIDX01"

#: Tulu 3 thresholds. Echoed in the report so the number is never presented
#: without the rule that produced it.
TOKEN_FRACTION = 0.50
INSTANCE_FRACTION = 0.02


def eval_ngrams(text: str, n: int = NGRAM) -> list[int]:
    words = normalize_for_dedup(text).split()
    if len(words) < n:
        return [hash64(" ".join(words))] if words else []
    return [hash64(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)]


@dataclass(slots=True)
class BenchmarkIndex:
    """Hashed 8-grams for one benchmark."""

    name: str
    n_instances: int
    #: gram hash -> instance ids containing it
    postings: dict[int, list[int]] = field(default_factory=dict)
    #: instance id -> number of distinct grams, used as the coverage denominator
    instance_size: list[int] = field(default_factory=list)
    license: str | None = None
    source: str | None = None

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
        import json  # noqa: PLC0415

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
    def load(cls, path: Path) -> "BenchmarkIndex":
        with open(path, "rb") as fh:
            if fh.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{path} is not a dropoutt contamination index")
            (head_len,) = struct.unpack("<I", fh.read(4))
            header = json_loads(fh.read(head_len))
            (n_grams,) = struct.unpack("<I", fh.read(4))
            postings: dict[int, list[int]] = {}
            for _ in range(n_grams):
                gram, count = struct.unpack("<QH", fh.read(10))
                ids = list(struct.unpack(f"<{count}I", fh.read(4 * count)))
                postings[gram] = ids
        return cls(
            name=header["name"],
            n_instances=header["n_instances"],
            postings=postings,
            instance_size=header.get("instance_size", []),
            license=header.get("license"),
            source=header.get("source"),
        )


@dataclass(slots=True)
class ContaminationIndex:
    """All loaded benchmark indices, plus the accumulator for one scan."""

    benchmarks: dict[str, BenchmarkIndex] = field(default_factory=dict)
    #: (benchmark, instance id) -> best covered-gram count seen from one record
    _best: dict[tuple[str, int], int] = field(default_factory=dict)
    #: (benchmark, instance id) -> the training record that produced it
    _witness: dict[tuple[str, int], tuple[str, str, int]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.benchmarks

    def add(self, index: BenchmarkIndex) -> None:
        self.benchmarks[index.name] = index

    def observe(self, text: str, doc_id: str, source_file: str, source_index: int) -> None:
        """Match one training record against every loaded benchmark."""
        grams = set(eval_ngrams(text))
        if not grams:
            return
        for name, bench in self.benchmarks.items():
            hits: dict[int, int] = {}
            for g in grams:
                for inst in bench.postings.get(g, ()):
                    hits[inst] = hits.get(inst, 0) + 1
            for inst, covered in hits.items():
                key = (name, inst)
                if covered > self._best.get(key, 0):
                    self._best[key] = covered
                    self._witness[key] = (doc_id, source_file, source_index)

    def results(self) -> dict[str, dict[str, object]]:
        """Apply the Tulu 3 rule and report per benchmark."""
        out: dict[str, dict[str, object]] = {}
        for name, bench in self.benchmarks.items():
            contaminated: list[tuple[int, float]] = []
            for inst in range(bench.n_instances):
                size = bench.instance_size[inst] if inst < len(bench.instance_size) else 0
                if size <= 0:
                    continue
                covered = self._best.get((name, inst), 0)
                frac = covered / size
                if frac > TOKEN_FRACTION:
                    contaminated.append((inst, frac))
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


def load_indices(directory: Path) -> ContaminationIndex:
    idx = ContaminationIndex()
    if not directory.exists():
        return idx
    for path in sorted(directory.glob("*.idx")):
        try:
            idx.add(BenchmarkIndex.load(path))
        except Exception:
            continue
    return idx
