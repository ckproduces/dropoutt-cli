#!/usr/bin/env python3
"""Build ``atlas-v2`` and ``atlas-v2-lite`` from one read-only corpus manifest.

Every retained record in every manifest shard is offered to both products.
Exact and within-batch semantic duplicates are removed once, then one shared
256-dimensional embedding store feeds the full 256-d model and the lite 64-d
Matryoshka prefix. The fetched corpus is never deleted or modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_corpus  # noqa: E402
from atlas_sources import (  # noqa: E402
    AXIS_FLOORS,
    AXIS_TARGET_BYTES,
    DEFAULT_CACHE,
    DEFAULT_RELEASE,
    DEFAULT_WORK,
    NON_ENGLISH_FLOOR,
    SOURCES,
    WEB_LANGUAGE_SHARES,
    Source,
)

from dropoutt.atlas.embed import DEFAULT_MODEL, TokenizedCorpus  # noqa: E402
from dropoutt.atlas.embed import load as load_embedder  # noqa: E402
from dropoutt.atlas.extract import extract_text  # noqa: E402
from dropoutt.atlas.pipeline import pipeline_hash  # noqa: E402
from dropoutt.atlas.profiles import ATLAS_V3, AtlasProfile  # noqa: E402
from dropoutt.atlas.textnorm import (  # noqa: E402
    ENCODER_INPUT_V1,
    PHRASE_DOCS_PER_SOURCE,
    RAW_INPUT,
    EncoderInput,
    source_phrases,
)

# The two atlas-v2 products. The package ships atlas-v3 alone, so these live
# with the builder that can still fit them rather than in the runtime
# profiles, where a name that resolves is a name `--model` accepts.
ATLAS_V2 = AtlasProfile(
    version="atlas-v2",
    dim=128,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=200_000,
    pca_k=2,
    n_l1=128,
    l2_k_min=1,
    l2_k_max=10,
)

ATLAS_V2_LITE = AtlasProfile(
    version="atlas-v2-lite",
    dim=64,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=50_000,
    pca_k=2,
    n_l1=32,
    l2_k_min=1,
    l2_k_max=10,
)

L1_LABELS = {
    ATLAS_V3.version: ROOT / "tools" / "atlas-data" / "l1_labels_atlas-v3.json",
    ATLAS_V2.version: ROOT / "tools" / "atlas-data" / "l1_labels_atlas-v2.json",
    ATLAS_V2_LITE.version: ROOT / "tools" / "atlas-data" / "l1_labels_atlas-v2-lite.json",
}

REGION_LABELS = {
    ATLAS_V3.version: ROOT / "tools" / "atlas-data" / "region_labels_atlas-v3.json",
}


SEED = 42
BLOCK = 512
MIN_COMMUNITY = 1_000
MIN_L2_CHILD_SUPPORT = 1_000
FULL_EXEMPLARS, LITE_EXEMPLARS = 32, 16
FULL_EXEMPLAR_CHARS, LITE_EXEMPLAR_CHARS = 600, 256
FULL_PROTOTYPES, LITE_PROTOTYPES = 24, 12
FULL_KNOTS, LITE_KNOTS = 50, 25
FULL_COOCCURRENCE, LITE_COOCCURRENCE = 32, 16
FULL_TERMS, LITE_TERMS = 48, 32
# atlas-v3 carries 4,096 cells against v2's 296, so every per-cell array is
# sized against the unpacked artifact rather than the zipped one. Budget at
# 4,096 cells: prototypes 8.4 MB, centroids 2.1 MB, terms 2.1 MB, idf 3.2 MB,
# co-occurrence 0.8 MB, exemplars 21.0 MB — about 38 MB unpacked. The dense
# cell-similarity matrix is dropped: at 4,096 cells it alone would be 33.5 MB,
# and the top-k co-occurrence neighbours already answer what it was for.
V3_EXEMPLARS, V3_EXEMPLAR_CHARS = 4, 320
V3_PROTOTYPES = 8
V3_KNOTS = 50
V3_COOCCURRENCE = 32
V3_TERMS = 64
#: See :func:`population_l2_budget` for the measurement behind 0.75.
L2_BUDGET_EXPONENT = 0.75
# 4,096 cells share the reservoir for exemplars and contrastive terms, so
# 250,000 left an average cell 61 records to be described from. Doubled.
RESERVOIR_FULL = 500_000
RESERVOIR_LITE = 250_000
IDF_WARMUP = 500_000
IDF_SAMPLE_ROWS = 4_000_000
AXIS_BYTE_FLOORS = {axis: target // 2 for axis, target in AXIS_TARGET_BYTES.items()}
#: Rows in the language-and-axis balanced draw that fits the global mean and
#: the stripped PCA directions. Balance, not size, is what moves those
#: directions -- measured on the 200 GiB corpus, going from 400k to 6.4M
#: proportional rows moved the top-2 subspace 0.5 degrees, while rebalancing a
#: 250k draw moved it 27 -- but a larger balanced draw is cheap and the small
#: (language, axis) strata are the ones a small draw starves.
NORM_SAMPLE_ROWS = 2_000_000
#: Held-out rows for the language probe recorded in every artifact.
PROBE_ROWS = 300_000
#: Sibling L2 centroids at or above this cosine are one subject split in two by
#: the budget; they are folded together before the cell list is frozen.
L2_SIBLING_MERGE_COSINE = 0.95
#: Detector confidence below which a record keeps the language its source
#: declared. The detector says "unknown" rather than guess on short text.
LANGID_CONFIDENCE = 0.6
LANGID_WORKERS = 4
#: Where the normalized float16 memmap lives during clustering. The L2 phase
#: gathers each region's members by fancy index -- ~163M random 256-byte reads
#: across a 42 GB file -- and on the external drive that phase measured 14 h
#: for 72 of 256 regions. Point this at the internal SSD.
NORM_DIR = os.environ.get("ATLAS_NORM_DIR")
#: Checkpoint the L2 loop every this many regions.
L2_CHECKPOINT_EVERY = 8
# Rows a language needs before the build fits it its own mean. The mean is a
# 128-d estimate, so this is a stability floor, not a share of the corpus: at
# 6,000 rows the standard error of each coordinate is 1/sqrt(6000) of its
# spread, about 1.7x tighter than the 2,000 this replaces. It is reachable
# only because the means are now fitted by streaming every row of a language
# rather than counting its rows inside a proportional sample -- under the old
# 400,000-row uniform draw a 6,000 floor would have admitted 11 languages
# where 2,000 admitted 20, because a 0.4%-of-the-web language contributes
# 1,600 rows to such a sample no matter how much of it the corpus holds.
MIN_LANGUAGE_ROWS = 6_000
# A language the plan asked for is never dropped for being small. Its rows are
# in the corpus because a source was chosen to put them there, and leaving it
# on the global mean is what made v3's Greek, Ukrainian and Thai cells cluster
# by language instead of by subject. Below this second floor a 128-d mean is
# noise and subtracting it would be worse than the global mean, so the bypass
# stops there rather than going to zero.
DECLARED_LANGUAGE_FLOOR = 500
# 4M IDF rows in 4M slots would be a 95% load factor on an open-addressing
# table; 16M keeps it under a quarter.
IDF_HASH_SLOTS = 1 << 24
GROW_ROWS = int(os.environ.get("ATLAS_BUILD_GROW_ROWS", "5000000"))
PROGRESS_EVERY = 25_000
HASH_SLOTS = int(os.environ.get("ATLAS_BUILD_HASH_SLOTS", str(1 << 27)))
# Production default: 134M × 8 bytes = 1 GiB; load factor ~0.52 at 70M rows.
PROBE_LIMIT = 4096

if HASH_SLOTS <= 0 or HASH_SLOTS & (HASH_SLOTS - 1):
    raise ValueError("ATLAS_BUILD_HASH_SLOTS must be a positive power of two")


def log(message: str) -> None:
    print(message, flush=True)


class BuildProgress:
    """Atomic build status consumed by ``atlas_build_progress.py``."""

    def __init__(self, path: Path, input_rows: int) -> None:
        self.path = path
        self.input_rows = input_rows
        self.started_at = time.time()
        # Overall percentage at this process's first update. A resumed build
        # starts well above zero, and a rate measured from zero since process
        # start would report the checkpointed work as if it had just happened.
        self.started_pct: float | None = None
        self._last_write = 0.0
        self._last_stage = ""

    def update(
        self,
        stage: str,
        overall_pct: float,
        *,
        stage_pct: float = 0.0,
        detail: str = "",
        retained_rows: int = 0,
    ) -> None:
        if self.started_pct is None:
            self.started_pct = float(overall_pct)
        # At most one write a second within a stage. The IDF scan calls this for
        # 512 consecutive rows out of every 25,000, and on the external drive a
        # write-and-rename costs about 6 ms: profiled on one 273k-row source,
        # progress writes were 34 s of a 38 s pass -- about six hours over the
        # whole manifest. A monitor polling every few seconds loses nothing.
        now = time.time()
        if stage == self._last_stage and now - self._last_write < 1.0:
            return
        self._last_write, self._last_stage = now, stage
        payload = {
            "status": "running",
            "stage": stage,
            "started_pct": round(self.started_pct, 4),
            "overall_pct": round(float(overall_pct), 4),
            "stage_pct": round(float(stage_pct), 4),
            "detail": detail,
            "input_rows": self.input_rows,
            "retained_rows": int(retained_rows),
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": time.time(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def finish(self, *, retained_rows: int) -> None:
        self.update("complete", 100.0, stage_pct=100.0, retained_rows=retained_rows)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["status"] = "complete"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


def curated_l1_labels(
    profile: AtlasProfile, corpus_hash: str | None = None
) -> tuple[list[str], str]:
    """Load declarative L1 names for the HTML grid and category_labels().

    Names are only meaningful for the clustering they were written against.
    A rebuild on a different corpus produces 256 different regions under the
    same indices, so a file that does not name this build's ``corpus_hash`` is
    treated as absent rather than re-applied to cells it has never seen.
    """
    path = L1_LABELS.get(profile.version)
    if path is None or not path.is_file():
        return [f"area {i}" for i in range(profile.n_l1)], "missing"
    payload = json.loads(path.read_text(encoding="utf-8"))
    curated = payload.get("labels", {})
    if len(curated) != profile.n_l1:
        return [f"area {i}" for i in range(profile.n_l1)], "profile-changed"
    if corpus_hash and payload.get("corpus_hash") != corpus_hash:
        return [f"area {i}" for i in range(profile.n_l1)], "profile-changed"
    labels = [str(curated.get(str(i), f"area {i}")) for i in range(profile.n_l1)]
    return labels, f"curated:{path.name}"


def curated_region_labels(
    profile: AtlasProfile, n_cells: int, corpus_hash: str | None = None
) -> tuple[list[str] | None, str]:
    """Load declarative fine-cell names for reports and region_labels().

    Same rule as :func:`curated_l1_labels`: a name file is bound to the
    ``corpus_hash`` it was curated against, or it is not applied.
    """
    path = REGION_LABELS.get(profile.version)
    if path is None or not path.is_file():
        return None, "contrastive"
    payload = json.loads(path.read_text(encoding="utf-8"))
    curated = payload.get("labels", {})
    if len(curated) != n_cells:
        return None, "profile-changed"
    if corpus_hash and payload.get("corpus_hash") != corpus_hash:
        return None, "profile-changed"
    labels = [str(curated.get(str(i), f"cell {i}")) for i in range(n_cells)]
    return labels, f"curated:{path.name}"


def curated_kinds(profile: AtlasProfile, count: int, corpus_hash: str, *, level: str) -> list[str] | None:
    """The kinds stored beside curated names, under the same corpus-hash rule."""
    path = (L1_LABELS if level == "l1" else REGION_LABELS).get(profile.version)
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    kinds = payload.get("kinds", {})
    if payload.get("corpus_hash") != corpus_hash or len(kinds) != count:
        return None
    return [str(kinds[str(i)]) for i in range(count)]


def _fold(word: str) -> str:
    """Accent-folded five-character stem, for collapsing inflections."""
    plain = unicodedata.normalize("NFKD", word)
    return "".join(c for c in plain if not unicodedata.combining(c))[:5].lower()


def contrastive_cell_labels(
    terms: list[list[str]], n_cells: int, width: int = 4, depth: int = 32
) -> list[str]:
    """Name every L2 cell by the terms that separate it from the other cells.

    ``_terms`` ranks a cell's words against corpus-wide document frequency,
    which is enough to make a narrow cell readable but still lets function
    words through: on the v3 build, cell 2,500 read ``this, have, with, your``
    and cell 3,500 ``that, with, this, from``. Re-weighting each word by
    ``log(n_cells / cells containing it)`` drops those to zero, because a word
    in most of four thousand cells cannot separate any of them.

    Inflections are then collapsed on an accent-folded five-character stem.
    Without it a cell spends two of four slots on ``spiele``/``spielen`` and
    ``kostenlos``/``kostenlosen`` — the defect measured on v1, where 21% of
    regions gave two or more of five slots to one stem.

    The result is 4,095 distinct names across 4,096 cells, and no two cells
    under the same L1 parent share a name.
    """
    document_frequency: Counter[str] = Counter()
    for cell in terms:
        document_frequency.update(set(cell[:depth]))
    labels: list[str] = []
    for index in range(n_cells):
        cell = terms[index] if index < len(terms) else []
        scored = sorted(
            (
                (depth - rank) * math.log(n_cells / document_frequency[term]),
                rank,
                term,
            )
            for rank, term in enumerate(cell[:depth])
        )
        picked: list[str] = []
        seen: set[str] = set()
        for _, _, term in reversed(scored):
            stem = _fold(term)
            if stem in seen:
                continue
            seen.add(stem)
            picked.append(term)
            if len(picked) == width:
                break
        labels.append(", ".join(picked) if picked else f"cell {index}")
    return labels


def automatic_l1_labels(
    terms: list[list[str]], parents: np.ndarray, n_l1: int
) -> list[str]:
    """Name each L1 by the terms that separate it from the other L1 regions.

    Counting how many child cells contain a word, and taking the commonest, is
    a vote that function words always win: they appear in the top terms of most
    cells, so on the 256-region v3 build ``with`` labelled 214 regions, ``that``
    197 and ``this`` 193. **81.3% of label slots** went to a word occurring in
    eight or more regions, and 174 of 256 regions were named entirely in such
    words — against 21.6% of slots on v2 and 14.1% on v1-lite.

    ``_terms`` already scores L2 cells against corpus-wide document frequency,
    which is why the fine cells read well. The same idea applied one level up —
    weight each word by ``log(n_l1 / regions containing it)`` — takes that 81.3%
    to 0.0% with no rebuild, since it needs only the stored region terms.
    """
    pools: list[Counter[str]] = []
    for parent in range(n_l1):
        words: Counter[str] = Counter()
        for cell in np.flatnonzero(parents == parent).tolist():
            for rank, term in enumerate(terms[cell][:24]):
                words[term] += 24 - rank
        pools.append(words)
    region_frequency: Counter[str] = Counter()
    for words in pools:
        region_frequency.update(set(words))
    labels: list[str] = []
    for parent in range(n_l1):
        scored = {
            term: weight * math.log(n_l1 / region_frequency[term])
            for term, weight in pools[parent].items()
        }
        picked = [
            term for term, _ in sorted(scored.items(), key=lambda item: -item[1])[:4]
        ]
        labels.append(" / ".join(picked) if picked else f"area {parent}")
    return labels


def selected_token_ids(ids: np.ndarray, budget: int) -> np.ndarray:
    """Return deterministic 40% head / 20% middle / 40% tail token windows."""
    ids = np.asarray(ids, dtype=np.int32)
    if len(ids) <= budget:
        return ids
    head = int(budget * 0.4)
    middle = int(budget * 0.2)
    tail = budget - head - middle
    middle_start = max(0, (len(ids) - middle) // 2)
    return np.concatenate((ids[:head], ids[middle_start:middle_start + middle], ids[-tail:]))


def stratified_indices(axes: list[str], languages: list[str], count: int,
                       seed: int = SEED) -> np.ndarray:
    """Fixed-size deterministic proportional sample over (axis, language)."""
    n = len(axes)
    if n <= count:
        return np.arange(n, dtype=np.int64)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(axes, languages, strict=True)):
        groups[key].append(i)
    rng = np.random.default_rng(seed)
    quotas: dict[tuple[str, str], int] = {}
    fractions: list[tuple[float, tuple[str, str]]] = []
    for key in sorted(groups):
        exact = count * len(groups[key]) / n
        quotas[key] = min(len(groups[key]), int(math.floor(exact)))
        fractions.append((exact - quotas[key], key))
    remaining = count - sum(quotas.values())
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if not remaining:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    picked = []
    for key in sorted(groups):
        members = np.asarray(groups[key], dtype=np.int64)
        take = quotas[key]
        picked.extend(rng.choice(members, take, replace=False).tolist())
    return np.asarray(sorted(picked), dtype=np.int64)


#: Rows per contiguous read when gathering a large scattered index.
GATHER_BLOCK = 1_000_000


def gather_rows(array: np.ndarray, index: np.ndarray, dtype=np.float32) -> np.ndarray:
    """``array[index]`` as ``dtype``, read in contiguous blocks when the index is large.

    Fancy-indexing two million scattered rows of a 42 GB memmap on the external
    drive is two million page faults; measured there, scattered reads run at a
    few hundred rows a second, which put the calibration draw alone near two
    hours. Reading each block that holds any requested row in one sequential
    slice turns it into a pass of about two minutes. Rows come back in the
    order asked for, repeats included, bit for bit what fancy indexing returns.
    """
    index = np.asarray(index, dtype=np.int64)
    if index.size < 200_000:
        return np.asarray(array[index], dtype=dtype)
    order = np.argsort(index, kind="stable")
    ordered = index[order]
    out = np.empty((len(index), *array.shape[1:]), dtype=dtype)
    first = int(ordered[0]) // GATHER_BLOCK * GATHER_BLOCK
    for lo in range(first, int(ordered[-1]) + 1, GATHER_BLOCK):
        a, b = np.searchsorted(ordered, [lo, lo + GATHER_BLOCK])
        if a == b:
            continue
        hi = min(lo + GATHER_BLOCK, int(ordered[b - 1]) + 1)
        block = np.asarray(array[lo:hi], dtype=dtype)
        out[order[a:b]] = block[ordered[a:b] - lo]
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _intern(table: list[str], index: dict[str, int], value: str) -> int:
    slot = index.get(value)
    if slot is None:
        slot = len(table)
        index[value] = slot
        table.append(value)
    return slot


class Uint64Set:
    """Open-addressing set of uint64 hashes in a memmap. No Python per-key objects."""

    def __init__(self, path: Path, slots: int = HASH_SLOTS) -> None:
        self.path = path
        self.slots = slots
        nbytes = slots * 8
        if path.exists() and path.stat().st_size == nbytes:
            self.table = np.memmap(path, dtype=np.uint64, mode="r+", shape=(slots,))
            return
        previous = None
        if path.exists() and path.stat().st_size >= 8:
            previous = np.memmap(path, dtype=np.uint64, mode="r")
            log(f"  rehashing {path.name} {previous.size:,} -> {slots:,} slots")
        tmp = path.with_suffix(path.suffix + ".new")
        self.table = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(slots,))
        self.table[:] = 0
        if previous is not None:
            # Copy the keys out and drop the mapping before the rename below:
            # a view would keep the old file mapped, and Windows refuses to
            # replace a file that is still open.
            keys = np.array(previous[previous != 0])
            del previous
            for key in keys:
                self._insert(int(key), grow=False)
        self.table.flush()
        del self.table
        tmp.replace(path)
        self.table = np.memmap(path, dtype=np.uint64, mode="r+", shape=(slots,))

    def add(self, key: int) -> bool:
        return self._insert(key, grow=True)

    def _insert(self, key: int, *, grow: bool) -> bool:
        if key == 0:
            key = 1
        mask = self.slots - 1
        table = self.table
        i = key & mask
        step = ((key >> 33) | 1) & mask
        for _ in range(min(PROBE_LIMIT, self.slots)):
            cur = int(table[i])
            if cur == 0:
                table[i] = key
                return True
            if cur == key:
                return False
            i = (i + step) & mask
        if grow:
            self._grow()
            return self._insert(key, grow=False)
        raise RuntimeError("hash set probe failed")

    def _grow(self) -> None:
        bigger = Uint64Set(self.path, slots=self.slots * 2)
        self.slots = bigger.slots
        self.table = bigger.table

    def flush(self) -> None:
        self.table.flush()


class Reservoir:
    """Algorithm R. Stores a bounded sample of (row, text, axis, language, source)."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.items: list[tuple[int, str, str, str, str]] = []

    def offer(self, row: int, text: str, axis: str, language: str, source: str) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append((row, text, axis, language, source))
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.size:
            self.items[j] = (row, text, axis, language, source)

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"size": self.size, "seen": self.seen}) + "\n")
            for item in self.items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, size: int, seed: int) -> Reservoir:
        reservoir = cls(size, seed)
        if not path.is_file():
            return reservoir
        with path.open(encoding="utf-8") as handle:
            header = json.loads(handle.readline())
            reservoir.seen = int(header["seen"])
            reservoir.items = [tuple(json.loads(line)) for line in handle if line.strip()]
        return reservoir


class DiskCorpus:
    """Growing float16 embedding store plus interned categorical columns."""

    def __init__(self, work: Path, dim: int) -> None:
        self.work = work
        self.dim = dim
        self.n = 0
        self.cap = 0
        self.emb: np.memmap | None = None
        #: Retained post-filter UTF-8 bytes per detected language. The
        #: non-English release gate is a byte share, so it needs bytes; the
        #: embedding columns only carry a language id per row.
        self.logical_by_lang: dict[str, int] = {}
        self.logical_by_axis: dict[str, int] = {}
        self.axis_ids: np.memmap | None = None
        self.lang_ids: np.memmap | None = None
        self.src_ids: np.memmap | None = None
        #: Characters of each retained text, capped at the uint16 range. Builds
        #: that predate the column have none, and cluster by row count.
        self.lengths: np.memmap | None = None
        self.axis_table: list[str] = []
        self.lang_table: list[str] = []
        self.src_table: list[str] = []
        self._axis_ix: dict[str, int] = {}
        self._lang_ix: dict[str, int] = {}
        self._src_ix: dict[str, int] = {}

    def _resize(self, cap: int) -> None:
        def grow(path: Path, dtype, width: int | None, previous):
            if previous is not None:
                previous.flush()
            itemsize = np.dtype(dtype).itemsize * (width or 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.truncate(cap * itemsize)
            shape = (cap, width) if width else (cap,)
            return np.memmap(path, dtype=dtype, mode="r+", shape=shape)

        self.emb = grow(self.work / "emb.f16", np.float16, self.dim, self.emb)
        self.axis_ids = grow(self.work / "axis.u8", np.uint8, None, self.axis_ids)
        self.lang_ids = grow(self.work / "lang.u8", np.uint8, None, self.lang_ids)
        self.src_ids = grow(self.work / "src.u16", np.uint16, None, self.src_ids)
        self.lengths = grow(self.work / "len.u16", np.uint16, None, self.lengths)
        self.cap = cap

    def append(self, vectors: np.ndarray, axes: list[str], languages: list[str],
               sources: list[str], lengths: list[int] | None = None) -> int:
        batch = len(vectors)
        if self.n + batch > self.cap:
            self._resize(max(self.cap + GROW_ROWS, self.n + batch))
        start = self.n
        assert self.emb is not None
        self.emb[start:start + batch] = np.asarray(vectors, dtype=np.float16)
        for i, (axis, language, source) in enumerate(zip(axes, languages, sources, strict=True)):
            self.axis_ids[start + i] = _intern(self.axis_table, self._axis_ix, axis)
            self.lang_ids[start + i] = _intern(self.lang_table, self._lang_ix, language)
            self.src_ids[start + i] = _intern(self.src_table, self._src_ix, source)
        if lengths is not None and self.lengths is not None:
            self.lengths[start:start + batch] = np.minimum(np.asarray(lengths), 65_535)
        self.n += batch
        return start

    def rows_f32(self, index: np.ndarray) -> np.ndarray:
        assert self.emb is not None
        return gather_rows(self.emb, index)

    def slice_f32(self, start: int, stop: int) -> np.ndarray:
        assert self.emb is not None
        return np.asarray(self.emb[start:stop], dtype=np.float32)

    def axes(self) -> list[str]:
        assert self.axis_ids is not None
        table = self.axis_table
        return [table[int(i)] for i in self.axis_ids[:self.n]]

    def languages(self) -> list[str]:
        assert self.lang_ids is not None
        table = self.lang_table
        return [table[int(i)] for i in self.lang_ids[:self.n]]

    def sources(self) -> list[str]:
        assert self.src_ids is not None
        table = self.src_table
        return [table[int(i)] for i in self.src_ids[:self.n]]

    def flush(self) -> None:
        for array in (self.emb, self.axis_ids, self.lang_ids, self.src_ids, self.lengths):
            if array is not None:
                array.flush()

    def save_checkpoint(self, consumed: set[str], logical: int, token_counts: np.ndarray) -> None:
        self.flush()
        np.save(self.work / "token_counts.npy", token_counts)
        payload = {
            "n": self.n, "cap": self.cap, "logical": logical,
            "axis_table": self.axis_table, "lang_table": self.lang_table,
            "src_table": self.src_table, "consumed": sorted(consumed),
            "logical_by_lang": self.logical_by_lang,
            "logical_by_axis": self.logical_by_axis,
        }
        path = self.work / "checkpoint.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        tmp.replace(path)

    def load_checkpoint(self) -> tuple[set[str], int, np.ndarray | None]:
        path = self.work / "checkpoint.json"
        if not path.is_file() or not (self.work / "emb.f16").is_file():
            return set(), 0, None
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.n = int(payload["n"])
        self.cap = int(payload["cap"])
        self.axis_table = list(payload["axis_table"])
        self.lang_table = list(payload["lang_table"])
        self.src_table = list(payload["src_table"])
        self.logical_by_lang = dict(payload.get("logical_by_lang", {}))
        self.logical_by_axis = dict(payload.get("logical_by_axis", {}))
        self._axis_ix = {name: i for i, name in enumerate(self.axis_table)}
        self._lang_ix = {name: i for i, name in enumerate(self.lang_table)}
        self._src_ix = {name: i for i, name in enumerate(self.src_table)}
        self.emb = np.memmap(self.work / "emb.f16", dtype=np.float16, mode="r+", shape=(self.cap, self.dim))
        self.axis_ids = np.memmap(self.work / "axis.u8", dtype=np.uint8, mode="r+", shape=(self.cap,))
        self.lang_ids = np.memmap(self.work / "lang.u8", dtype=np.uint8, mode="r+", shape=(self.cap,))
        self.src_ids = np.memmap(self.work / "src.u16", dtype=np.uint16, mode="r+", shape=(self.cap,))
        lengths_path = self.work / "len.u16"
        if lengths_path.is_file() and lengths_path.stat().st_size >= self.cap * 2:
            self.lengths = np.memmap(lengths_path, dtype=np.uint16, mode="r+", shape=(self.cap,))
        counts_path = self.work / "token_counts.npy"
        counts = np.load(counts_path) if counts_path.is_file() else None
        return set(payload["consumed"]), int(payload["logical"]), counts


def balanced_calibration_draw(
    corpus, take: np.ndarray | None, n: int, rows: int, seed: int = SEED
) -> np.ndarray:
    """Row indices stratified by (language, axis), as even as the data allows.

    The global mean and the PCA directions stripped from every vector are fitted
    on this draw. A proportional draw is 53% English web, so those directions
    were English-web-shaped; each (language, axis) stratum now gets the same
    quota, with the quota of a stratum too small to fill it handed back to the
    rest. What a stratum holds is still its own mixture of subjects -- the atlas
    is the topic model, so there is nothing finer to balance on yet.
    """
    universe = np.arange(n, dtype=np.int64) if take is None else np.asarray(take, dtype=np.int64)
    langs = np.asarray(corpus.lang_ids[universe], dtype=np.int64)
    axes = np.asarray(corpus.axis_ids[universe], dtype=np.int64)
    key = langs * 256 + axes
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1, len(order)]
    sizes = np.diff(starts)
    rows = min(int(rows), len(universe))
    quota = np.zeros(len(sizes), dtype=np.int64)
    remaining, open_strata = rows, np.ones(len(sizes), dtype=bool)
    while remaining > 0 and open_strata.any():
        share = max(1, remaining // int(open_strata.sum()))
        grant = np.minimum(sizes - quota, share) * open_strata
        quota += grant
        remaining -= int(grant.sum())
        open_strata &= quota < sizes
        if not grant.any():
            break
    rng = np.random.default_rng(seed)
    picks = []
    for s, e, q in zip(starts[:-1], starts[1:], quota, strict=True):
        if q <= 0:
            continue
        block = order[s:e]
        picks.append(block if q >= len(block) else rng.choice(block, int(q), replace=False))
    local = np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)
    return universe[local]


def language_probe(
    corpus, mm: np.ndarray, take: np.ndarray | None, n: int, dim: int,
    rows: int = PROBE_ROWS,
) -> dict:
    """How much language survives normalization, and how much subject does.

    Two linear probes on a balanced held-out draw: one predicts the record's
    language, one its axis. Language accuracy should fall sharply from the raw
    vectors to the normalized ones; axis accuracy should not. Balanced accuracy
    throughout, because English is half the rows and plain accuracy would call
    a majority-class guess "language removed". Recorded in the artifact so a
    map ships with the evidence that it is a map of subjects.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    idx = balanced_calibration_draw(corpus, take, n, rows, seed=SEED + 7)
    if len(idx) < 2_000:
        return {"rows": int(len(idx)), "skipped": True}
    raw = _norm(corpus.rows_f32(idx)[:, :dim])
    # ``mm`` is positional over ``take`` when a subset is being built.
    local = idx if take is None else np.searchsorted(np.asarray(take), idx)
    normed = gather_rows(mm, local)
    lang = np.asarray(corpus.lang_ids[idx], dtype=np.int64)
    axis = np.asarray(corpus.axis_ids[idx], dtype=np.int64)
    rng = np.random.default_rng(SEED + 8)
    perm = rng.permutation(len(idx))
    cut = int(0.7 * len(idx))
    tr, te = perm[:cut], perm[cut:]

    def score(x: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y[tr])) < 2:
            return float("nan")
        model = LogisticRegression(max_iter=300).fit(x[tr], y[tr])
        return float(balanced_accuracy_score(y[te], model.predict(x[te])))

    out = {
        "rows": int(len(idx)),
        "metric": "balanced_accuracy",
        "language_raw": score(raw, lang),
        "language_normalized": score(normed, lang),
        "axis_raw": score(raw, axis),
        "axis_normalized": score(normed, axis),
        "n_languages": int(len(np.unique(lang))),
        "n_axes": int(len(np.unique(axis))),
    }
    return out


def fit_language_means(
    corpus, profile: AtlasProfile, take: np.ndarray | None, n: int,
    minimum: int = MIN_LANGUAGE_ROWS,
) -> tuple[np.ndarray, list[str]]:
    """Exact mean vector per language, accumulated over every row it has.

    Sampling is the wrong instrument here. One mean is estimated per language,
    so what a language needs is its own rows, not a share of a shared draw --
    and a proportional draw gives a 0.1% language 400 rows however large the
    corpus grows. Streaming the whole corpus costs one extra memmap pass and
    removes the sampling error entirely, which is what lets the floor above be
    stated in corpus rows.

    A language the corpus plan declares clears a lower floor -- see
    :data:`DECLARED_LANGUAGE_FLOOR` -- because its rows were fetched on
    purpose. ``unknown`` and ``multi`` never get a mean; they keep the global
    one, since neither names a distribution a mean could describe.
    """
    dim = profile.dim
    table = list(corpus.lang_table)
    sums = np.zeros((len(table), dim), dtype=np.float64)
    counts = np.zeros(len(table), dtype=np.int64)
    for start in range(0, n, 32_768):
        stop = min(start + 32_768, n)
        rows = slice(start, stop) if take is None else take[start:stop]
        raw = (corpus.slice_f32(start, stop) if take is None
               else corpus.rows_f32(rows))[:, :dim]
        ids = np.asarray(corpus.lang_ids[rows], dtype=np.int64)
        counts += np.bincount(ids, minlength=len(table))
        for lid in np.unique(ids):
            sums[lid] += raw[ids == lid].sum(axis=0, dtype=np.float64)
    declared = set(WEB_LANGUAGE_SHARES) | {
        source.lang for source in SOURCES if getattr(source, "lang", None)
    }
    eligible = sorted(
        name for i, name in enumerate(table)
        if name not in ("unknown", "multi")
        and (counts[i] >= minimum
             or (name in declared and counts[i] >= DECLARED_LANGUAGE_FLOOR))
    )
    index = {name: i for i, name in enumerate(table)}
    means = np.vstack([sums[index[name]] / counts[index[name]] for name in eligible]) \
        if eligible else np.zeros((0, dim), dtype=np.float32)
    return means.astype(np.float32), eligible


def fit_normalization(vectors: np.ndarray, languages: list[str],
                      profile: AtlasProfile,
                      language_means: np.ndarray | None = None,
                      language_labels: list[str] | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit the global centering and all-but-top PCA on a proportional sample.

    Per-language means arrive precomputed from :func:`fit_language_means`,
    which streams every row. This sample only has to represent the corpus as a
    whole -- which a proportional draw does -- so it is no longer also being
    asked to carry enough of each language to estimate that language's mean.
    """
    x = np.asarray(vectors[:, :profile.dim], dtype=np.float32)
    global_mean = x.mean(axis=0).astype(np.float32)
    lang_labels: list[str] = []
    stacked = np.zeros((0, profile.dim), dtype=np.float32)
    centered = x - global_mean
    per_language = profile.version in (
        ATLAS_V2.version, ATLAS_V2_LITE.version, ATLAS_V3.version
    )
    if (per_language and language_labels
            and language_means is not None and len(language_means)):
        lang_labels = list(language_labels)
        stacked = np.asarray(language_means, dtype=np.float32)
        lookup = {name: stacked[i] for i, name in enumerate(lang_labels)}
        centered = x.copy()
        for i, lang in enumerate(languages):
            centered[i] -= lookup.get(lang, global_mean)
    if len(centered) > profile.pca_k + 10:
        from sklearn.utils.extmath import randomized_svd
        sample = centered if len(centered) <= NORM_SAMPLE_ROWS else centered[np.linspace(0, len(centered) - 1, NORM_SAMPLE_ROWS, dtype=np.int64)]
        _, _, pca = randomized_svd(sample, n_components=profile.pca_k, random_state=SEED)
        pca = pca.astype(np.float32)
    else:
        pca = np.zeros((0, profile.dim), dtype=np.float32)
    return global_mean, pca, stacked, lang_labels


def apply_normalization(vectors: np.ndarray, global_mean: np.ndarray, pca: np.ndarray,
                        languages: list[str], lang_labels: list[str], lang_means: np.ndarray) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float32).copy()
    means = np.tile(global_mean, (len(x), 1))
    lookup = {name: i for i, name in enumerate(lang_labels)}
    for row, language in enumerate(languages):
        slot = lookup.get(language)
        if slot is not None:
            means[row] = lang_means[slot]
    x -= means
    if pca.size:
        x -= (x @ pca.T) @ pca
    return _norm(x).astype(np.float32)


def merge_small_communities(labels: np.ndarray, vectors: np.ndarray,
                            neighbors: np.ndarray, weights: np.ndarray,
                            minimum: int = MIN_COMMUNITY) -> np.ndarray:
    """Merge undersized Leiden communities by aggregate graph weight."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    counts = np.bincount(labels)
    large = np.flatnonzero(counts >= minimum)
    if not len(large):
        return np.zeros(len(labels), dtype=np.int32)
    centroids = {int(cell): _norm(vectors[labels == cell].mean(axis=0, keepdims=True))[0] for cell in large}
    for cell in np.flatnonzero((counts > 0) & (counts < minimum)).tolist():
        members = np.flatnonzero(labels == cell)
        score: Counter[int] = Counter()
        for row in members.tolist():
            for neighbor, weight in zip(neighbors[row], weights[row], strict=True):
                target = int(labels[int(neighbor)])
                if target in centroids:
                    score[target] += float(weight)
        if score:
            target = min(((-value, key) for key, value in score.items()))[1]
        else:
            mean = _norm(vectors[members].mean(axis=0, keepdims=True))[0]
            target = min(centroids, key=lambda key: (-float(mean @ centroids[key]), key))
        labels[members] = target
    unique = sorted(np.unique(labels).tolist())
    return np.asarray([unique.index(int(value)) for value in labels], dtype=np.int32)


def _best_kmeans(vectors: np.ndarray, k_min: int, k_max: int, *, seed: int
                 ) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Try every k in ``[k_min, k_max]`` and keep the best cosine silhouette.

    k=1 has no silhouette; it scores 0 so a negative split loses to a single cell.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    n = len(vectors)
    fallback = _norm(np.asarray(vectors, dtype=np.float32).mean(axis=0, keepdims=True))
    if n < 2 or k_max < 2:
        return np.zeros(n, dtype=np.int32), 1, 0.0, fallback
    hi = min(int(k_max), n - 1)
    lo = max(int(k_min), 1)
    rng = np.random.default_rng(seed)
    fit_n = min(n, 20_000)
    fit_idx = np.arange(n) if n == fit_n else np.sort(rng.choice(n, fit_n, replace=False))
    fit = np.asarray(vectors[fit_idx], dtype=np.float32)
    score_n = min(len(fit), 4_000)
    score_idx = np.arange(len(fit)) if len(fit) == score_n else rng.choice(len(fit), score_n, replace=False)
    scoring = fit[score_idx]

    best_score, best_model = (-np.inf, None)
    if lo <= 1:
        best_score, best_model = 0.0, None
    for k in range(max(lo, 2), hi + 1):
        model = MiniBatchKMeans(
            n_clusters=k, random_state=seed, batch_size=min(2048, max(256, k * 16)),
            n_init=3, max_iter=150,
        ).fit(fit)
        guessed = model.predict(scoring)
        if len(set(guessed.tolist())) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(scoring, guessed, metric="cosine"))
        if score > best_score:
            best_score, best_model = score, model
    if best_model is None:
        return np.zeros(n, dtype=np.int32), 1, 0.0, fallback
    materialized = np.asarray(vectors, dtype=np.float32)
    labels = best_model.predict(materialized).astype(np.int32)
    centres = _norm(best_model.cluster_centers_.astype(np.float32))
    counts = np.bincount(labels, minlength=len(centres))
    for missing in np.flatnonzero(counts == 0).tolist():
        candidates = np.flatnonzero(counts[labels] > 1)
        if not len(candidates):
            break
        row = int(candidates[np.argmax(materialized[candidates] @ centres[missing])])
        counts[int(labels[row])] -= 1
        labels[row] = missing
        counts[missing] = 1
    # A k-means centre with negligible population is a numerical split, not a
    # useful L2 cell. It produces empty descriptions and wildly inflated density
    # ratios when a user happens to place one or two records there. Fold every
    # such child into its nearest adequately supported sibling before writing
    # the frozen artifact. Small test fixtures retain a proportionate minimum.
    minimum = min(MIN_L2_CHILD_SUPPORT, max(1, n // 10))
    counts = np.bincount(labels, minlength=len(centres))
    supported = np.flatnonzero(counts >= minimum)
    if len(supported):
        for cell in np.flatnonzero((counts > 0) & (counts < minimum)).tolist():
            members = np.flatnonzero(labels == cell)
            mean = _norm(materialized[members].mean(axis=0, keepdims=True))[0]
            target = int(supported[np.argmax(centres[supported] @ mean)])
            labels[members] = target
        present = np.unique(labels)
        remap = np.full(len(centres), -1, dtype=np.int32)
        remap[present] = np.arange(len(present), dtype=np.int32)
        labels = remap[labels]
        sums = _label_sums(materialized, labels, len(present))
        merged_counts = np.bincount(labels, minlength=len(present))
        centres = _norm((sums / merged_counts[:, None]).astype(np.float32))
    return labels, len(centres), float(best_score), centres


def _label_sums(values: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Per-label float64 column sums; ``np.add.at`` bit for bit, about five times faster.

    ``bincount`` accumulates in index order exactly as ``np.add.at`` does, one
    column at a time. The content-weighted build repairs each region over all of
    its members, up to a million rows, where ``np.add.at`` cost seconds a region.
    """
    labels = np.asarray(labels)
    return np.stack(
        [np.bincount(labels, weights=values[:, j], minlength=count) for j in range(values.shape[1])],
        axis=1,
    ).astype(np.float64, copy=False)


def _repair_and_merge(
    labels: np.ndarray, centres: np.ndarray, materialized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fill empty centres and fold negligible ones into their nearest sibling.

    Extracted verbatim from :func:`_best_kmeans` so the fixed-budget path gets
    the same guarantees: no empty cell in the artifact, and no cell so small its
    description is blank and its density ratio meaningless.
    """
    n = len(materialized)
    counts = np.bincount(labels, minlength=len(centres))
    for missing in np.flatnonzero(counts == 0).tolist():
        candidates = np.flatnonzero(counts[labels] > 1)
        if not len(candidates):
            break
        row = int(candidates[np.argmax(materialized[candidates] @ centres[missing])])
        counts[int(labels[row])] -= 1
        labels[row] = missing
        counts[missing] = 1
    minimum = min(MIN_L2_CHILD_SUPPORT, max(1, n // 10))
    counts = np.bincount(labels, minlength=len(centres))
    supported = np.flatnonzero(counts >= minimum)
    if len(supported):
        for cell in np.flatnonzero((counts > 0) & (counts < minimum)).tolist():
            members = np.flatnonzero(labels == cell)
            mean = _norm(materialized[members].mean(axis=0, keepdims=True))[0]
            target = int(supported[np.argmax(centres[supported] @ mean)])
            labels[members] = target
        present = np.unique(labels)
        remap = np.full(len(centres), -1, dtype=np.int32)
        remap[present] = np.arange(len(present), dtype=np.int32)
        labels = remap[labels]
        sums = _label_sums(materialized, labels, len(present))
        merged_counts = np.bincount(labels, minlength=len(present))
        centres = _norm((sums / merged_counts[:, None]).astype(np.float32))
    return labels, centres


def spherical_kmeans(
    vectors: np.ndarray, k: int, *, seed: int, niter: int = 25, nredo: int = 3
) -> np.ndarray:
    """Fit ``k`` unit-norm centroids, preferring faiss and falling back to sklearn.

    faiss trains on every point with exact Lloyd iterations where sklearn's
    MiniBatchKMeans samples; measured on this corpus at k=128 over 2M vectors
    that is 4.3 s against 5.7 s, and the gap widens with k — k=2048 over 2M is a
    minute, which is what makes a four-thousand-cell map buildable at all.

    faiss defaults to ``max_points_per_centroid=256``, which would silently train
    a 4096-cell map on a million points no matter how many it was handed, so the
    cap is lifted explicitly.

    ``nredo`` restarts from fresh seeds and keeps the lowest-inertia fit. One
    start is enough to land a degenerate optimum on trivially separable data:
    three orthogonal blobs, k=3, seed 7 put two centroids in one blob (cosine
    0.996 apart) and one centroid across the other two. The sibling-merge pass
    then correctly folds the pair, and the region ends up a cell short of its
    budget for a reason that had nothing to do with its subjects.
    """
    x = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    try:
        import faiss

        model = faiss.Kmeans(
            x.shape[1],
            int(k),
            niter=niter,
            nredo=int(nredo),
            seed=int(seed),
            spherical=True,
            max_points_per_centroid=1_000_000_000,
            verbose=False,
        )
        model.train(x)
        return _norm(np.asarray(model.centroids, dtype=np.float32))
    except Exception:
        from sklearn.cluster import MiniBatchKMeans

        model = MiniBatchKMeans(
            n_clusters=int(k),
            random_state=int(seed),
            batch_size=max(1_024, min(16_384, 32 * int(k))),
            n_init=int(nredo),
            max_iter=300,
        ).fit(x)
        return _norm(model.cluster_centers_.astype(np.float32))


#: How a build may weight records when it fits centroids. ``rows`` is every
#: build before content weighting; ``length`` weights a record by the
#: characters the encoder reads (capped at the profile window); ``sqrt-length``
#: by their square root, a middle course for corpora of short instructions.
FIT_MODES = ("rows", "length", "sqrt-length")


def fit_weights_for(lengths: np.ndarray, mode: str, cap: int) -> np.ndarray | None:
    if mode == "rows":
        return None
    chars = np.minimum(np.asarray(lengths, dtype=np.float64), float(cap))
    chars = np.maximum(chars, 1.0)
    if mode == "length":
        return chars
    if mode == "sqrt-length":
        return np.sqrt(chars)
    raise ValueError(f"unknown fit mode {mode!r}; expected one of {FIT_MODES}")


def _checkpoint_matches(path: Path, corpus_hash: str, fit_mode: str) -> bool:
    """A clustering checkpoint resumes only the corpus and fit mode that wrote it."""
    if not path.is_file():
        return False
    saved = np.load(path, allow_pickle=True)
    mode = str(saved["fit_mode"]) if "fit_mode" in saved.files else "rows"
    return str(saved["corpus_hash"]) == corpus_hash and mode == fit_mode


def weighted_draw(weights: np.ndarray, count: int) -> np.ndarray:
    """``count`` row indices at even steps of cumulative weight, ascending.

    Systematic rather than random, like every other draw in the build, so the
    same corpus always fits the same centroids. A row of weight twice the step
    is drawn twice; a row lighter than the step is drawn with that probability's
    worth of spacing, not at random.
    """
    w = np.asarray(weights, dtype=np.float64)
    count = int(count)
    if not len(w) or count <= 0:
        return np.zeros(0, dtype=np.int64)
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 0:
        return np.linspace(0, len(w) - 1, count, dtype=np.int64)
    positions = (np.arange(count, dtype=np.float64) + 0.5) * (total / count)
    return np.searchsorted(cumulative, positions, side="right").clip(0, len(w) - 1).astype(np.int64)


def _fixed_k_kmeans(
    vectors: np.ndarray, k: int, *, seed: int
) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Split one L1 into exactly ``k`` cells, with no model selection at all.

    Returns the same shape as :func:`_best_kmeans` so the two paths are
    interchangeable; the score is reported as 0.0 because no score was consulted.
    """
    materialized = np.asarray(vectors, dtype=np.float32)
    n = len(materialized)
    k = max(1, min(int(k), n))
    if k < 2 or n < 2:
        return (
            np.zeros(n, dtype=np.int32),
            1,
            0.0,
            _norm(materialized.mean(axis=0, keepdims=True)),
        )
    centres = spherical_kmeans(materialized, k, seed=seed)
    labels = (materialized @ centres.T).argmax(axis=1).astype(np.int32)
    labels, centres = _repair_and_merge(labels, centres, materialized)
    labels, centres = _merge_close_siblings(labels, centres, materialized)
    return labels, len(centres), 0.0, centres


SIBLING_MERGES: list[int] = []


def _merge_close_siblings(
    labels: np.ndarray, centres: np.ndarray, materialized: np.ndarray,
    threshold: float = L2_SIBLING_MERGE_COSINE,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold sibling cells whose centroids are near-identical into one.

    A population budget asks a region for a fixed number of cells and k-means
    delivers them whether or not the region has that many subjects, so a
    region with three real subjects and a budget of five hands back two pairs
    that differ only in where the boundary fell. Mathematically separate,
    semantically one. Merging is transitive over pairs at or above the
    threshold and the number folded is kept for the release notes.
    """
    k = len(centres)
    if k < 2:
        return labels, centres
    sims = centres @ centres.T
    parent = np.arange(k)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.triu_indices(k, 1)
    for i, j in zip(ii[sims[ii, jj] >= threshold], jj[sims[ii, jj] >= threshold], strict=True):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(k)])
    kept = np.unique(roots)
    if len(kept) == k:
        return labels, centres
    remap = np.full(k, -1, dtype=np.int32)
    remap[kept] = np.arange(len(kept), dtype=np.int32)
    labels = remap[roots[labels]]
    sums = _label_sums(materialized, labels, len(kept))
    counts = np.bincount(labels, minlength=len(kept))
    SIBLING_MERGES.append(k - len(kept))
    return labels, _norm((sums / np.maximum(counts, 1)[:, None]).astype(np.float32))


def population_l2_budget(counts: np.ndarray, profile: AtlasProfile) -> np.ndarray:
    """Split the L2 budget across L1 regions by population, not by silhouette.

    ``allocate_l2_budget`` spends the budget where a silhouette curve says the
    next split helps most. Measured on the v2 corpus that curve is noise: it
    varies by less than 0.02 across k=2..64 and its differences are smaller than
    its sampling error, so the marginal-coherence term is a random number and
    the 0.002 support term never gets to matter. Population is the honest
    criterion left.

    Cells go as population to the power :data:`L2_BUDGET_EXPONENT`, which trades
    two things against each other. Proportional allocation (exponent 1) gives
    every cell about the same number of records — even calibration — but starves
    the smallest L1 regions, which is the wrong direction for a corpus that is
    85% general web: resolution would follow volume, and the regions worth
    telling apart (code, mathematics, law) are the small ones. A square root
    fixes the starvation but swings cell populations by 11x.

    Measured on a lognormal spread of 256 L1 populations, 4,096 cells:

    ==============  ===========  =======================  ===================
    exponent        min cells    records/cell spread      cells to top decile
    ==============  ===========  =======================  ===================
    1.00            1            1.5x                     30.9%
    0.75            2            3.3x                     24.5%
    0.50            4            11.2x                    18.9%
    ==============  ===========  =======================  ===================

    0.75 is the compromise: no region reduced to a single cell, populations
    within a factor of three or so, and the largest tenth of regions holding
    30.9% of the text takes 24.5% of the resolution rather than all of it.
    """
    counts = np.asarray(counts, dtype=np.int64)
    budget = int(profile.l2_budget or 0)
    live = counts > 0
    if budget <= 0 or not live.any():
        return np.where(live, 1, 0).astype(np.int32)

    ceiling = np.zeros(len(counts), dtype=np.int64)
    ceiling[live] = np.minimum(profile.l2_k_max, np.maximum(1, counts[live] - 1))
    floor = np.zeros(len(counts), dtype=np.int64)
    floor[live] = np.minimum(max(1, profile.l2_k_min), ceiling[live])
    target = int(min(max(budget, floor.sum()), ceiling.sum()))

    weight = np.zeros(len(counts), dtype=np.float64)
    weight[live] = counts[live].astype(np.float64) ** L2_BUDGET_EXPONENT
    ideal = weight / weight.sum() * target

    # Largest-remainder rounding, then push the residual into whichever regions
    # are still under their ceiling, largest first. Deterministic on ties.
    allocated = np.clip(np.floor(ideal), floor, ceiling).astype(np.int64)
    while allocated.sum() < target:
        room = allocated < ceiling
        if not room.any():
            break
        deficit = np.where(room, ideal - allocated, -np.inf)
        allocated[int(np.argmax(deficit))] += 1
    while allocated.sum() > target:
        over = allocated > floor
        if not over.any():
            break
        surplus = np.where(over, allocated - ideal, -np.inf)
        allocated[int(np.argmax(surplus))] -= 1
    return allocated.astype(np.int32)


def _kmeans_score_curve(
    vectors: np.ndarray, k_min: int, k_max: int, *, seed: int
) -> dict[int, float]:
    """Return a deterministic, bounded-cost silhouette curve for one L1."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    n = len(vectors)
    hi = min(int(k_max), n - 1)
    lo = min(max(int(k_min), 1), hi)
    if hi < 2:
        return {1: 0.0}
    rng = np.random.default_rng(seed)
    fit_n = min(n, 10_000)
    fit_idx = np.arange(n) if n == fit_n else np.sort(rng.choice(n, fit_n, replace=False))
    fit = np.asarray(vectors[fit_idx], dtype=np.float32)
    score_n = min(len(fit), 1_000)
    score_idx = (
        np.arange(len(fit))
        if len(fit) == score_n
        else np.sort(rng.choice(len(fit), score_n, replace=False))
    )
    scoring = fit[score_idx]
    scores: dict[int, float] = {}
    for k in range(max(lo, 2), hi + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=min(2_048, max(256, k * 16)),
            n_init=2,
            max_iter=100,
        ).fit(fit)
        guessed = model.predict(scoring)
        scores[k] = (
            -1.0
            if len(set(guessed.tolist())) < 2
            else float(silhouette_score(scoring, guessed, metric="cosine"))
        )
    if lo <= 1:
        scores[1] = 0.0
    return scores


def allocate_l2_budget(
    counts: np.ndarray,
    curves: list[dict[int, float]],
    profile: AtlasProfile,
) -> np.ndarray:
    """Allocate the global L2 budget by marginal coherence and L1 support."""
    lower = np.zeros(len(counts), dtype=np.int32)
    upper = np.zeros(len(counts), dtype=np.int32)
    for parent, count in enumerate(counts.tolist()):
        if count <= 0:
            continue
        upper[parent] = min(profile.l2_k_max, max(1, count - 1))
        lower[parent] = min(profile.l2_k_min, upper[parent])
    budget = profile.l2_budget or int(lower.sum())
    target = min(max(int(budget), int(lower.sum())), int(upper.sum()))
    allocated = lower.copy()
    positive = counts[counts > 0]
    mean_count = float(positive.mean()) if len(positive) else 1.0
    while int(allocated.sum()) < target:
        candidates: list[tuple[float, int]] = []
        for parent, current in enumerate(allocated.tolist()):
            if current >= upper[parent]:
                continue
            curve = curves[parent]
            before = curve.get(current, curve.get(min(curve), 0.0))
            after = curve.get(current + 1, before)
            support = math.log1p(float(counts[parent]) / mean_count)
            priority = (after - before) + 0.002 * support
            candidates.append((priority, parent))
        if not candidates:
            break
        _, parent = max(candidates, key=lambda item: (item[0], counts[item[1]], -item[1]))
        allocated[parent] += 1
    return allocated


def discover_cells(vectors: np.ndarray, profile: AtlasProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.cluster import MiniBatchKMeans

    if len(vectors) < profile.n_l1:
        raise ValueError(f"{profile.version} needs at least {profile.n_l1} records")
    fit = vectors if len(vectors) <= 500_000 else vectors[np.linspace(0, len(vectors) - 1, 500_000, dtype=np.int64)]
    l1 = MiniBatchKMeans(n_clusters=profile.n_l1, random_state=SEED, batch_size=16_384,
                         n_init=3, max_iter=300).fit(fit)
    l1_centroids = _norm(l1.cluster_centers_.astype(np.float32))
    l1_assign = np.empty(len(vectors), dtype=np.int32)
    for start in range(0, len(vectors), 65_536):
        block = vectors[start:start + 65_536]
        l1_assign[start:start + len(block)] = (block @ l1_centroids.T).argmax(axis=1)
    cell_assign = np.empty(len(vectors), dtype=np.int32)
    cell_parent: list[int] = []
    next_cell = 0
    for parent in range(profile.n_l1):
        members = np.flatnonzero(l1_assign == parent)
        if not len(members):
            continue
        local, n_local, score, _ = _best_kmeans(
            vectors[members], profile.l2_k_min, profile.l2_k_max, seed=SEED + parent,
        )
        cell_assign[members] = local + next_cell
        cell_parent.extend([parent] * n_local)
        next_cell += n_local
        log(f"    L1 {parent:3d}: {len(members):,} rows → {n_local} cells (k* silhouette {score:.3f})")
    return cell_assign, np.asarray(cell_parent, dtype=np.int32), l1_centroids


def _terms(texts: list[str], assignment: np.ndarray, n_cells: int, limit: int) -> list[list[str]]:
    per_cell = [Counter() for _ in range(n_cells)]
    document_frequency: Counter[str] = Counter()
    for text, cell in zip(texts, assignment.tolist(), strict=True):
        words = {word.lower() for word in text.split() if len(word) > 3 and word.isalpha()}
        document_frequency.update(words)
        per_cell[cell].update(words)
    total = max(sum(document_frequency.values()), 1)
    return [[word for _, word in sorted(
        ((count * math.log(1 + total / (1 + document_frequency[word])), word) for word, count in counts.items()), reverse=True
    )[:limit]] for counts in per_cell]


def _calibration(vectors: np.ndarray, assignment: np.ndarray, centroids: np.ndarray, knots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = len(centroids)
    distances = 1.0 - np.einsum("ij,ij->i", vectors, centroids[assignment])
    refs = np.zeros((n_cells, knots), dtype=np.float32)
    support = np.bincount(assignment, minlength=n_cells).astype(np.int32)
    for cell in range(n_cells):
        local = distances[assignment == cell]
        if len(local):
            refs[cell] = np.quantile(local, np.linspace(0.01, 0.99, knots)).astype(np.float32)
    return refs, support, distances


def _cooccurrence(assignment: np.ndarray, sources: list[str], n_cells: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    source_ids = {value: i for i, value in enumerate(sorted(set(sources)))}
    counts = np.zeros((n_cells, len(source_ids)), np.int32)
    np.add.at(counts, (assignment, np.asarray([source_ids[value] for value in sources])), 1)
    present = (counts > 0).astype(np.int32)
    similarity = present @ present.T
    np.fill_diagonal(similarity, 0)
    take = min(k, max(n_cells - 1, 1))
    indices = np.argsort(-similarity, axis=1)[:, :take].astype(np.int32)
    scores = np.take_along_axis(similarity, indices, axis=1).astype(np.float32)
    return indices, scores


def _prototypes(vectors: np.ndarray, assignment: np.ndarray, centroids: np.ndarray, count: int) -> np.ndarray:
    result = np.zeros((len(centroids), count, vectors.shape[1]), np.float16)
    for cell in range(len(centroids)):
        members = np.flatnonzero(assignment == cell)
        if not len(members):
            continue
        ranks = np.argsort(-(vectors[members] @ centroids[cell]))
        picks = members[ranks[np.linspace(0, len(ranks) - 1, min(count, len(ranks)), dtype=np.int64)]]
        result[cell, :len(picks)] = vectors[picks].astype(np.float16)
    return result


def _exemplars(texts: list[str], assignment: np.ndarray, vectors: np.ndarray,
               centroids: np.ndarray, count: int, chars: int) -> np.ndarray:
    output = np.full((len(centroids), count), "", dtype=f"U{chars}")
    for cell in range(len(centroids)):
        members = np.flatnonzero(assignment == cell)
        if not len(members):
            continue
        ranked = members[np.argsort(-(vectors[members] @ centroids[cell]))[:count]]
        output[cell, :len(ranked)] = [texts[i][:chars] for i in ranked]
    return output


def _crosswalk(full_assignment: np.ndarray, lite_assignment: np.ndarray,
               n_full: int, n_lite: int) -> dict:
    pairs = np.zeros((n_full, n_lite), dtype=np.int64)
    np.add.at(pairs, (full_assignment, lite_assignment), 1)
    return {
        "method": "modal_population_overlap_on_shared_lite_reference",
        "full_to_lite": pairs.argmax(axis=1).astype(int).tolist(),
        "lite_to_full": pairs.argmax(axis=0).astype(int).tolist(),
        "full_modal_share": (pairs.max(axis=1) / np.maximum(pairs.sum(axis=1), 1)).round(6).tolist(),
        "lite_modal_share": (pairs.max(axis=0) / np.maximum(pairs.sum(axis=0), 1)).round(6).tolist(),
    }


def _tokenize(embedder, texts: list[str], budget: int) -> tuple[np.ndarray, np.ndarray]:
    chunks, indptr = [], [0]
    for start in range(0, len(texts), BLOCK):
        tokenized = embedder.tokenize(texts[start:start + BLOCK], max_length=budget)
        for row in range(tokenized.n_docs):
            ids = selected_token_ids(tokenized.token_ids[tokenized.indptr[row]:tokenized.indptr[row + 1]], budget)
            chunks.append(ids)
            indptr.append(indptr[-1] + len(ids))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.int32), np.asarray(indptr, dtype=np.int64)


def _encode(embedder, token_ids: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    return embedder.encode_tokenized(TokenizedCorpus(
        token_ids=token_ids, indptr=indptr,
        n_docs=len(indptr) - 1, max_length=int(np.max(np.diff(indptr), initial=0)),
    ))


def _prepare_batch(raw_texts: list[str], axes: list[str], seen: Uint64Set) -> tuple[list[str], list[int]]:
    kept_texts: list[str] = []
    kept_index: list[int] = []
    for i, (raw, axis) in enumerate(zip(raw_texts, axes, strict=True)):
        text, _ = extract_text(raw, detected_format="code" if axis == "code" else None)
        text = text[:ATLAS_V2.max_chars]
        if len(text) < 80:
            continue
        digest = int.from_bytes(hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(), "little")
        if not seen.add(digest):
            continue
        kept_texts.append(text)
        kept_index.append(i)
    return kept_texts, kept_index


def _count_tokens(embedder, texts: list[str], counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids, pointers = _tokenize(embedder, texts, ATLAS_V2.max_tokens)
    if ids.size:
        np.add.at(counts, ids, 1)
    return ids, pointers


def _idf_from_counts(counts: np.ndarray) -> tuple[dict[int, float], tuple[np.ndarray, np.ndarray], float]:
    total = int(counts.sum())
    observed = np.flatnonzero(counts)
    if total <= 0 or not len(observed):
        empty = np.zeros(0, np.int32)
        return {}, (empty, np.zeros(0, np.float32)), 0.0
    probs = counts[observed] / total
    log_probs = np.log(probs).astype(np.float32)
    mapping = {int(token): float(logp) for token, logp in zip(observed, log_probs, strict=True)}
    mass = float(counts[observed].sum() / total)
    return mapping, (observed.astype(np.int32), log_probs), mass


def _encode_texts(embedder, texts: list[str]) -> np.ndarray:
    ids, pointers = _tokenize(embedder, texts, ATLAS_V2.max_tokens)
    return _encode(embedder, ids, pointers)


_LANGID_POOL = None
#: Detector codes that the corpus plan spells differently.
_LANGID_ALIASES = {"nb": "no", "nn": "no"}


def detect_languages(texts: list[str], declared: list[str]) -> list[str]:
    """The language each record is actually in, falling back to its source's.

    Until this build a record's language was whatever its source declared,
    inherited by every row in the shard. That is wrong often enough to matter:
    the shipped atlas-v3 partitioned Ukrainian text into cells named Serbian
    and Bulgarian because the shard said so, and the per-language centering
    then subtracted the wrong mean and left the language in. The detector is
    batched byte n-grams at ~10k records/s a core, so it runs in a small pool
    beside the encoder rather than serially in front of it.
    """
    global _LANGID_POOL
    if not texts:
        return []
    if _LANGID_POOL is None:
        from concurrent.futures import ProcessPoolExecutor

        import atlas_langid_worker

        _LANGID_POOL = ProcessPoolExecutor(
            max_workers=LANGID_WORKERS, initializer=atlas_langid_worker.init
        )
    import atlas_langid_worker

    step = max(32, -(-len(texts) // LANGID_WORKERS))
    chunks = [texts[i:i + step] for i in range(0, len(texts), step)]
    flat: list[tuple[str, float]] = []
    for part in _LANGID_POOL.map(atlas_langid_worker.detect_chunk, chunks):
        flat.extend(part)
    return resolve_languages(flat, declared)


def resolve_languages(detections: list[tuple[str, float]], declared: list[str]) -> list[str]:
    """Detected language where the detector commits, the source's otherwise."""
    out: list[str] = []
    for (lang, conf), fallback in zip(detections, declared, strict=True):
        lang = _LANGID_ALIASES.get(lang, lang)
        out.append(lang if lang != "unknown" and conf >= LANGID_CONFIDENCE else fallback)
    return out


def _ingest_prepared(
    corpus: DiskCorpus,
    embedder,
    texts: list[str],
    axes: list[str],
    languages: list[str],
    sources: list[str],
    reservoir: Reservoir,
    lite_reservoir: Reservoir,
    token_counts: np.ndarray | None,
    *,
    detected: list[str] | None = None,
    token_ids: list[np.ndarray] | None = None,
) -> int:
    """Encode, semantically dedup and append one batch of extracted texts.

    ``detected`` and ``token_ids`` carry what a :func:`atlas_langid_worker.prepare_chunk`
    worker already computed for these texts; without them the batch is
    detected and tokenized here, as the single-process build always did.
    """
    if not texts:
        return 0
    if token_counts is not None:
        _count_tokens(embedder, texts, token_counts)
    languages = detected if detected is not None else detect_languages(texts, languages)
    if token_ids is not None:
        pointers = np.zeros(len(token_ids) + 1, dtype=np.int64)
        np.cumsum([len(ids) for ids in token_ids], out=pointers[1:])
        flat = np.concatenate(token_ids) if token_ids else np.zeros(0, np.int32)
        vectors = _encode(embedder, flat.astype(np.int32, copy=False), pointers)
    else:
        vectors = _encode_texts(embedder, texts)
    keep = np.ones(len(vectors), dtype=bool)
    block = _norm(vectors)
    sims = block @ block.T
    keep[np.where(np.triu(sims >= 0.995, 1).any(axis=0))[0]] = False
    kept = np.flatnonzero(keep)
    texts = [texts[i] for i in kept]
    axes = [axes[i] for i in kept]
    languages = [languages[i] for i in kept]
    sources = [sources[i] for i in kept]
    vectors = vectors[kept]
    start = corpus.append(vectors, axes, languages, sources, [len(text) for text in texts])
    for offset, (text, axis, language, source) in enumerate(zip(texts, axes, languages, sources, strict=True)):
        row = start + offset
        nbytes = len(text.encode("utf-8"))
        corpus.logical_by_lang[language] = corpus.logical_by_lang.get(language, 0) + nbytes
        corpus.logical_by_axis[axis] = corpus.logical_by_axis.get(axis, 0) + nbytes
        reservoir.offer(row, text[:FULL_EXEMPLAR_CHARS], axis, language, source)
        lite_reservoir.offer(row, text[:ATLAS_V2_LITE.max_chars], axis, language, source)
    return len(texts)


def _source_language(src: Source) -> str:
    lang = src.lang
    if lang.startswith("code:"):
        return "en"
    return lang


def manifest_sources(cache: Path) -> tuple[list[Source], dict]:
    """Resolve every retained manifest shard without touching the network."""
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing corpus manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: list[Source] = []
    slugs: set[str] = set()
    allowed = {
        "hf_id", "config", "split", "fields", "axis", "target", "lang",
        "loader", "path", "revision", "canonical", "target_bytes",
        "source_role", "license_policy",
    }
    for meta in manifest.get("sources", []):
        if int(meta.get("rows", 0)) <= 0:
            continue
        requested = {key: value for key, value in meta.get("requested", {}).items()
                     if key in allowed}
        requested["fields"] = tuple(requested.get("fields", ("text",)))
        source = Source(**requested)
        slug = str(meta.get("slug", ""))
        if source.slug != slug:
            raise ValueError(f"manifest slug mismatch: {slug} != {source.slug}")
        if slug in slugs:
            raise ValueError(f"duplicate manifest slug: {slug}")
        if not fetch_corpus.cached_shard_path(cache, slug).is_file():
            raise FileNotFoundError(f"missing manifest shard: {slug}")
        slugs.add(slug)
        sources.append(source)
    expected = int(manifest.get("totals", {}).get("sources_with_rows", 0))
    if len(sources) != expected:
        raise ValueError(f"manifest resolved {len(sources)} sources; expected {expected}")
    return sources, manifest


def allocate_idf_sample(rows_by_slug: dict[str, int], total: int) -> dict[str, int]:
    """Allocate an exact proportional IDF sample across manifest shards."""
    available = sum(rows_by_slug.values())
    target = min(max(int(total), 0), available)
    if not target or not available:
        return dict.fromkeys(rows_by_slug, 0)
    exact = {
        slug: target * rows / available
        for slug, rows in rows_by_slug.items()
    }
    quotas = {slug: min(rows_by_slug[slug], int(value)) for slug, value in exact.items()}
    remainder = target - sum(quotas.values())
    order = sorted(
        rows_by_slug,
        key=lambda slug: (-(exact[slug] - quotas[slug]), slug),
    )
    for slug in order:
        if not remainder:
            break
        if quotas[slug] < rows_by_slug[slug]:
            quotas[slug] += 1
            remainder -= 1
    if sum(quotas.values()) != target:
        raise RuntimeError("IDF sample allocation did not preserve its target")
    return quotas


def _scan_cache_for_idf(
    cache: Path,
    sources: list[Source],
    embedder,
    counts: np.ndarray,
    seen: Uint64Set,
    rows_by_slug: dict[str, int],
    sample_rows: int = IDF_SAMPLE_ROWS,
    source_progress: Callable[[int, Source, int, int], None] | None = None,
    phrases: list[np.ndarray] | None = None,
) -> int:
    """Fit SIF IDF on a proportional, systematic manifest sample.

    When ``phrases`` is a list, the same pass also finds each source's template
    phrases (see :mod:`dropoutt.atlas.textnorm`) from an even stride of up to
    :data:`PHRASE_DOCS_PER_SOURCE` of the documents it already tokenizes, and
    appends them. A source's phrases are fitted when the source ends, so only
    one source's token ids are ever held.
    """
    scanned = 0
    quotas = allocate_idf_sample(rows_by_slug, sample_rows)
    for source_index, src in enumerate(sources, 1):
        shard = fetch_corpus.cached_shard_path(cache, src.slug)
        if not shard.is_file():
            continue
        batch: list[str] = []
        axes: list[str] = []
        source_rows = 0
        expected = rows_by_slug[src.slug]
        quota = quotas[src.slug]
        positions = np.linspace(0, expected - 1, quota, dtype=np.int64)
        position_index = 0
        phrase_stride = max(1, quota // PHRASE_DOCS_PER_SOURCE)
        phrase_docs: list[np.ndarray] = []
        offered = 0

        def count(
            kept: list[str],
            phrase_docs: list[np.ndarray] = phrase_docs,
            phrase_stride: int = phrase_stride,
        ) -> None:
            nonlocal offered
            ids, pointers = _count_tokens(embedder, kept, counts)
            if phrases is None:
                return
            for row in range(len(kept)):
                if offered % phrase_stride == 0 and len(phrase_docs) < PHRASE_DOCS_PER_SOURCE:
                    phrase_docs.append(ids[pointers[row]:pointers[row + 1]].copy())
                offered += 1

        for text in fetch_corpus.iter_cached_texts(cache, src.slug, ATLAS_V2.max_chars):
            source_rows += 1
            row_index = source_rows - 1
            if position_index >= quota or row_index != int(positions[position_index]):
                if source_progress is not None and source_rows % PROGRESS_EVERY < BLOCK:
                    source_progress(source_index, src, source_rows, scanned)
                continue
            position_index += 1
            batch.append(text)
            axes.append(src.axis)
            if len(batch) >= BLOCK:
                kept, _ = _prepare_batch(batch, axes, seen)
                if kept:
                    count(kept)
                    scanned += len(kept)
                batch, axes = [], []
            if source_progress is not None and source_rows % PROGRESS_EVERY < BLOCK:
                source_progress(source_index, src, source_rows, scanned)
        if batch:
            kept, _ = _prepare_batch(batch, axes, seen)
            if kept:
                count(kept)
                scanned += len(kept)
        found = ""
        if phrases is not None:
            fitted = source_phrases(phrase_docs)
            phrases.append(fitted)
            found = f"  template phrases={len(fitted):,}"
        log(f"  idf {src.slug[:52]:<52} running types={int((counts > 0).sum()):,}{found}")
        if source_progress is not None:
            source_progress(source_index, src, source_rows, scanned)
    return scanned


def _consume_cache(
    cache: Path,
    sources: list[Source],
    corpus: DiskCorpus,
    embedder,
    seen: Uint64Set,
    reservoir: Reservoir,
    lite_reservoir: Reservoir,
    token_counts: np.ndarray | None,
    consumed: set[str],
    checkpoint: Callable[[int], None] | None = None,
    source_progress: Callable[[int, Source, int], None] | None = None,
    prepare_pool=None,
) -> tuple[int, int, set[str]]:
    """Encode every manifest shard while keeping the corpus cache read-only.

    With ``prepare_pool`` (a process pool initialised by
    :func:`atlas_langid_worker.init` with a tokenizer) the per-record stages run
    in the pool, several raw batches ahead of the main process, and each batch
    comes back in order to the same dedup, encoding and append it always went
    through. The output is the single-process output, row for row.
    """
    rows = 0
    logical = 0
    for source_index, src in enumerate(sources, 1):
        if src.slug in consumed:
            continue
        shard = fetch_corpus.cached_shard_path(cache, src.slug)
        if not shard.is_file():
            continue
        if prepare_pool is not None:
            added_rows, added_logical = _consume_source_parallel(
                cache, src, source_index, corpus, embedder, seen, reservoir,
                lite_reservoir, prepare_pool, source_progress,
            )
            rows += added_rows
            logical += added_logical
            consumed.add(src.slug)
            log(f"  used  {src.slug[:52]:<52} corpus={corpus.n:,}  {logical / (1024 ** 3):.2f} GiB")
            corpus.flush()
            if checkpoint is not None:
                checkpoint(logical)
            continue
        batch_text: list[str] = []
        batch_axis: list[str] = []
        batch_lang: list[str] = []
        batch_src: list[str] = []
        source_rows = 0
        language = _source_language(src)
        for text in fetch_corpus.iter_cached_texts(cache, src.slug, ATLAS_V2.max_chars):
            source_rows += 1
            batch_text.append(text)
            batch_axis.append(src.axis)
            batch_lang.append(language)
            batch_src.append(src.slug)
            if len(batch_text) >= BLOCK:
                kept, index = _prepare_batch(batch_text, batch_axis, seen)
                if kept:
                    kept_bytes = sum(len(t.encode("utf-8")) for t in kept)
                    logical += kept_bytes
                    rows += _ingest_prepared(
                        corpus, embedder, kept,
                        [batch_axis[i] for i in index],
                        [batch_lang[i] for i in index],
                        [batch_src[i] for i in index],
                        reservoir, lite_reservoir, token_counts,
                    )
                batch_text, batch_axis, batch_lang, batch_src = [], [], [], []
                if source_progress is not None and source_rows % PROGRESS_EVERY < BLOCK:
                    source_progress(source_index, src, source_rows)
        if batch_text:
            kept, index = _prepare_batch(batch_text, batch_axis, seen)
            if kept:
                kept_bytes = sum(len(t.encode("utf-8")) for t in kept)
                logical += kept_bytes
                rows += _ingest_prepared(
                    corpus, embedder, kept,
                    [batch_axis[i] for i in index],
                    [batch_lang[i] for i in index],
                    [batch_src[i] for i in index],
                    reservoir, lite_reservoir, token_counts,
                )
        consumed.add(src.slug)
        log(f"  used  {src.slug[:52]:<52} corpus={corpus.n:,}  {logical / (1024 ** 3):.2f} GiB")
        corpus.flush()
        if source_progress is not None:
            source_progress(source_index, src, source_rows)
        if checkpoint is not None:
            checkpoint(logical)
    return rows, logical, consumed


#: Raw batches submitted to the prepare pool ahead of the main process.
PREPARE_AHEAD = 24
#: Workers for per-record ingest stages; the main process keeps one core.
PREPARE_WORKERS = max(4, (os.cpu_count() or 8) - 3)


def _consume_source_parallel(
    cache: Path, src: Source, source_index: int, corpus: DiskCorpus, embedder,
    seen: Uint64Set, reservoir: Reservoir, lite_reservoir: Reservoir, pool,
    source_progress: Callable[[int, Source, int], None] | None,
) -> tuple[int, int]:
    """One source through the prepare pool; returns (rows appended, logical bytes)."""
    from collections import deque

    import atlas_langid_worker

    language = _source_language(src)
    is_code = src.axis == "code"
    pending: deque = deque()
    rows = logical = 0
    submitted = 0
    done_rows = 0

    def finish_one() -> None:
        nonlocal rows, logical, done_rows
        future, size = pending.popleft()
        texts, digests, detections, token_ids = future.result()
        slot = 0
        kept: list[str] = []
        kept_languages: list[str] = []
        kept_ids: list[np.ndarray] = []
        for text, digest in zip(texts, digests, strict=True):
            if text is None:
                continue
            here = slot
            slot += 1
            if not seen.add(digest):
                continue
            kept.append(text)
            kept_languages.append(resolve_languages([detections[here]], [language])[0])
            kept_ids.append(token_ids[here])
        done_rows += size
        if kept:
            logical += sum(len(t.encode("utf-8")) for t in kept)
            rows += _ingest_prepared(
                corpus, embedder, kept, [src.axis] * len(kept), [language] * len(kept),
                [src.slug] * len(kept), reservoir, lite_reservoir, None,
                detected=kept_languages, token_ids=kept_ids,
            )
        if source_progress is not None and done_rows % PROGRESS_EVERY < BLOCK:
            source_progress(source_index, src, done_rows)

    batch: list[tuple[str, bool]] = []
    for text in fetch_corpus.iter_cached_texts(cache, src.slug, ATLAS_V2.max_chars):
        batch.append((text, is_code))
        if len(batch) >= BLOCK:
            pending.append((pool.submit(atlas_langid_worker.prepare_chunk, batch), len(batch)))
            submitted += len(batch)
            batch = []
            while len(pending) > PREPARE_AHEAD:
                finish_one()
    if batch:
        pending.append((pool.submit(atlas_langid_worker.prepare_chunk, batch), len(batch)))
    while pending:
        finish_one()
    if source_progress is not None:
        source_progress(source_index, src, done_rows)
    return rows, logical


def prepare_pool(policy: EncoderInput, workers: int = PREPARE_WORKERS):
    """A spawn pool whose workers extract, hash, detect and tokenize like the build."""
    from concurrent.futures import ProcessPoolExecutor

    import atlas_langid_worker

    from dropoutt.atlas.embed import local_model_dir
    from dropoutt.config import cache_dir

    tokenizer = local_model_dir(DEFAULT_MODEL, cache_dir()) / "tokenizer.json"
    if not tokenizer.is_file():
        raise SystemExit(f"prepare pool needs the encoder tokenizer at {tokenizer}")
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=atlas_langid_worker.init,
        initargs=(str(tokenizer), policy.declaration(), ATLAS_V2.max_chars, ATLAS_V2.max_tokens, 80),
    )


def _build_from_memmap(
    profile: AtlasProfile,
    corpus: DiskCorpus,
    reservoir: Reservoir,
    idf: tuple[np.ndarray, np.ndarray] | None,
    encoder_hash: str,
    corpus_hash: str,
    out: Path,
    *,
    idf_fit_records: int = 0,
    subset: np.ndarray | None = None,
    lite_items: list[tuple[int, str, str, str, str]] | None = None,
    progress: Callable[[float, str], None] | None = None,
    encoder_input: EncoderInput = RAW_INPUT,
    fit_weights: np.ndarray | None = None,
    fit_mode: str = "rows",
) -> dict:
    """Normalise, cluster and write one artifact.

    ``fit_weights`` (one non-negative weight per row, positional over ``subset``
    when one is given) makes the map follow content rather than row count: L1
    centroids train on a weight-systematic draw, L2 budgets split the L1
    regions by weight, each L2 fit draws its members by weight, and centroids
    are weighted means. Every record is still assigned and counted, so
    ``region_size`` and the density model stay in records. Without weights the
    build is the row-count build it always was.
    """
    last_progress = -1

    def report(percent: float, detail: str) -> None:
        nonlocal last_progress
        whole = int(percent)
        if progress is not None and (whole > last_progress or percent >= 100):
            progress(percent, detail)
            last_progress = whole

    if subset is None:
        n = corpus.n
        take = None
    else:
        take = subset
        n = len(take)
    sample_idx = balanced_calibration_draw(corpus, take, n, NORM_SAMPLE_ROWS)
    sample = corpus.rows_f32(sample_idx)[:, :profile.dim]
    sample_langs = [corpus.lang_table[int(i)] for i in corpus.lang_ids[sample_idx]]
    report(2.0, f"balanced calibration draw: {len(sample_idx):,} rows")

    # Clustering has no ingest-style checkpoint of its own, and a kill during
    # it threw away a two-hour normalization pass and fourteen hours of L2
    # fits. Three small checkpoint files now let it pick up where it stopped:
    # the normalization outputs beside the memmap, the L1 fit, and the L2
    # loop's progress every few regions. Each is keyed to the corpus hash so a
    # different corpus can never resume from them.
    norm_dir = Path(NORM_DIR) if NORM_DIR else corpus.work
    norm_dir.mkdir(parents=True, exist_ok=True)
    path = norm_dir / f"norm-{profile.version}.f16"
    ck_norm = corpus.work / f"cluster-norm-{profile.version}.npz"
    ck_l1 = corpus.work / f"cluster-l1-{profile.version}.npz"
    ck_l2 = corpus.work / f"cluster-l2-{profile.version}.npz"
    expected_bytes = n * profile.dim * 2
    resumed_norm = (
        ck_norm.is_file() and path.is_file() and path.stat().st_size == expected_bytes
        and str(np.load(ck_norm, allow_pickle=True)["corpus_hash"]) == corpus_hash
    )
    if resumed_norm:
        saved = np.load(ck_norm, allow_pickle=True)
        mean, pca = saved["mean"], saved["pca"]
        lang_means, lang_labels = saved["lang_means"], list(saved["lang_labels"])
        probe = json.loads(str(saved["probe"]))
        mm = np.memmap(path, dtype=np.float16, mode="r+", shape=(n, profile.dim))
        log(f"  resumed normalization from {ck_norm.name} ({len(lang_labels)} language means)")
        report(16.0, "normalization resumed from checkpoint")
    else:
        lang_mean_table, lang_mean_labels = fit_language_means(corpus, profile, take, n)
        report(3.0, f"fitted means for {len(lang_mean_labels)} languages")
        mean, pca, lang_means, lang_labels = fit_normalization(
            sample, sample_langs, profile, lang_mean_table, lang_mean_labels
        )
        mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, profile.dim))
    for start in ([] if resumed_norm else range(0, n, 32_768)):
        stop = min(start + 32_768, n)
        rows = slice(start, stop) if take is None else take[start:stop]
        raw = corpus.slice_f32(start, stop)[:, :profile.dim] if take is None else corpus.rows_f32(rows)[:, :profile.dim]
        langs = [corpus.lang_table[int(i)] for i in corpus.lang_ids[rows]]
        mm[start:stop] = apply_normalization(raw, mean, pca, langs, lang_labels, lang_means)
        report(15.0 * stop / n, "normalizing full corpus")
    mm.flush()
    if not resumed_norm:
        probe = language_probe(corpus, mm, take, n, profile.dim)
        np.savez_compressed(
            ck_norm, mean=mean, pca=pca, lang_means=lang_means,
            lang_labels=np.asarray(lang_labels, dtype=object),
            probe=json.dumps(probe), corpus_hash=corpus_hash, n=n,
        )
    if not resumed_norm and not probe.get("skipped"):
        log(
            "  language probe (balanced acc): "
            f"language {probe['language_raw']:.3f} -> {probe['language_normalized']:.3f}, "
            f"axis {probe['axis_raw']:.3f} -> {probe['axis_normalized']:.3f}"
        )
        if probe["language_normalized"] > 0.5 * probe["language_raw"]:
            log("  warning: normalization removed under half of linearly "
                "recoverable language")
        if probe["axis_normalized"] < 0.9 * probe["axis_raw"]:
            log("  warning: normalization cost more than a tenth of axis "
                "separability")
    report(16.0, "language probe recorded")

    if n < profile.n_l1:
        raise ValueError(f"{profile.version} needs at least {profile.n_l1} records")
    # A k-means fit stops improving at roughly a thousand records per centroid:
    # measured against a full-pool reference on the v2 corpus, AMI climbs 0.539
    # (50k) -> 0.628 (1M) -> 0.640 (2M) and the reseed noise floor is 0.642, so
    # 2M is already indistinguishable from using everything. The old flat 500k
    # was below that line for any L1 count above ~500.
    l1_fit_rows = min(n, max(2_000_000, 1_000 * profile.n_l1))
    kmeans_idx = np.linspace(0, n - 1, l1_fit_rows, dtype=np.int64)
    weights = None
    if fit_weights is not None:
        weights = np.asarray(fit_weights[:n], dtype=np.float64)
        weights = weights / max(float(weights.mean()), 1e-12)
    # The calibration and prototype sample stays a row draw below: distances are
    # a statement about records. Only the centroids are fitted to content.
    l1_train_idx = kmeans_idx if weights is None else weighted_draw(weights, l1_fit_rows)
    if _checkpoint_matches(ck_l1, corpus_hash, fit_mode):
        saved = np.load(ck_l1)
        l1_centroids, l1_assign = saved["l1_centroids"], saved["l1_assign"]
        log(f"  resumed L1 regions from {ck_l1.name}")
        report(30.0, "L1 regions resumed from checkpoint")
    else:
        l1_centroids = spherical_kmeans(
            gather_rows(mm, l1_train_idx), profile.n_l1, seed=SEED
        )
        l1_assign = np.empty(n, dtype=np.int32)
        for start in range(0, n, 65_536):
            stop = min(start + 65_536, n)
            l1_assign[start:stop] = (np.asarray(mm[start:stop], dtype=np.float32) @ l1_centroids.T).argmax(axis=1)
            report(15.0 + 15.0 * stop / n, "assigning L1 regions")
        np.savez_compressed(
            ck_l1, l1_centroids=l1_centroids, l1_assign=l1_assign, corpus_hash=corpus_hash,
            fit_mode=fit_mode,
        )

    assignment = np.empty(n, dtype=np.int32)
    cell_parent: list[int] = []
    next_cell = 0
    l1_counts = np.bincount(l1_assign, minlength=profile.n_l1).astype(np.int64)
    l1_order = np.argsort(l1_assign, kind="stable")
    l1_offsets = np.r_[0, np.cumsum(l1_counts)]
    budgeted = profile.l2_budget is not None
    if budgeted:
        if weights is None:
            population = l1_counts
        else:
            # Content, not rows: a region of one-line prompts gets the cells its
            # text warrants. Never more cells than the region has records.
            content = np.bincount(l1_assign, weights=weights, minlength=profile.n_l1)
            population = np.minimum(np.rint(content).astype(np.int64), l1_counts)
        planned = population_l2_budget(population, profile)
        log(
            f"  allocating {int(planned.sum()):,} L2 cells across "
            f"{int((l1_counts > 0).sum())} populated L1 regions by "
            f"{'content' if weights is not None else 'population'}^{L2_BUDGET_EXPONENT}"
        )
    else:
        planned = None
        log(
            f"  selecting independent best-k L2 cells in "
            f"[{profile.l2_k_min}, {profile.l2_k_max}]"
        )
    # Every L2 fit gathered its region's members by fancy index into the
    # normalized memmap. Members are spread across the whole file at a stride
    # of a few hundred rows, so on the external drive each row was its own
    # page fault: measured, 72 regions took 14 hours. One sequential pass
    # rewrites the vectors grouped by region -- each 64k-row chunk becomes at
    # most 256 contiguous runs -- and afterwards a region is one slice. Rows
    # are written in ascending global order within each region, which is the
    # order ``l1_order`` already holds, so ``sorted_mm[a:b]`` lines up with
    # ``l1_order[a:b]`` row for row.
    sorted_path = norm_dir / f"norm-sorted-{profile.version}.f16"
    ck_sorted = corpus.work / f"cluster-sorted-{profile.version}.npz"
    sorted_ok = (
        ck_sorted.is_file() and sorted_path.is_file()
        and sorted_path.stat().st_size == expected_bytes
        and str(np.load(ck_sorted)["corpus_hash"]) == corpus_hash
    )
    if sorted_ok:
        sorted_mm = np.memmap(sorted_path, dtype=np.float16, mode="r", shape=(n, profile.dim))
        log(f"  resumed region-sorted vectors from {sorted_path.name}")
    else:
        sorted_mm = np.memmap(sorted_path, dtype=np.float16, mode="w+", shape=(n, profile.dim))
        cursor = l1_offsets[:-1].copy()
        for start in range(0, n, 65_536):
            stop = min(start + 65_536, n)
            block = np.asarray(mm[start:stop])
            lab = l1_assign[start:stop]
            order = np.argsort(lab, kind="stable")
            lab_sorted = lab[order]
            runs = np.r_[0, np.flatnonzero(np.diff(lab_sorted)) + 1, len(lab_sorted)]
            for r0, r1 in pairwise(runs):
                region = int(lab_sorted[r0])
                k = int(r1 - r0)
                sorted_mm[cursor[region]:cursor[region] + k] = block[order[r0:r1]]
                cursor[region] += k
            report(30.0 + 2.0 * stop / n, "grouping vectors by region")
        sorted_mm.flush()
        if not np.array_equal(cursor, l1_offsets[1:]):
            raise RuntimeError("region-sorted rewrite did not fill every region exactly")
        np.savez_compressed(ck_sorted, corpus_hash=corpus_hash, n=n)
        sorted_mm = np.memmap(sorted_path, dtype=np.float16, mode="r", shape=(n, profile.dim))

    first_parent = 0
    if _checkpoint_matches(ck_l2, corpus_hash, fit_mode):
        saved = np.load(ck_l2)
        assignment[:] = saved["assignment"]
        cell_parent = saved["cell_parent"].astype(int).tolist()
        next_cell = int(saved["next_cell"])
        first_parent = int(saved["parent_done"]) + 1
        SIBLING_MERGES[:] = saved["sibling_merges"].astype(int).tolist()
        log(f"  resumed L2 fits from {ck_l2.name}: {first_parent} regions done, {next_cell} cells")

    def l2_checkpoint(parent_done: int) -> None:
        tmp = ck_l2.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, assignment=assignment, cell_parent=np.asarray(cell_parent, dtype=np.int32),
            next_cell=next_cell, parent_done=parent_done,
            sibling_merges=np.asarray(SIBLING_MERGES, dtype=np.int32),
            corpus_hash=corpus_hash, fit_mode=fit_mode,
        )
        tmp.replace(ck_l2)

    for parent in range(first_parent, profile.n_l1):
        members = l1_order[l1_offsets[parent]:l1_offsets[parent + 1]]
        if not len(members):
            if (parent + 1) % L2_CHECKPOINT_EVERY == 0:
                l2_checkpoint(parent)
            continue
        # Fitting scales with the cells being asked for, not with a flat cap: a
        # region splitting into 40 cells needs far more than one splitting into 4.
        l1_cap = (
            max(400_000, 2_000 * int(planned[parent])) if budgeted else 400_000
        )

        def _fit(
            vecs: np.ndarray, parent: int = parent
        ) -> tuple[np.ndarray, int, float, np.ndarray]:
            if budgeted:
                return _fixed_k_kmeans(vecs, int(planned[parent]), seed=SEED + parent)
            return _best_kmeans(
                vecs, profile.l2_k_min, profile.l2_k_max, seed=SEED + parent
            )

        lo, hi = int(l1_offsets[parent]), int(l1_offsets[parent + 1])
        block16 = np.asarray(sorted_mm[lo:hi])  # one contiguous read
        if weights is not None:
            # Fit on a content-weighted draw of the region (a long record may
            # appear twice, most one-liners not at all), then give every member
            # its nearest centre and repair the partition over all of them, so
            # no cell of the artifact is empty or negligible.
            draw = weighted_draw(weights[members], min(l1_cap, len(members)))
            _, _, score, fitted = _fit(block16[draw].astype(np.float32))
            region = block16.astype(np.float32)
            local = np.empty(len(members), dtype=np.int32)
            for start in range(0, len(members), 65_536):
                local[start:start + 65_536] = (
                    region[start:start + 65_536] @ fitted.T
                ).argmax(axis=1)
            local, local_centroids = _repair_and_merge(local, fitted, region)
            n_local = len(local_centroids)
            assignment[members] = local + next_cell
            del region
        elif len(members) <= l1_cap:
            local, n_local, score, local_centroids = _fit(
                block16.astype(np.float32)
            )
            assignment[members] = local + next_cell
        else:
            pos = np.linspace(0, len(members) - 1, l1_cap, dtype=np.int64)
            local, n_local, score, local_centroids = _fit(
                block16[pos].astype(np.float32)
            )
            assignment[members[pos]] = local + next_cell
            rest = np.setdiff1d(np.arange(len(members)), pos)
            for start in range(0, len(rest), 65_536):
                part = rest[start:start + 65_536]
                assignment[members[part]] = (
                    block16[part].astype(np.float32) @ local_centroids.T
                ).argmax(axis=1) + next_cell
        del block16
        cell_parent.extend([parent] * n_local)
        next_cell += n_local
        if budgeted:
            log(f"    L1 {parent:3d}: {len(members):,} rows → {n_local} cells "
                f"(budget {int(planned[parent])})")
        else:
            log(f"    L1 {parent:3d}: {len(members):,} rows → {n_local} cells "
                f"(best-k silhouette {score:.3f})")
        report(30.0 + 40.0 * (parent + 1) / profile.n_l1, "selecting and fitting L2 cells")
        if (parent + 1) % L2_CHECKPOINT_EVERY == 0 or parent + 1 == profile.n_l1:
            l2_checkpoint(parent)
    parents = np.asarray(cell_parent, dtype=np.int32)
    n_cells = len(parents)
    cluster_idx = kmeans_idx
    cluster_vecs = gather_rows(mm, cluster_idx)
    assignment_sample = assignment[cluster_idx]

    # Rebuild centroids from the full assignment. ``counts`` stays in records --
    # it is the reference density a scan is compared against -- while the
    # centroid is the content-weighted mean when the build is weighted.
    sums = np.zeros((n_cells, profile.dim), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for start in range(0, n, 65_536):
        stop = min(start + 65_536, n)
        vecs = np.asarray(mm[start:stop], dtype=np.float32)
        cells = assignment[start:stop]
        order = np.argsort(cells, kind="stable")
        ordered_cells = cells[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_cells)) + 1]
        unique = ordered_cells[starts]
        block = vecs[order].astype(np.float64)
        if weights is not None:
            block *= weights[start:stop][order][:, None]
        sums[unique] += np.add.reduceat(block, starts, axis=0)
        counts += np.bincount(cells, minlength=n_cells)
        report(70.0 + 25.0 * stop / n, "refining centroids from all records")
    centroids = _norm((sums / np.maximum(counts, 1)[:, None]).astype(np.float32))

    if profile.version == ATLAS_V3.version:
        # Per-cell payloads are multiplied by fourteen times as many cells, so
        # v2's per-cell sizes do not survive the move. numpy stores fixed-width
        # unicode at four bytes a character, which is where v2's 7.5 MB artifact
        # became 28.8 MB once unpacked: exemplar_texts alone was 22.7 MB of it,
        # and nothing under src/ ever reads that array. Keeping v2's 32x600 here
        # would cost 314 MB unpacked for a payload the runtime never touches.
        knot_count, proto_count = V3_KNOTS, V3_PROTOTYPES
        exemplar_count, exemplar_chars = V3_EXEMPLARS, V3_EXEMPLAR_CHARS
        term_count, cooc_count = V3_TERMS, V3_COOCCURRENCE
    else:
        knot_count = FULL_KNOTS if profile.version == ATLAS_V2.version else LITE_KNOTS
        proto_count = FULL_PROTOTYPES if profile.version == ATLAS_V2.version else LITE_PROTOTYPES
        exemplar_count = FULL_EXEMPLARS if profile.version == ATLAS_V2.version else LITE_EXEMPLARS
        exemplar_chars = FULL_EXEMPLAR_CHARS if profile.version == ATLAS_V2.version else LITE_EXEMPLAR_CHARS
        term_count = FULL_TERMS if profile.version == ATLAS_V2.version else LITE_TERMS
        cooc_count = FULL_COOCCURRENCE if profile.version == ATLAS_V2.version else LITE_COOCCURRENCE

    # Calibration / prototypes from the cluster sample to bound RAM.
    refs, _, _ = _calibration(cluster_vecs, assignment_sample, centroids, knot_count)
    support = counts.astype(np.int32)
    prototypes = _prototypes(cluster_vecs, assignment_sample, centroids, proto_count)

    items = lite_items if lite_items is not None else reservoir.items
    res_rows = np.asarray([item[0] for item in items], dtype=np.int64)
    if subset is None:
        valid = (res_rows >= 0) & (res_rows < n)
        res_local = res_rows[valid]
        keep_items = [
            item for item, keep in zip(items, valid.tolist(), strict=True) if keep
        ]
        res_texts = [item[1] for item in keep_items]
        res_sources = [item[4] for item in keep_items]
    else:
        inverse = {int(row): i for i, row in enumerate(take.tolist())}
        keep_items = [item for item in items if int(item[0]) in inverse]
        res_local = np.asarray([inverse[int(item[0])] for item in keep_items], dtype=np.int64)
        res_texts = [item[1] for item in keep_items]
        res_sources = [item[4] for item in keep_items]
    res_assign = assignment[res_local] if len(res_local) else np.zeros(0, np.int32)
    res_vecs = gather_rows(mm, res_local) if len(res_local) else np.zeros((0, profile.dim), np.float32)
    exemplars = _exemplars(res_texts, res_assign, res_vecs, centroids, exemplar_count, exemplar_chars) if res_texts else np.full((n_cells, exemplar_count), "", dtype=f"U{exemplar_chars}")
    terms = _terms(res_texts, res_assign, n_cells, term_count) if res_texts else [[] for _ in range(n_cells)]
    if n <= 2_000_000:
        source_ids = corpus.src_ids[:n] if take is None else corpus.src_ids[take]
        full_sources = [corpus.src_table[int(i)] for i in source_ids]
    else:
        full_sources = res_sources
    full_assign_for_cooc = assignment if n <= 2_000_000 else res_assign
    cooc_ids, cooc_scores = _cooccurrence(full_assign_for_cooc, full_sources, n_cells, cooc_count)

    l1_labels, l1_labels_source = curated_l1_labels(profile, corpus_hash)
    if l1_labels_source in {"missing", "profile-changed"}:
        l1_labels = automatic_l1_labels(terms, parents, profile.n_l1)
        l1_labels_source = "automatic-contrastive-terms"
    declaration = {
        **asdict(profile), "seed": SEED,
        "min_community_size": MIN_COMMUNITY, "encoder_weight_hash": encoder_hash,
        "corpus_hash": corpus_hash,
        "l2_method": (
            "sqrt_population_budget_kmeans+sibling_cosine_merge"
            if profile.l2_budget is not None else "per_l1_best_k_cosine_silhouette"
        ),
        "l2_sibling_merge_cosine": L2_SIBLING_MERGE_COSINE,
        "l2_sibling_merges": int(sum(SIBLING_MERGES)),
        "language_labels_source": "detected-per-record:dropoutt.langid",
        "language_probe": probe,
        "idf_strategy": "manifest-stratified-systematic",
        "idf_fit_records": idf_fit_records,
    }
    # Raw-text builds leave the declaration as it was, so rebuilding the v2 pair
    # does not move its pipeline hash for a policy it does not use.
    if encoder_input.active:
        declaration["encoder_input"] = encoder_input.declaration()
    if fit_mode != "rows":
        declaration["fit_weights"] = fit_mode
    region_labels, region_labels_source = curated_region_labels(profile, n_cells, corpus_hash)
    if region_labels is None:
        region_labels = contrastive_cell_labels(terms, n_cells)
        if region_labels_source == "profile-changed":
            region_labels_source = "contrastive:profile-changed"
        else:
            region_labels_source = "contrastive"
    meta = {
        "version": profile.version, "profile": asdict(profile),
        "embed_model": DEFAULT_MODEL,
        "pipeline_hash": pipeline_hash(declaration),
        "encoder_weight_hash": encoder_hash, "corpus_hash": corpus_hash,
        "n_regions": n_cells, "n_reference_records": int(counts.sum()),
        "n_l1": profile.n_l1, "pooling": profile.pooling,
        "l2": (
            {
                "method": "sqrt_population_budget",
                "k_min": profile.l2_k_min, "k_max": profile.l2_k_max,
                "budget": profile.l2_budget,
                "metric": None,
            }
            if profile.l2_budget is not None
            else {
                "method": "per_l1_best_k_cosine_silhouette",
                "k_min": profile.l2_k_min, "k_max": profile.l2_k_max,
                "budget": None,
                "metric": "cosine_silhouette",
            }
        ),
        "idf": {
            "strategy": "manifest-stratified-systematic",
            "fit_records": idf_fit_records,
        },
        # Evidence the map ships with. These sat in ``declaration`` -- which
        # only feeds the pipeline hash -- for one build, and that artifact
        # left the factory without its probe.
        "language_labels_source": "detected-per-record:dropoutt.langid",
        "language_probe": probe,
        "l2_sibling_merge_cosine": L2_SIBLING_MERGE_COSINE,
        "l2_sibling_merges": int(sum(SIBLING_MERGES)),
        "normalization": {
            "per_language": profile.version
            in (ATLAS_V2.version, ATLAS_V2_LITE.version, ATLAS_V3.version),
            "pca_k": profile.pca_k, "min_language_members": MIN_LANGUAGE_ROWS,
            "language_mean_fit": "streamed-all-rows",
            "calibration_draw": "language-axis-balanced",
            "calibration_rows": NORM_SAMPLE_ROWS,
            "lang_labels": lang_labels,
        },
        "user_resolution": "l2", "flat_cells": True, "l1_is_build_scaffold_only": True,
        "l1_labels": l1_labels,
        "l1_labels_source": l1_labels_source,
        "region_terms": [", ".join(value) for value in terms],
        "region_labels": region_labels,
        "region_labels_source": region_labels_source,
        "distance_knots": knot_count,
        "artifact_size_target_mb": (
            [15, 25] if profile.version == ATLAS_V3.version
            else [35, 50] if profile.version == ATLAS_V2.version else [2, 5]
        ),
    }
    if encoder_input.active:
        meta["encoder_input"] = encoder_input.declaration()
    if fit_mode != "rows":
        meta["fit_weights"] = fit_mode
    if region_labels_source.startswith("curated:"):
        kinds = curated_kinds(profile, n_cells, corpus_hash, level="l2")
        if kinds is not None:
            meta["region_kinds"] = kinds
    if l1_labels_source.startswith("curated:"):
        kinds = curated_kinds(profile, profile.n_l1, corpus_hash, level="l1")
        if kinds is not None:
            meta["l1_kinds"] = kinds
    l1_reference_size = np.zeros(profile.n_l1, dtype=np.int64)
    np.add.at(l1_reference_size, parents, counts)
    l1_child_count = np.bincount(parents, minlength=profile.n_l1).astype(np.int32)
    payload = dict(  # noqa: C408
        centroids=centroids, region_category=parents, l1_parent=parents,
        region_size=support, l1_centroids=l1_centroids,
        l1_size=l1_reference_size,
        l1_reference_size=l1_reference_size,
        l1_child_count=l1_child_count,
        coords=np.zeros((n_cells, 2), np.float32), norm_mean=mean, norm_pca=pca,
        norm_lang_means=lang_means, distance_refs=refs, distance_refs_support=support,
        distance_refs_reliable=support >= MIN_COMMUNITY,
        prototype_vectors=prototypes, exemplar_texts=exemplars,
        cooccurrence_ids=cooc_ids, cooccurrence_scores=cooc_scores,
        probe_coef=np.zeros((0, profile.dim), np.float32),
        probe_intercept=np.zeros(0, np.float32), probe_classes=np.zeros(0, np.int32),
        meta=np.asarray([json.dumps(meta)]),
    )
    if idf is not None:
        payload["idf_token_ids"], payload["idf_log_probs"] = idf
    if encoder_input.damps_phrases:
        payload["encoder_phrase_hashes"] = np.asarray(encoder_input.phrase_hashes, dtype=np.uint64)
    if profile.version == ATLAS_V2.version:
        payload["cell_similarity"] = (centroids @ centroids.T).astype(np.float16)
    np.savez_compressed(out, **payload)
    report(100.0, "artifact written")
    del mm
    path.unlink(missing_ok=True)
    del sorted_mm
    sorted_path.unlink(missing_ok=True)
    for stale in (ck_norm, ck_l1, ck_l2, ck_sorted):
        stale.unlink(missing_ok=True)
    return {
        "profile": profile, "assignment": assignment,
        "meta": meta, "artifact": out, "size_mb": out.stat().st_size / 1e6,
    }


def _corpus_gates(corpus, mass: float, input_logical: int, logical: int,
                  input_rows: int, n_sources: int, elapsed: float) -> dict:
    """Shared release gates.

    ``non_english_share`` is measured two ways because the floor is a byte
    share. ``NON_ENGLISH_FLOOR`` is ``1 - WEB_LANGUAGE_SHARES["en"]`` — 50.5% of
    the corpus *by bytes* — but the builder only ever compared it against a share
    of *rows*. English records are shorter than the corpus average, so the v2
    build reported 48.3% by rows against a byte floor and logged a gate failure
    on a corpus that met its plan at 51.3% by bytes. Both are recorded; the gate
    reads the one the floor is denominated in.
    """
    axis_hist = np.bincount(
        np.asarray(corpus.axis_ids[:corpus.n]), minlength=len(corpus.axis_table)
    )
    axis_counts = Counter(
        {name: int(axis_hist[i]) for i, name in enumerate(corpus.axis_table)}
    )
    axis_bytes = dict(getattr(corpus, "logical_by_axis", {}) or {})
    language_hist = np.bincount(
        np.asarray(corpus.lang_ids[:corpus.n]), minlength=len(corpus.lang_table)
    )
    english_rows = sum(
        int(language_hist[i])
        for i, name in enumerate(corpus.lang_table)
        if name in ("en", "unknown")
    )
    non_english_rows = 1.0 - english_rows / max(corpus.n, 1)
    counted = sum(int(v) for v in getattr(corpus, "logical_by_lang", {}).values())
    # A build resumed from a checkpoint written before the per-language ledger
    # existed only counts bytes for the sources this run ingested. That is a
    # share of part of the corpus, which would be worse than no answer, so it is
    # only trusted when it accounts for essentially all the retained bytes.
    if counted and logical and counted >= 0.99 * logical:
        non_english_bytes = 1.0 - _english_logical_bytes(corpus) / counted
        byte_share_complete = True
    else:
        non_english_bytes = non_english_rows
        byte_share_complete = False
    return {
        "input_logical_bytes": input_logical,
        "post_filter_logical_bytes": logical,
        "idf_observed_token_mass": mass,
        "non_english_share": non_english_bytes,
        "non_english_share_by_bytes": non_english_bytes if byte_share_complete else None,
        "non_english_byte_ledger_complete": byte_share_complete,
        "non_english_bytes_counted": counted,
        "non_english_share_by_rows": non_english_rows,
        "non_english_floor": NON_ENGLISH_FLOOR,
        # Byte-denominated, like the targets. The row floor this replaces
        # (target bytes // 4,000) failed the code axis at 100% of its byte plan
        # because source files run longer than 4,000 bytes, and would pass a
        # books axis of short rows well below plan. Half the byte target.
        "axis_floors": {
            axis: int(axis_bytes.get(axis, 0)) >= floor
            for axis, floor in AXIS_BYTE_FLOORS.items()
        },
        "axis_share_of_target": {
            axis: round(int(axis_bytes.get(axis, 0)) / max(target, 1), 4)
            for axis, target in AXIS_TARGET_BYTES.items()
        },
        "axis_rows": {axis: int(v) for axis, v in axis_counts.items()},
        "input_records": input_rows,
        "manifest_sources": n_sources,
        "elapsed_s": round(elapsed, 1),
    }


def _english_logical_bytes(corpus) -> int:
    """Retained post-filter UTF-8 bytes counted as English."""
    return sum(
        int(value)
        for lang, value in getattr(corpus, "logical_by_lang", {}).items()
        if lang in ("en", "unknown")
    )


def _finish_v3(args, built, corpus, mass, sources, ledger_path, manifest_path,
               manifest_digest, input_logical, logical, input_rows, progress, t0) -> int:
    """Write release notes and gates for the single-artifact v3 product."""
    gates = _corpus_gates(
        corpus, mass, input_logical, logical, input_rows, len(sources),
        time.time() - t0,
    )
    gates["atlas_v3_records"] = corpus.n
    notes = {
        "seed": SEED,
        "encoder_input": built["meta"].get("encoder_input"),
        "source_ledger": str(ledger_path),
        "manifest": str(manifest_path),
        "manifest_hash": manifest_digest,
        "gates": gates,
        "artifacts": {
            built["profile"].version: {
                "path": str(built["artifact"]),
                "size_mb": built["artifact"].stat().st_size / 1e6,
                "n_regions": built["meta"]["n_regions"],
                "n_l1": built["profile"].n_l1,
                "l2_budget": built["profile"].l2_budget,
            }
        },
    }
    (args.out_dir / "atlas-v3-release-notes.json").write_text(
        json.dumps(notes, indent=2) + "\n"
    )
    if hashlib.blake2b(manifest_path.read_bytes(), digest_size=16).hexdigest() != manifest_digest:
        raise SystemExit("corpus manifest changed during the build")
    progress.finish(retained_rows=corpus.n)
    size_mb = built["artifact"].stat().st_size / 1e6
    log(f"done in {time.time() - t0:.0f}s")
    log(f"  atlas-v3  {built['meta']['n_regions']} cells  {size_mb:.1f} MB "
        f"({built['profile'].n_l1} L1)")
    if gates["non_english_byte_ledger_complete"]:
        measured = f"{gates['non_english_share_by_bytes']:.1%} by bytes"
    else:
        measured = (
            f"{gates['non_english_share_by_rows']:.1%} by rows "
            f"(byte ledger incomplete on a resumed build)"
        )
    log(f"  non-English {measured}  IDF mass {mass:.4f}  rows {corpus.n:,}")
    if not args.allow_small_corpus:
        if mass < 0.99:
            log("warning: IDF token mass below 99%")
        if gates["non_english_share"] < NON_ENGLISH_FLOOR:
            log(f"warning: non-English share below {NON_ENGLISH_FLOOR:.1%}")
        missing = [axis for axis, ok in gates["axis_floors"].items() if not ok]
        if missing:
            log(f"warning: axis floors missed: {', '.join(missing)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--source-ledger", type=Path)
    parser.add_argument("--allow-small-corpus", action="store_true")
    parser.add_argument(
        "--fit-weights", choices=FIT_MODES, default="rows",
        help=(
            "how records weigh when atlas-v3 fits its centroids. 'rows' counts "
            "every record once; 'length' weighs each by the text the encoder reads, "
            "so a corpus fifth made of one-line prompts no longer claims a fifth of "
            "the cells; 'sqrt-length' is between the two. Placement and densities "
            "stay per record either way."
        ),
    )
    parser.add_argument(
        "--product",
        choices=("atlas-v2-pair", "atlas-v3"),
        default="atlas-v2-pair",
        help=(
            "atlas-v2-pair builds atlas-v2 and atlas-v2-lite as two separate, "
            "non-comparable coordinate systems. atlas-v3 builds one artifact "
            "whose L1 is a strict prefix of its L2, so a coarse report is an "
            "exact union of fine cells."
        ),
    )
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.source_ledger or args.cache / "source-ledger.json"
    sources, manifest = manifest_sources(args.cache)
    input_rows = int(manifest["totals"]["rows"])
    input_logical = int(manifest["totals"]["logical_bytes"])
    manifest_path = args.cache / "manifest.json"
    manifest_digest = hashlib.blake2b(
        manifest_path.read_bytes(), digest_size=16
    ).hexdigest()
    progress = BuildProgress(args.work / "build-progress.json", input_rows)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_DATASETS_MULTIPROCESSING_MAX_WORKERS", "0")
    log(
        f"atlas-v2 shared-corpus build  sources={len(sources)}  "
        f"input={input_rows:,} rows/{input_logical / (1024 ** 3):.2f} GiB  "
        f"work={args.work}"
    )
    embedder = load_embedder(DEFAULT_MODEL, out_dim=ATLAS_V2.dim)
    if embedder is None or embedder.dim < ATLAS_V2.dim:
        raise SystemExit("atlas-v2 requires the 256-column quantized encoder cache")
    # atlas-v3 reads text through the v1 encoder input policy; the v2 pair was
    # fitted on raw text and stays that way. The policy is bound before the IDF
    # pass, because the token counts SIF weights come from must be counts of the
    # text the encoder will actually read.
    policy = ENCODER_INPUT_V1 if args.product == "atlas-v3" else RAW_INPUT
    embedder = embedder.bind_input(policy)
    phrases_path = args.work / "encoder-phrases.npy"
    policy_path = args.work / "encoder-input.json"
    declared_policy = {key: value for key, value in policy.declaration().items()
                       if not key.startswith("phrase_count") and key != "phrase_digest"}
    if policy_path.is_file():
        previous = json.loads(policy_path.read_text(encoding="utf-8"))
        if previous != declared_policy:
            raise SystemExit(
                f"{args.work} was started with encoder input {previous}, not "
                f"{declared_policy}; use a fresh --work directory"
            )
    elif (args.work / "checkpoint.json").is_file() or (args.work / "idf-summary.json").is_file():
        if policy.active:
            raise SystemExit(
                f"{args.work} holds a build started before the encoder input policy; "
                "its token counts and vectors are of raw text. Use a fresh --work directory."
            )
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(declared_policy, indent=2) + "\n", encoding="utf-8")

    corpus = DiskCorpus(args.work, ATLAS_V2.dim)
    seen = Uint64Set(args.work / "seen.u64")
    reservoir = Reservoir.load(args.work / "reservoir.jsonl", RESERVOIR_FULL, SEED)
    lite_reservoir = Reservoir.load(args.work / "lite-reservoir.jsonl", RESERVOIR_LITE, SEED + 1)
    token_counts = np.zeros(embedder.vocab_size, dtype=np.int64)
    t0 = time.time()
    consumed, logical, saved_counts = corpus.load_checkpoint()
    if saved_counts is not None and saved_counts.shape == token_counts.shape:
        token_counts = saved_counts
    mapping, idf_tables, mass = _idf_from_counts(token_counts)
    if mapping and policy.phrase_weight != 1.0:
        if not phrases_path.is_file():
            raise SystemExit(f"resuming {args.work} needs {phrases_path.name}, which is missing")
        policy = policy.with_phrases(np.load(phrases_path))
        embedder = embedder.bind_input(policy)
    sif = embedder.bind_idf(mapping) if mapping else embedder
    idf_summary_path = args.work / "idf-summary.json"
    idf_fit_records = 0
    if idf_summary_path.is_file():
        idf_fit_records = int(
            json.loads(idf_summary_path.read_text(encoding="utf-8")).get("fit_records", 0)
        )
    rows_by_slug = {
        str(meta["slug"]): int(meta.get("rows", 0))
        for meta in manifest["sources"]
    }
    cumulative_rows = []
    running_rows = 0
    for source in sources:
        running_rows += rows_by_slug[source.slug]
        cumulative_rows.append(running_rows)

    if not mapping:
        log(f"phase 1: stratified SIF IDF sample ({IDF_SAMPLE_ROWS:,} records)")
        progress.update("idf", 0.0, detail="starting stratified IDF sample")
        idf_seen_path = args.work / "seen-idf.u64"
        idf_seen = Uint64Set(idf_seen_path, slots=IDF_HASH_SLOTS)

        def idf_source_progress(
            index: int, source: Source, source_rows: int, selected_rows: int
        ) -> None:
            expected = rows_by_slug[source.slug]
            prior = cumulative_rows[index - 2] if index > 1 else 0
            stage_pct = 100.0 * (prior + min(source_rows, expected)) / max(input_rows, 1)
            progress.update(
                "idf",
                0.15 * stage_pct,
                stage_pct=stage_pct,
                detail=f"{source.slug} ({source_rows:,}/{expected:,} rows)",
                retained_rows=selected_rows,
            )

        found_phrases: list[np.ndarray] | None = [] if policy.phrase_weight != 1.0 else None
        warmed = _scan_cache_for_idf(
            args.cache,
            sources,
            embedder,
            token_counts,
            idf_seen,
            rows_by_slug,
            source_progress=idf_source_progress,
            phrases=found_phrases,
        )
        idf_fit_records = warmed
        idf_seen.flush()
        del idf_seen
        gc.collect()
        idf_seen_path.unlink(missing_ok=True)
        mapping, idf_tables, mass = _idf_from_counts(token_counts)
        if warmed < IDF_WARMUP:
            log(f"  warning: IDF pass supplied only {warmed:,} unique texts")
        if found_phrases is not None:
            table = (
                np.unique(np.concatenate(found_phrases))
                if found_phrases else np.zeros(0, dtype=np.uint64)
            )
            np.save(phrases_path, table)
            policy = policy.with_phrases(table)
            embedder = embedder.bind_input(policy)
            log(f"  template phrases={len(table):,} across {len(sources)} sources")
        sif = embedder.bind_idf(mapping) if mapping else embedder
        idf_summary_path.write_text(
            json.dumps(
                {
                    "strategy": "manifest-stratified-systematic",
                    "target_records": IDF_SAMPLE_ROWS,
                    "fit_records": idf_fit_records,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        log(f"  IDF types={len(mapping):,} token-mass={mass:.4f}  {time.time() - t0:.0f}s")
    elif corpus.n:
        log(f"resume at {corpus.n:,} rows, {len(consumed)} sources already consumed")
        # Say so in the progress file at once. With ingest complete the next
        # report is the first clustering one, twenty minutes of I/O away, and
        # until then a monitor would show the previous run's final frame with
        # a dead pid and call this build stopped.
        resumed_rows = sum(rows_by_slug[slug] for slug in consumed)
        resumed_pct = 100.0 * resumed_rows / max(input_rows, 1)
        progress.update(
            "embedding", 15.0 + 0.45 * resumed_pct, stage_pct=resumed_pct,
            detail=f"resumed: {len(consumed)}/{len(sources)} sources already consumed",
            retained_rows=corpus.n,
        )

    log("phase 2: encode every manifest shard; source corpus remains read-only")
    base_logical = logical

    def checkpoint(added_logical: int) -> None:
        complete_rows = sum(rows_by_slug[slug] for slug in consumed)
        stage_pct = 100.0 * complete_rows / max(input_rows, 1)
        corpus.save_checkpoint(consumed, base_logical + added_logical, token_counts)
        seen.flush()
        reservoir.save(args.work / "reservoir.jsonl")
        lite_reservoir.save(args.work / "lite-reservoir.jsonl")
        progress.update(
            "embedding",
            15.0 + 0.45 * stage_pct,
            stage_pct=stage_pct,
            detail=f"{len(consumed)}/{len(sources)} shards",
            retained_rows=corpus.n,
        )

    def embedding_source_progress(index: int, source: Source, source_rows: int) -> None:
        expected = rows_by_slug[source.slug]
        prior = cumulative_rows[index - 2] if index > 1 else 0
        stage_pct = 100.0 * (prior + min(source_rows, expected)) / max(input_rows, 1)
        progress.update(
            "embedding",
            15.0 + 0.45 * stage_pct,
            stage_pct=stage_pct,
            detail=f"{source.slug} ({source_rows:,}/{expected:,} rows)",
            retained_rows=corpus.n,
        )

    pool = prepare_pool(policy) if not os.environ.get("ATLAS_BUILD_SERIAL") else None
    if pool is not None:
        log(f"  per-record stages in {PREPARE_WORKERS} worker processes")
    _, added_logical, consumed = _consume_cache(
        args.cache,
        sources,
        corpus,
        sif,
        seen,
        reservoir,
        lite_reservoir,
        None,
        consumed,
        checkpoint=checkpoint,
        source_progress=embedding_source_progress,
        prepare_pool=pool,
    )
    if pool is not None:
        pool.shutdown()
    logical = base_logical + added_logical
    checkpoint(added_logical)
    if len(consumed) != len(sources):
        raise SystemExit(f"only consumed {len(consumed)}/{len(sources)} manifest shards")
    if corpus.n < max(ATLAS_V2.n_l1, 2 * MIN_COMMUNITY) and not args.allow_small_corpus:
        raise SystemExit(f"post-ingest corpus is too small: {corpus.n} rows")
    corpus_hash = hashlib.blake2b(
        f"{manifest_digest}:{corpus.n}:{logical}:{','.join(corpus.src_table)}".encode(),
        digest_size=16,
    ).hexdigest()
    if args.product == "atlas-v3":
        log(f"phase 3: cluster atlas-v3 on all {corpus.n:,} retained records")
        fit_weights = None
        if args.fit_weights != "rows":
            if corpus.lengths is None:
                raise SystemExit(
                    f"--fit-weights {args.fit_weights} needs per-record lengths, and "
                    f"{args.work} was ingested before they were recorded"
                )
            fit_weights = fit_weights_for(
                corpus.lengths[:corpus.n], args.fit_weights, ATLAS_V3.max_chars
            )
            log(f"  fitting centroids by {args.fit_weights}")

        def v3_progress(stage_pct: float, detail: str) -> None:
            progress.update(
                "atlas-v3", 60.0 + 0.38 * stage_pct,
                stage_pct=stage_pct, detail=detail, retained_rows=corpus.n,
            )

        built = _build_from_memmap(
            ATLAS_V3, corpus, reservoir, idf_tables, embedder.weight_hash, corpus_hash,
            args.out_dir / "atlas-v3.npz",
            idf_fit_records=idf_fit_records,
            progress=v3_progress,
            encoder_input=policy,
            fit_weights=fit_weights,
            fit_mode=args.fit_weights,
        )
        return _finish_v3(
            args, built, corpus, mass, sources, ledger_path, manifest_path,
            manifest_digest, input_logical, logical, input_rows, progress, t0,
        )

    log(f"phase 3: cluster atlas-v2 on all {corpus.n:,} retained records")

    def full_progress(stage_pct: float, detail: str) -> None:
        progress.update(
            "atlas-v2",
            60.0 + 0.25 * stage_pct,
            stage_pct=stage_pct,
            detail=detail,
            retained_rows=corpus.n,
        )

    full = _build_from_memmap(
        ATLAS_V2, corpus, reservoir, idf_tables, embedder.weight_hash, corpus_hash,
        args.out_dir / "atlas-v2.npz",
        idf_fit_records=idf_fit_records,
        progress=full_progress,
    )
    log(f"phase 4: cluster atlas-v2-lite on the same {corpus.n:,} retained records")

    def lite_progress(stage_pct: float, detail: str) -> None:
        progress.update(
            "atlas-v2-lite",
            85.0 + 0.13 * stage_pct,
            stage_pct=stage_pct,
            detail=detail,
            retained_rows=corpus.n,
        )

    lite = _build_from_memmap(
        ATLAS_V2_LITE,
        corpus,
        lite_reservoir,
        idf_tables,
        embedder.weight_hash,
        corpus_hash,
        args.out_dir / "atlas-v2-lite.npz",
        idf_fit_records=idf_fit_records,
        progress=lite_progress,
    )
    progress.update("crosswalk", 98.0, detail="full-corpus population crosswalk")
    crosswalk = _crosswalk(
        full["assignment"],
        lite["assignment"],
        full["meta"]["n_regions"],
        lite["meta"]["n_regions"],
    )
    for build in (full, lite):
        with np.load(build["artifact"], allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        meta = json.loads(str(payload["meta"][0]))
        meta["crosswalk"] = crosswalk
        payload["meta"] = np.asarray([json.dumps(meta)])
        temporary = build["artifact"].with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(build["artifact"])

    axis_hist = np.bincount(np.asarray(corpus.axis_ids[:corpus.n]), minlength=len(corpus.axis_table))
    axis_counts = Counter({name: int(axis_hist[i]) for i, name in enumerate(corpus.axis_table)})
    language_hist = np.bincount(
        np.asarray(corpus.lang_ids[:corpus.n]), minlength=len(corpus.lang_table)
    )
    english = sum(
        int(language_hist[i])
        for i, name in enumerate(corpus.lang_table)
        if name in ("en", "unknown")
    )
    non_english = 1.0 - english / max(corpus.n, 1)
    gates = {
        "input_logical_bytes": input_logical,
        "post_filter_logical_bytes": logical,
        "idf_observed_token_mass": mass,
        "non_english_share": non_english,
        "axis_floors": {axis: axis_counts.get(axis, 0) >= floor for axis, floor in AXIS_FLOORS.items()},
        "atlas_v2_records": corpus.n,
        "atlas_v2_lite_records": corpus.n,
        "input_records": input_rows,
        "manifest_sources": len(sources),
        "elapsed_s": round(time.time() - t0, 1),
    }
    notes = {
        "seed": SEED,
        "source_ledger": str(ledger_path),
        "manifest": str(manifest_path),
        "manifest_hash": manifest_digest,
        "gates": gates,
        "artifacts": {
            build["profile"].version: {
                "path": str(build["artifact"]), "size_mb": build["artifact"].stat().st_size / 1e6,
                "n_regions": build["meta"]["n_regions"],
            } for build in (full, lite)
        },
    }
    (args.out_dir / "atlas-v2-release-notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    if hashlib.blake2b(manifest_path.read_bytes(), digest_size=16).hexdigest() != manifest_digest:
        raise SystemExit("corpus manifest changed during the build")
    progress.finish(retained_rows=corpus.n)
    log(f"done in {time.time() - t0:.0f}s")
    log(f"  atlas-v2      {full['meta']['n_regions']} cells  {full['artifact'].stat().st_size / 1e6:.1f} MB")
    log(f"  atlas-v2-lite {lite['meta']['n_regions']} cells  {lite['artifact'].stat().st_size / 1e6:.1f} MB")
    log(f"  non-English {non_english:.1%}  IDF mass {mass:.4f}  rows {corpus.n:,}")
    if not args.allow_small_corpus:
        if mass < 0.99:
            log("warning: IDF token mass below 99%")
        if non_english < NON_ENGLISH_FLOOR:
            log(f"warning: non-English share below {NON_ENGLISH_FLOOR:.1%}")
        missing = [axis for axis, ok in gates["axis_floors"].items() if not ok]
        if missing:
            log(f"warning: axis floors missed: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
