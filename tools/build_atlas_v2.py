#!/usr/bin/env python3
"""Build ``atlas-v2`` and ``atlas-v2-lite`` on a disk-constrained machine.

Sources are streamed one at a time, embedded, then deleted. The only durable
working set is a float16 embedding memmap plus compact row metadata. Cached
JSONL shards from an earlier fetch are consumed first and removed immediately.

Production builds target ``LOGICAL_BYTE_TARGET`` (40 GiB of retained UTF-8).
``--allow-small-corpus`` is the synthetic-fixture escape hatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_corpus
from atlas_sources import (
    AXIS_FLOORS,
    LOGICAL_BYTE_TARGET,
    NON_ENGLISH_FLOOR,
    SOURCES,
    Source,
)
from dropoutt.atlas.embed import DEFAULT_MODEL, TokenizedCorpus, load as load_embedder
from dropoutt.atlas.extract import extract_text
from dropoutt.atlas.pipeline import pipeline_hash
from dropoutt.atlas.profiles import ATLAS_V2, ATLAS_V2_LITE, AtlasProfile


SEED = 42
BLOCK = 2_048
MIN_COMMUNITY = 200
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
LITE_RECORDS = 130_000
FULL_EXEMPLARS, LITE_EXEMPLARS = 64, 24
FULL_EXEMPLAR_CHARS, LITE_EXEMPLAR_CHARS = 800, 384
FULL_PROTOTYPES, LITE_PROTOTYPES = 32, 12
FULL_KNOTS, LITE_KNOTS = 50, 33
FULL_COOCCURRENCE, LITE_COOCCURRENCE = 48, 24
FULL_TERMS, LITE_TERMS = 64, 32
RESERVOIR_FULL = 500_000
RESERVOIR_LITE = 250_000
IDF_WARMUP = 500_000
GROW_ROWS = 250_000
SOURCE_BUDGET_S = 1_800.0
SOURCE_STALL_S = 600.0


def log(message: str) -> None:
    print(message, flush=True)


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


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _intern(table: list[str], index: dict[str, int], value: str) -> int:
    slot = index.get(value)
    if slot is None:
        slot = len(table)
        index[value] = slot
        table.append(value)
    return slot


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


class DiskCorpus:
    """Growing float16 embedding store plus interned categorical columns."""

    def __init__(self, work: Path, dim: int) -> None:
        self.work = work
        self.dim = dim
        self.n = 0
        self.cap = 0
        self.emb: np.memmap | None = None
        self.axis_ids: np.memmap | None = None
        self.lang_ids: np.memmap | None = None
        self.src_ids: np.memmap | None = None
        self.axis_table: list[str] = []
        self.lang_table: list[str] = []
        self.src_table: list[str] = []
        self._axis_ix: dict[str, int] = {}
        self._lang_ix: dict[str, int] = {}
        self._src_ix: dict[str, int] = {}

    def _resize(self, cap: int) -> None:
        def grow(path: Path, dtype, width: int | None, previous):
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
        self.cap = cap

    def append(self, vectors: np.ndarray, axes: list[str], languages: list[str],
               sources: list[str]) -> int:
        batch = len(vectors)
        if self.n + batch > self.cap:
            self._resize(max(self.cap * 2 if self.cap else GROW_ROWS, self.n + batch))
        start = self.n
        assert self.emb is not None
        self.emb[start:start + batch] = np.asarray(vectors, dtype=np.float16)
        for i, (axis, language, source) in enumerate(zip(axes, languages, sources, strict=True)):
            self.axis_ids[start + i] = _intern(self.axis_table, self._axis_ix, axis)
            self.lang_ids[start + i] = _intern(self.lang_table, self._lang_ix, language)
            self.src_ids[start + i] = _intern(self.src_table, self._src_ix, source)
        self.n += batch
        return start

    def rows_f32(self, index: np.ndarray) -> np.ndarray:
        assert self.emb is not None
        return np.asarray(self.emb[index], dtype=np.float32)

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
        for array in (self.emb, self.axis_ids, self.lang_ids, self.src_ids):
            if array is not None:
                array.flush()


def fit_normalization(vectors: np.ndarray, languages: list[str],
                      profile: AtlasProfile) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit the declared global or per-language centering and all-but-top PCA."""
    x = np.asarray(vectors[:, :profile.dim], dtype=np.float32)
    global_mean = x.mean(axis=0).astype(np.float32)
    lang_labels: list[str] = []
    lang_means: list[np.ndarray] = []
    centered = x - global_mean
    if profile.version == ATLAS_V2.version:
        counts = Counter(languages)
        eligible = sorted(lang for lang, n in counts.items() if n >= 2_000 and lang not in ("unknown", "multi"))
        by_lang = {lang: x[np.asarray([v == lang for v in languages])].mean(axis=0) for lang in eligible}
        if by_lang:
            centered = x.copy()
            fallback = global_mean
            lookup = by_lang
            for i, lang in enumerate(languages):
                centered[i] -= lookup.get(lang, fallback)
            lang_labels = eligible
            lang_means = [by_lang[lang].astype(np.float32) for lang in eligible]
    if len(centered) > profile.pca_k + 10:
        from sklearn.utils.extmath import randomized_svd
        sample = centered if len(centered) <= 400_000 else centered[np.linspace(0, len(centered) - 1, 400_000, dtype=np.int64)]
        _, _, pca = randomized_svd(sample, n_components=profile.pca_k, random_state=SEED)
        pca = pca.astype(np.float32)
    else:
        pca = np.zeros((0, profile.dim), dtype=np.float32)
    stacked = np.vstack(lang_means).astype(np.float32) if lang_means else np.zeros((0, profile.dim), np.float32)
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


def _knn_graph(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """HNSW cosine graph. Optional dependencies are imported only at call time."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("atlas-v2 build needs faiss-cpu") from exc
    n, dim = vectors.shape
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    data = np.ascontiguousarray(vectors, dtype=np.float32)
    index.add(data)
    take = min(k + 1, n)
    scores, ids = index.search(data, take)
    neighbors = np.empty((n, min(k, n - 1)), dtype=np.int32)
    weights = np.empty_like(neighbors, dtype=np.float32)
    for row in range(n):
        kept = [(int(i), float(s)) for i, s in zip(ids[row], scores[row], strict=True) if int(i) != row]
        kept = kept[: neighbors.shape[1]]
        while len(kept) < neighbors.shape[1]:
            kept.append((row, 0.0))
        for col, (neighbor, score) in enumerate(kept):
            neighbors[row, col] = neighbor
            weights[row, col] = max(score, 0.0)
    return neighbors, weights


def _leiden(vectors: np.ndarray, k: int, gamma: float, recursive: bool) -> np.ndarray:
    """Cluster one L1 partition and optionally split its largest communities once."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise RuntimeError("atlas-v2 build needs igraph and leidenalg") from exc
    if len(vectors) < MIN_COMMUNITY * 2:
        return np.zeros(len(vectors), dtype=np.int32)
    neighbors, weights = _knn_graph(vectors, min(k, len(vectors) - 1))
    edges: dict[tuple[int, int], float] = {}
    for row in range(len(vectors)):
        for neighbor, weight in zip(neighbors[row], weights[row], strict=True):
            a, b = sorted((row, int(neighbor)))
            if a == b:
                continue
            edges[(a, b)] = max(edges.get((a, b), 0.0), float(weight))
    graph = ig.Graph(n=len(vectors), edges=list(edges), directed=False)
    graph.es["weight"] = [edges[edge] for edge in edges]
    labels = np.asarray(leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=gamma, seed=SEED,
    ).membership, dtype=np.int32)
    labels = merge_small_communities(labels, vectors, neighbors, weights)
    if recursive:
        next_label = int(labels.max()) + 1
        for cell in np.unique(labels):
            members = np.flatnonzero(labels == cell)
            if len(members) < 4_000:
                continue
            child = _leiden(vectors[members], k, gamma, recursive=False)
            if child.max() > 0:
                labels[members] = child + next_label
                next_label += int(child.max()) + 1
        labels = merge_small_communities(labels, vectors, neighbors, weights)
    return labels


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
        local = _leiden(vectors[members], profile.knn_k, profile.leiden_gamma,
                        profile.version == ATLAS_V2.version)
        cell_assign[members] = local + next_cell
        cell_parent.extend([parent] * (int(local.max()) + 1))
        next_cell += int(local.max()) + 1
        log(f"    L1 {parent:3d}: {len(members):,} rows → {int(local.max()) + 1} cells")
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


def _prepare_batch(raw_texts: list[str], axes: list[str], seen: set[bytes]) -> tuple[list[str], list[int]]:
    kept_texts: list[str] = []
    kept_index: list[int] = []
    for i, (raw, axis) in enumerate(zip(raw_texts, axes, strict=True)):
        text, _ = extract_text(raw, detected_format="code" if axis == "code" else None)
        text = text[:ATLAS_V2.max_chars]
        if len(text) < 80:
            continue
        digest = hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest()
        if digest in seen:
            continue
        seen.add(digest)
        kept_texts.append(text)
        kept_index.append(i)
    return kept_texts, kept_index


def _count_tokens(embedder, texts: list[str], counts: np.ndarray) -> None:
    ids, _ = _tokenize(embedder, texts, ATLAS_V2.max_tokens)
    if ids.size:
        np.add.at(counts, ids, 1)


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
) -> int:
    if not texts:
        return 0
    if token_counts is not None:
        _count_tokens(embedder, texts, token_counts)
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
    start = corpus.append(vectors, axes, languages, sources)
    for offset, (text, axis, language, source) in enumerate(zip(texts, axes, languages, sources, strict=True)):
        row = start + offset
        reservoir.offer(row, text[:FULL_EXEMPLAR_CHARS], axis, language, source)
        lite_reservoir.offer(row, text[:ATLAS_V2_LITE.max_chars], axis, language, source)
    return len(texts)


def _source_language(src: Source) -> str:
    lang = src.lang
    if lang.startswith("code:"):
        return "en"
    return lang


def _scan_cache_for_idf(cache: Path, embedder, counts: np.ndarray, seen: set[bytes]) -> int:
    """Tokenize existing shards once so SIF has a frozen IDF before encoding."""
    scanned = 0
    for src in SOURCES:
        shard = fetch_corpus.cached_shard_path(cache, src.slug)
        if not shard.is_file():
            continue
        batch: list[str] = []
        axes: list[str] = []
        for text in fetch_corpus.iter_cached_texts(cache, src.slug, ATLAS_V2.max_chars):
            batch.append(text)
            axes.append(src.axis)
            if len(batch) >= BLOCK:
                kept, _ = _prepare_batch(batch, axes, seen)
                if kept:
                    _count_tokens(embedder, kept, counts)
                    scanned += len(kept)
                batch, axes = [], []
        if batch:
            kept, _ = _prepare_batch(batch, axes, seen)
            if kept:
                _count_tokens(embedder, kept, counts)
                scanned += len(kept)
        log(f"  idf {src.slug[:52]:<52} running types={int((counts > 0).sum()):,}")
    return scanned


def _consume_cache(
    cache: Path,
    corpus: DiskCorpus,
    embedder,
    seen: set[bytes],
    reservoir: Reservoir,
    lite_reservoir: Reservoir,
    token_counts: np.ndarray,
) -> tuple[int, int, set[str]]:
    """Encode cached shards and delete each directory immediately afterwards."""
    rows = 0
    logical = 0
    consumed: set[str] = set()
    for src in SOURCES:
        shard = fetch_corpus.cached_shard_path(cache, src.slug)
        if not shard.is_file():
            continue
        batch_text: list[str] = []
        batch_axis: list[str] = []
        batch_lang: list[str] = []
        batch_src: list[str] = []
        language = _source_language(src)
        for text in fetch_corpus.iter_cached_texts(cache, src.slug, ATLAS_V2.max_chars):
            batch_text.append(text)
            batch_axis.append(src.axis)
            batch_lang.append(language)
            batch_src.append(src.slug)
            if len(batch_text) >= BLOCK:
                kept, index = _prepare_batch(batch_text, batch_axis, seen)
                if kept:
                    logical += sum(len(t.encode("utf-8")) for t in kept)
                    rows += _ingest_prepared(
                        corpus, embedder, kept,
                        [batch_axis[i] for i in index],
                        [batch_lang[i] for i in index],
                        [batch_src[i] for i in index],
                        reservoir, lite_reservoir, token_counts,
                    )
                batch_text, batch_axis, batch_lang, batch_src = [], [], [], []
        if batch_text:
            kept, index = _prepare_batch(batch_text, batch_axis, seen)
            if kept:
                logical += sum(len(t.encode("utf-8")) for t in kept)
                rows += _ingest_prepared(
                    corpus, embedder, kept,
                    [batch_axis[i] for i in index],
                    [batch_lang[i] for i in index],
                    [batch_src[i] for i in index],
                    reservoir, lite_reservoir, token_counts,
                )
        fetch_corpus.delete_cached_source(cache, src.slug)
        consumed.add(src.slug)
        log(f"  used  {src.slug[:52]:<52} corpus={corpus.n:,}  {logical / (1024 ** 3):.2f} GiB")
        corpus.flush()
    return rows, logical, consumed


def _stream_source(
    src: Source,
    corpus: DiskCorpus,
    embedder,
    seen: set[bytes],
    reservoir: Reservoir,
    lite_reservoir: Reservoir,
    token_counts: np.ndarray,
    *,
    row_limit: int,
    byte_limit: int | None,
    hf_home: Path,
) -> tuple[int, int]:
    rows = 0
    logical = 0
    batch_text: list[str] = []
    batch_axis: list[str] = []
    batch_lang: list[str] = []
    batch_src: list[str] = []
    language = _source_language(src)
    remaining_rows = row_limit
    remaining_bytes = byte_limit
    try:
        for text in fetch_corpus.iter_streamed_texts(
            src, budget=SOURCE_BUDGET_S, stall=SOURCE_STALL_S,
            max_chars=ATLAS_V2.max_chars, row_limit=remaining_rows,
            byte_limit=remaining_bytes,
        ):
            batch_text.append(text)
            batch_axis.append(src.axis)
            batch_lang.append(language)
            batch_src.append(src.slug)
            if len(batch_text) < BLOCK:
                continue
            kept, index = _prepare_batch(batch_text, batch_axis, seen)
            if kept:
                added_bytes = sum(len(t.encode("utf-8")) for t in kept)
                logical += added_bytes
                rows += _ingest_prepared(
                    corpus, embedder, kept,
                    [batch_axis[i] for i in index],
                    [batch_lang[i] for i in index],
                    [batch_src[i] for i in index],
                    reservoir, lite_reservoir, token_counts,
                )
            batch_text, batch_axis, batch_lang, batch_src = [], [], [], []
            if byte_limit is not None and logical >= byte_limit:
                break
            if rows >= row_limit:
                break
        if batch_text and (byte_limit is None or logical < byte_limit) and rows < row_limit:
            kept, index = _prepare_batch(batch_text, batch_axis, seen)
            if kept:
                logical += sum(len(t.encode("utf-8")) for t in kept)
                rows += _ingest_prepared(
                    corpus, embedder, kept,
                    [batch_axis[i] for i in index],
                    [batch_lang[i] for i in index],
                    [batch_src[i] for i in index],
                    reservoir, lite_reservoir, token_counts,
                )
    finally:
        fetch_corpus.wipe_tree(hf_home)
        hf_home.mkdir(parents=True, exist_ok=True)
    return rows, logical


def _build_from_memmap(
    profile: AtlasProfile,
    corpus: DiskCorpus,
    reservoir: Reservoir,
    idf: tuple[np.ndarray, np.ndarray] | None,
    encoder_hash: str,
    corpus_hash: str,
    out: Path,
    *,
    subset: np.ndarray | None = None,
    lite_items: list[tuple[int, str, str, str, str]] | None = None,
) -> dict:
    if subset is None:
        n = corpus.n
        take = np.arange(n, dtype=np.int64)
        sample_idx = take if n <= 400_000 else np.linspace(0, n - 1, 400_000, dtype=np.int64)
        sample = corpus.rows_f32(sample_idx)[:, :profile.dim]
        sample_langs = [corpus.lang_table[int(i)] for i in corpus.lang_ids[sample_idx]]
    else:
        take = subset
        n = len(take)
        sample_idx = take if n <= 400_000 else take[np.linspace(0, n - 1, 400_000, dtype=np.int64)]
        sample = corpus.rows_f32(sample_idx)[:, :profile.dim]
        sample_langs = [corpus.lang_table[int(i)] for i in corpus.lang_ids[sample_idx]]

    mean, pca, lang_means, lang_labels = fit_normalization(sample, sample_langs, profile)
    path = corpus.work / f"norm-{profile.version}.f16"
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n, profile.dim))
    for start in range(0, n, 32_768):
        stop = min(start + 32_768, n)
        rows = take[start:stop]
        raw = corpus.rows_f32(rows)[:, :profile.dim]
        langs = [corpus.lang_table[int(i)] for i in corpus.lang_ids[rows]]
        mm[start:stop] = apply_normalization(raw, mean, pca, langs, lang_labels, lang_means)
    mm.flush()

    from sklearn.cluster import MiniBatchKMeans
    if n < profile.n_l1:
        raise ValueError(f"{profile.version} needs at least {profile.n_l1} records")
    kmeans_idx = np.linspace(0, n - 1, min(n, 500_000), dtype=np.int64)
    kmeans = MiniBatchKMeans(
        n_clusters=profile.n_l1, random_state=SEED, batch_size=16_384, n_init=3, max_iter=300,
    ).fit(np.asarray(mm[kmeans_idx], dtype=np.float32))
    l1_centroids = _norm(kmeans.cluster_centers_.astype(np.float32))
    l1_assign = np.empty(n, dtype=np.int32)
    for start in range(0, n, 65_536):
        stop = min(start + 65_536, n)
        l1_assign[start:stop] = (np.asarray(mm[start:stop], dtype=np.float32) @ l1_centroids.T).argmax(axis=1)

    assignment = np.empty(n, dtype=np.int32)
    cell_parent: list[int] = []
    next_cell = 0
    l1_cap = 200_000
    for parent in range(profile.n_l1):
        members = np.flatnonzero(l1_assign == parent)
        if not len(members):
            continue
        if len(members) <= l1_cap:
            local_vecs = np.asarray(mm[members], dtype=np.float32)
            local = _leiden(local_vecs, profile.knn_k, profile.leiden_gamma,
                            profile.version == ATLAS_V2.version)
            assignment[members] = local + next_cell
            n_local = int(local.max()) + 1
        else:
            pick = members[np.linspace(0, len(members) - 1, l1_cap, dtype=np.int64)]
            local_vecs = np.asarray(mm[pick], dtype=np.float32)
            local = _leiden(local_vecs, profile.knn_k, profile.leiden_gamma,
                            profile.version == ATLAS_V2.version)
            n_local = int(local.max()) + 1
            local_centroids = _norm(np.vstack([
                local_vecs[local == cell].mean(axis=0) for cell in range(n_local)
            ]).astype(np.float32))
            assignment[pick] = local + next_cell
            rest = np.setdiff1d(members, pick)
            for start in range(0, len(rest), 65_536):
                part = rest[start:start + 65_536]
                assignment[part] = (np.asarray(mm[part], dtype=np.float32) @ local_centroids.T).argmax(axis=1) + next_cell
        cell_parent.extend([parent] * n_local)
        next_cell += n_local
        log(f"    L1 {parent:3d}: {len(members):,} rows → {n_local} cells")
    parents = np.asarray(cell_parent, dtype=np.int32)
    n_cells = len(parents)
    cluster_idx = kmeans_idx
    cluster_vecs = np.asarray(mm[cluster_idx], dtype=np.float32)
    assignment_sample = assignment[cluster_idx]

    # Rebuild centroids from the full assignment.
    sums = np.zeros((n_cells, profile.dim), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    for start in range(0, n, 65_536):
        stop = min(start + 65_536, n)
        vecs = np.asarray(mm[start:stop], dtype=np.float32)
        cells = assignment[start:stop]
        for local, cell in enumerate(cells.tolist()):
            sums[cell] += vecs[local]
            counts[cell] += 1
    centroids = _norm((sums / np.maximum(counts, 1)[:, None]).astype(np.float32))

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
        local_pos = {int(row): i for i, row in enumerate(take.tolist())}
        res_local = np.asarray([local_pos[int(row)] for row in res_rows if int(row) in local_pos], dtype=np.int64)
        res_texts = [item[1] for item in items if int(item[0]) in local_pos]
        res_sources = [item[4] for item in items if int(item[0]) in local_pos]
    else:
        inverse = {int(row): i for i, row in enumerate(take.tolist())}
        keep_items = [item for item in items if int(item[0]) in inverse]
        res_local = np.asarray([inverse[int(item[0])] for item in keep_items], dtype=np.int64)
        res_texts = [item[1] for item in keep_items]
        res_sources = [item[4] for item in keep_items]
    res_assign = assignment[res_local] if len(res_local) else np.zeros(0, np.int32)
    res_vecs = np.asarray(mm[res_local], dtype=np.float32) if len(res_local) else np.zeros((0, profile.dim), np.float32)
    exemplars = _exemplars(res_texts, res_assign, res_vecs, centroids, exemplar_count, exemplar_chars) if res_texts else np.full((n_cells, exemplar_count), "", dtype=f"U{exemplar_chars}")
    terms = _terms(res_texts, res_assign, n_cells, term_count) if res_texts else [[] for _ in range(n_cells)]
    full_sources = [corpus.src_table[int(i)] for i in corpus.src_ids[take]] if n <= 2_000_000 else res_sources
    full_assign_for_cooc = assignment if n <= 2_000_000 else res_assign
    cooc_ids, cooc_scores = _cooccurrence(full_assign_for_cooc, full_sources, n_cells, cooc_count)

    declaration = {
        **asdict(profile), "seed": SEED,
        "hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION, "efSearch": HNSW_EF_SEARCH},
        "min_community_size": MIN_COMMUNITY, "encoder_weight_hash": encoder_hash,
        "corpus_hash": corpus_hash, "l2_method": "leiden_knn_recursive",
    }
    meta = {
        "version": profile.version, "profile": asdict(profile),
        "pipeline_hash": pipeline_hash(declaration),
        "encoder_weight_hash": encoder_hash, "corpus_hash": corpus_hash,
        "n_regions": n_cells, "n_reference_records": int(counts.sum()),
        "n_l1": profile.n_l1, "pooling": profile.pooling,
        "leiden": {
            "gamma": profile.leiden_gamma, "k": profile.knn_k,
            "min_community_size": MIN_COMMUNITY,
            "recursive_split_large_communities": profile.version == ATLAS_V2.version,
        },
        "hnsw": declaration["hnsw"],
        "normalization": {
            "per_language": profile.version == ATLAS_V2.version,
            "pca_k": profile.pca_k, "min_language_members": 2_000,
            "lang_labels": lang_labels,
        },
        "user_resolution": "l2", "flat_cells": True, "l1_is_build_scaffold_only": True,
        "region_terms": [", ".join(value) for value in terms],
        "distance_knots": knot_count,
        "artifact_size_target_mb": [25, 40] if profile.version == ATLAS_V2.version else [2, 4],
    }
    payload = dict(
        centroids=centroids, region_category=parents, l1_parent=parents,
        region_size=support, l1_centroids=l1_centroids,
        l1_size=np.bincount(parents, minlength=profile.n_l1).astype(np.int32),
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
    if profile.version == ATLAS_V2.version:
        payload["cell_similarity"] = (centroids @ centroids.T).astype(np.float16)
    np.savez_compressed(out, **payload)
    del mm
    path.unlink(missing_ok=True)
    return {
        "profile": profile, "assignment": assignment, "take": take,
        "meta": meta, "artifact": out, "size_mb": out.stat().st_size / 1e6,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".atlas-cache")
    parser.add_argument("--work", type=Path, default=ROOT / ".atlas-work")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "src" / "dropoutt" / "data" / "atlas")
    parser.add_argument("--source-ledger", type=Path)
    parser.add_argument("--target-logical-bytes", type=int, default=LOGICAL_BYTE_TARGET)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--allow-small-corpus", action="store_true")
    args = parser.parse_args()
    from atlas_sources import BASELINE_SCALE
    scale = BASELINE_SCALE if args.scale is None else args.scale
    target_bytes = args.target_logical_bytes
    args.work.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.source_ledger or args.work / "source-ledger.json"
    hf_home = args.work / "hf"

    log(f"atlas-v2 stream build  target={target_bytes / (1024 ** 3):.1f} GiB  scale={scale}  work={args.work}")
    embedder = load_embedder(DEFAULT_MODEL, out_dim=ATLAS_V2.dim)
    if embedder is None or embedder.dim < ATLAS_V2.dim:
        raise SystemExit("atlas-v2 requires the 256-column quantized encoder cache")

    corpus = DiskCorpus(args.work, ATLAS_V2.dim)
    seen: set[bytes] = set()
    reservoir = Reservoir(RESERVOIR_FULL, SEED)
    lite_reservoir = Reservoir(RESERVOIR_LITE, SEED + 1)
    token_counts = np.zeros(embedder.vocab_size, dtype=np.int64)
    t0 = time.time()

    idf_seen: set[bytes] = set()
    log("phase 1: IDF from any cached shards")
    warmed = _scan_cache_for_idf(args.cache, embedder, token_counts, idf_seen)
    mapping, idf_tables, mass = _idf_from_counts(token_counts)
    if warmed < IDF_WARMUP:
        log(f"  cache only supplied {warmed:,} unique texts; SIF will refine during ingest")
    sif = embedder.bind_idf(mapping) if mapping else embedder
    log(f"  IDF types={len(mapping):,} token-mass={mass:.4f}  {time.time() - t0:.0f}s")

    log("phase 2: encode cached shards and delete them")
    # Re-run exact-hash against a fresh set so phase 1's seen does not hide rows.
    seen = set()
    _, logical, consumed = _consume_cache(
        args.cache, corpus, sif, seen, reservoir, lite_reservoir, token_counts,
    )
    mapping, idf_tables, mass = _idf_from_counts(token_counts)
    sif = embedder.bind_idf(mapping) if mapping else sif
    fetch_corpus.wipe_tree(args.cache)
    args.cache.mkdir(parents=True, exist_ok=True)
    log(f"  after cache: {corpus.n:,} rows, {logical / (1024 ** 3):.2f} GiB, IDF mass={mass:.4f}")

    log("phase 3: stream remaining sources one at a time")
    with fetch_corpus.isolate_hf_home(hf_home):
        for src in SOURCES:
            if logical >= target_bytes:
                break
            if src.slug in consumed:
                continue
            row_limit = max(50, int(src.target * scale))
            remaining = target_bytes - logical
            added_rows, added_bytes = _stream_source(
                src, corpus, sif, seen, reservoir, lite_reservoir, token_counts,
                row_limit=row_limit, byte_limit=remaining, hf_home=hf_home,
            )
            logical += added_bytes
            log(f"  stream {src.slug[:52]:<52} +{added_rows:,}  total {corpus.n:,}  {logical / (1024 ** 3):.2f} GiB")
            corpus.flush()
            if corpus.n and corpus.n % 50_000 < BLOCK:
                mapping, idf_tables, mass = _idf_from_counts(token_counts)
                if mapping:
                    sif = embedder.bind_idf(mapping)

        if logical < target_bytes:
            log("phase 3b: FineWeb-2 supplemental to the byte target")
            try:
                extras = list(fetch_corpus.iter_supplemental_sources())
            except Exception as exc:
                log(f"  supplemental listing failed: {exc}")
                extras = []
            for src in extras:
                if logical >= target_bytes:
                    break
                remaining = target_bytes - logical
                added_rows, added_bytes = _stream_source(
                    src, corpus, sif, seen, reservoir, lite_reservoir, token_counts,
                    row_limit=10**9, byte_limit=remaining, hf_home=hf_home,
                )
                logical += added_bytes
                log(f"  supp  {src.slug[:52]:<52} +{added_rows:,}  total {corpus.n:,}  {logical / (1024 ** 3):.2f} GiB")
                corpus.flush()

    fetch_corpus.wipe_tree(hf_home)
    mapping, idf_tables, mass = _idf_from_counts(token_counts)
    if corpus.n < max(ATLAS_V2.n_l1, 2 * MIN_COMMUNITY) and not args.allow_small_corpus:
        raise SystemExit(f"post-ingest corpus is too small: {corpus.n} rows")
    if logical < target_bytes and not args.allow_small_corpus:
        log(f"warning: only {logical:,} logical bytes of {target_bytes:,}; clustering what we have")

    languages = corpus.languages()
    axes = corpus.axes()
    corpus_hash = hashlib.blake2b(
        f"{corpus.n}:{logical}:{','.join(corpus.src_table)}".encode(), digest_size=16
    ).hexdigest()
    log(f"phase 4: cluster atlas-v2 on {corpus.n:,} × 256  ({time.time() - t0:.0f}s elapsed)")
    full = _build_from_memmap(
        ATLAS_V2, corpus, reservoir, idf_tables, embedder.weight_hash, corpus_hash,
        args.out_dir / "atlas-v2.npz",
    )

    lite_axes = [item[2] for item in lite_reservoir.items]
    lite_langs = [item[3] for item in lite_reservoir.items]
    lite_idx_in_res = stratified_indices(lite_axes, lite_langs, min(LITE_RECORDS, len(lite_reservoir.items)))
    lite_items = [lite_reservoir.items[i] for i in lite_idx_in_res]
    lite_rows = np.asarray([item[0] for item in lite_items], dtype=np.int64)
    log(f"phase 5: cluster atlas-v2-lite on {len(lite_rows):,} rows")
    lite = _build_from_memmap(
        ATLAS_V2_LITE, corpus, reservoir, None, embedder.weight_hash, corpus_hash,
        args.out_dir / "atlas-v2-lite.npz",
        subset=lite_rows, lite_items=lite_items,
    )

    # Crosswalk needs full assignment at lite row positions.
    full_at_lite = full["assignment"][lite_rows]
    crosswalk = _crosswalk(full_at_lite, lite["assignment"], full["meta"]["n_regions"], lite["meta"]["n_regions"])
    for build in (full, lite):
        with np.load(build["artifact"], allow_pickle=False) as data:
            payload = {name: data[name] for name in data.files}
        meta = json.loads(str(payload["meta"][0]))
        meta["crosswalk"] = crosswalk
        payload["meta"] = np.asarray([json.dumps(meta)])
        temporary = build["artifact"].with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(build["artifact"])

    axis_counts = Counter(axes)
    non_english = float(np.mean([value not in ("en", "unknown") for value in languages])) if languages else 0.0
    gates = {
        "logical_bytes": logical,
        "idf_observed_token_mass": mass,
        "non_english_share": non_english,
        "axis_floors": {axis: axis_counts.get(axis, 0) >= floor for axis, floor in AXIS_FLOORS.items()},
        "lite_records": len(lite_rows),
        "n_records": corpus.n,
        "elapsed_s": round(time.time() - t0, 1),
    }
    notes = {
        "seed": SEED, "source_ledger": str(ledger_path), "gates": gates,
        "artifacts": {
            build["profile"].version: {
                "path": str(build["artifact"]), "size_mb": build["artifact"].stat().st_size / 1e6,
                "n_regions": build["meta"]["n_regions"],
            } for build in (full, lite)
        },
    }
    (args.out_dir / "atlas-v2-release-notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    fetch_corpus.write_source_ledger(
        ledger_path,
        [{"slug": slug, "rows": int((np.asarray(corpus.src_ids[:corpus.n]) == i).sum()),
          "logical_bytes": 0, "status": "streamed", "source_role": "baseline"}
         for i, slug in enumerate(corpus.src_table)],
        {"scale": scale, "target_logical_bytes": target_bytes, "streamed": True},
    )
    log(f"done in {time.time() - t0:.0f}s")
    log(f"  atlas-v2      {full['meta']['n_regions']} cells  {full['artifact'].stat().st_size / 1e6:.1f} MB")
    log(f"  atlas-v2-lite {lite['meta']['n_regions']} cells  {lite['artifact'].stat().st_size / 1e6:.1f} MB")
    log(f"  non-English {non_english:.1%}  IDF mass {mass:.4f}  rows {corpus.n:,}")
    if not args.allow_small_corpus:
        if mass < 0.99:
            log("warning: IDF token mass below 99%")
        if non_english < NON_ENGLISH_FLOOR:
            log("warning: non-English share below 30%")
        missing = [axis for axis, ok in gates["axis_floors"].items() if not ok]
        if missing:
            log(f"warning: axis floors missed: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
