#!/usr/bin/env python3
"""Build atlas-lite-v3 from the on-disk reference corpus.

Reads the cache written by ``tools/fetch_corpus.py``; never touches the network
except to load the encoder weights. Everything large lives in a memory-mapped
working directory, so a 2M-record corpus does not have to fit in RAM.

    extract → dedup(near-exact) → tokenize → idf → embed → dedup(semantic)
            → normalize → cluster L1 → cluster L2 → assign → calibrate → label

What changed against v2, and why:

**Coarse and fine can no longer disagree.** v2 derived the L1 label from a
separate arg-max over L1 centroids while the fine cell came from an arg-max over
L2 centroids. On v2's own shipped reference records those two answers differed
for a quarter of them, so drilling down could flip the coarse answer — the exact
failure "lite is an exact prefix of the hierarchy" is supposed to make
impossible. Here the fine cell is assigned and the coarse label is *its parent*,
by construction.

**The number of children is measured, not declared.** v2 gave every one of 50
regions exactly 20 children whether or not that region had 20 things in it.
Here each parent is fitted at every k in 4..24, scored by silhouette, and the
global budget of 800 cells is handed out by marginal gain — a region with one
subject keeps four children and a region with real internal structure gets
twenty.

**Language is a nuisance parameter, not a topic.** v2's L1 regions included
"Turkish television and radio" and "Spanish-language server documentation":
language, not subject, and the scan already detects language separately. Two
normalizations are fitted — global mean removal and per-language mean removal —
and the probe set decides which ships. If per-language centering does not
actually put the same topic in the same cell across Turkish, English, Arabic and
Chinese, it does not ship.

**The tables it ships are the tables it reads.** v2 shipped per-cell distance
references, radial prototypes and a co-occurrence graph that no code path ever
opened. The coarse-resolution correction table the spec calls mandatory did not
exist at all. Both are fixed here, and the artifact-size floor that rewarded
shipping unread arrays is gone.

Usage:
    python tools/fetch_corpus.py --cache .atlas-cache --workers 10
    python tools/build_atlas.py  --cache .atlas-cache
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from atlas_sources import (  # noqa: E402
    AXIS_FLOORS,
    CODE_LANGUAGE_FLOOR,
    NON_ENGLISH_FLOOR,
    SOURCES,
)
from dropoutt.atlas.apply import Atlas  # noqa: E402
from dropoutt.atlas.embed import DEFAULT_MODEL, TokenizedCorpus  # noqa: E402
from dropoutt.atlas.embed import load as load_embedder  # noqa: E402
from dropoutt.atlas.extract import extract_text  # noqa: E402
from dropoutt.atlas.normalize import EMBED_DIM, SIF_A  # noqa: E402
from dropoutt.atlas.pipeline import PIPELINE_VERSION, pipeline_hash  # noqa: E402

SEED = 20260802
VERSION = "atlas-lite-v3"

N_L1 = 48
L2_K_MIN = 4
L2_K_MAX = 24
L2_BUDGET = 800
#: A child cell fitted on fewer members than this cannot be calibrated, so the
#: k-search is capped by member count before it is capped by the budget.
MIN_L2_FIT_MEMBERS = 300
MIN_CALIBRATION_MEMBERS = 200

SOFT_K = 5
SOFT_TEMPERATURE = 0.08
MINHASH_JACCARD = 0.8
MINHASH_PERMUTATIONS = 64
SEMANTIC_COSINE = 0.95
IDF_TOP_TOKENS = 120_000

BLOCK = 50_000
LABEL_TERMS = 12
LABEL_SAMPLE_PER_CELL = 150
LABEL_SAMPLE_CHARS = 600

DISTANCE_PERCENTILES = np.array(
    [1, 2, 5, 10, 15, 20, 25, 35, 50, 65, 75, 80, 85, 90, 95, 98, 99], dtype=np.float32
)
PROTOTYPE_PERCENTILES = np.array([2, 10, 25, 40, 60, 75, 90, 98], dtype=np.float32)
#: Knots of the coarse-resolution correction: distance measured against an L1
#: centroid, and the distance the same record actually has to its fine cell.
COARSE_KNOT_PERCENTILES = np.array(
    [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99], dtype=np.float32
)
COOCCURRENCE_NEIGHBORS = 16

#: Mean pairwise overlap of sibling term lists, above which naming an individual
#: child in a report is false precision and the parent should be named instead.
SIBLING_OVERLAP_MAX = 0.35

#: Languages needing at least this many records before they get their own mean
#: vector. Below it the global mean is a better estimate than a noisy one.
MIN_LANG_MEMBERS = 2_000

MAX_ARTIFACT_MB = 5.0

STOP_TERMS = {
    "that", "this", "with", "from", "have", "will", "your", "what", "which",
    "their", "they", "them", "about", "into", "than", "then", "when", "were",
    "been", "also", "some", "would", "could", "should", "there", "these",
    "olarak", "için", "gibi", "daha", "olan", "ancak", "veya", "değil",
    "return", "function", "import", "class", "const", "print",
}


def stamp(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# stage 1 — read the cache, extract, near-exact dedup
# ---------------------------------------------------------------------------


def iter_cache(cache: Path, manifest: dict):
    """Yield ``(source_meta, record)`` in a deterministic order.

    Sorted by slug so two builds over the same cache see the same corpus in the
    same order. v1 and v2 streamed straight from the Hub under a wall-clock
    budget, which meant no two builds ever had the same reference corpus and the
    word "reproducible" in the artifact meta was decoration.
    """
    axis_of = {s.slug: s.axis for s in SOURCES}
    lang_of = {s.slug: s.lang for s in SOURCES}
    for meta in sorted(manifest["sources"], key=lambda m: m["slug"]):
        if not meta.get("rows"):
            continue
        shard = cache / meta["slug"] / "records.jsonl.gz"
        if not shard.exists():
            continue
        slug = meta["slug"]
        axis = axis_of.get(slug, "unknown")
        lang_hint = lang_of.get(slug, "multi")
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                yield slug, axis, lang_hint, row


def _minhash_signatures(texts: list[str], num_perm: int):
    """Signatures for a block, computed in parallel.

    Shingling and hashing are per-record and independent; only the LSH
    insert/query has to stay ordered, and that is done serially by the caller so
    a duplicate pair inside one block is still caught in corpus order.
    """
    from concurrent.futures import ThreadPoolExecutor

    from datasketch import MinHash

    def one(text: str):
        mh = MinHash(num_perm=num_perm)
        words = text.lower().split()
        shingles = [
            " ".join(t).encode()
            for t in zip(words, words[1:], words[2:], strict=False)
        ]
        if shingles:
            mh.update_batch(shingles)
        return mh

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, texts))


def extract_and_dedup(cache: Path, manifest: dict, work: Path) -> dict:
    """Extract every record through the shared extractor, then near-exact dedup.

    v2 had a fast path that skipped format detection for any single-field prose
    source and labelled it ``plain`` unread, so 86% of the reference corpus was
    never seen by the extractor the spec calls the highest-value component. Here
    every record goes through ``extract_text``; only the code axis is told its
    format in advance, because a source file that happens to read like prose
    must still have its syntax stripped.
    """
    try:
        from datasketch import MinHashLSH
        have_lsh = True
    except ImportError:
        have_lsh = False
        stamp("  datasketch missing — falling back to exact-hash dedup")

    out_path = work / "records.jsonl.gz"
    lsh = MinHashLSH(threshold=MINHASH_JACCARD, num_perm=MINHASH_PERMUTATIONS) if have_lsh else None
    seen: set[str] = set()

    seen_rows = kept = dropped_short = dropped_dup = 0
    fmt_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    block: list[tuple] = []

    def flush(block: list[tuple], fh) -> None:
        nonlocal kept, dropped_dup
        if not block:
            return
        if have_lsh:
            signatures = _minhash_signatures([b[1] for b in block], MINHASH_PERMUTATIONS)
            for (rid, text, slug, axis, lang_hint, fmt), mh in zip(
                block, signatures, strict=True
            ):
                try:
                    if lsh.query(mh):
                        dropped_dup += 1
                        continue
                    lsh.insert(f"r{kept}", mh)
                except ValueError:
                    dropped_dup += 1
                    continue
                fh.write(json.dumps({
                    "id": rid, "text": text, "source": slug,
                    "axis": axis, "lang_hint": lang_hint, "fmt": fmt,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                fmt_counts[fmt] += 1
                axis_counts[axis] += 1
                source_counts[slug] += 1
                kept += 1
        else:
            import hashlib

            for rid, text, slug, axis, lang_hint, fmt in block:
                key = hashlib.blake2b(
                    re.sub(r"\s+", " ", text.lower())[:500].encode(), digest_size=8
                ).hexdigest()
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)
                fh.write(json.dumps({
                    "id": rid, "text": text, "source": slug,
                    "axis": axis, "lang_hint": lang_hint, "fmt": fmt,
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                fmt_counts[fmt] += 1
                axis_counts[axis] += 1
                source_counts[slug] += 1
                kept += 1
        block.clear()

    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=5) as fh:
        for slug, axis, lang_hint, row in iter_cache(cache, manifest):
            seen_rows += 1
            raw = row.get("text") or ""
            forced = "code" if axis == "code" else None
            text, fmt = extract_text(raw, detected_format=forced)
            if not text:
                dropped_short += 1
                continue
            block.append((row.get("id", ""), text[:2000], slug, axis, lang_hint, fmt))
            if len(block) >= 20_000:
                flush(block, fh)
                if kept and kept % 200_000 < 20_000:
                    stamp(f"    {kept:,} kept of {seen_rows:,} read…")
        flush(block, fh)

    return {
        "read": seen_rows,
        "kept": kept,
        "dropped_extraction": dropped_short,
        "dropped_near_exact": dropped_dup,
        "formats": dict(fmt_counts),
        "axes": dict(axis_counts),
        "sources": dict(source_counts),
        "path": out_path,
    }


def iter_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


# ---------------------------------------------------------------------------
# stage 2 — tokenize once, to disk
# ---------------------------------------------------------------------------


def tokenize_to_disk(embedder, records_path: Path, work: Path, n_records: int) -> dict:
    """One tokenization pass, cached on disk and reused by IDF and pooling.

    v1 tokenized twice — once to fit the IDF table and once to pool. v2 merged
    them but held the whole token cache in memory. Writing it here means the
    embed pass, the previous-atlas re-pool for the crosswalk, and any re-run
    with a different IDF cut all read the same arrays.
    """
    from dropoutt.langid import LanguageDetector

    detector = LanguageDetector()
    ids_path = work / "tokens.i32"
    lengths = np.zeros(n_records, dtype=np.int64)
    langs: list[str] = []
    lang_conf = np.zeros(n_records, dtype=np.float32)

    texts: list[str] = []
    written = 0
    index = 0
    with ids_path.open("wb") as fh:
        for record in iter_records(records_path):
            texts.append(record["text"])
            result = detector.detect(record["text"])
            langs.append(result.lang)
            lang_conf[index] = result.confidence
            index += 1
            if len(texts) >= BLOCK:
                tokens = embedder.tokenize(texts)
                fh.write(tokens.token_ids.astype(np.int32, copy=False).tobytes())
                lengths[written : written + len(texts)] = np.diff(tokens.indptr)
                written += len(texts)
                texts = []
                if written % (BLOCK * 10) == 0:
                    stamp(f"    tokenized {written:,}…")
        if texts:
            tokens = embedder.tokenize(texts)
            fh.write(tokens.token_ids.astype(np.int32, copy=False).tobytes())
            lengths[written : written + len(texts)] = np.diff(tokens.indptr)
            written += len(texts)

    indptr = np.zeros(n_records + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    np.save(work / "indptr.npy", indptr)
    np.save(work / "lang_conf.npy", lang_conf)
    (work / "langs.json").write_text(json.dumps(langs), encoding="utf-8")
    return {"ids_path": ids_path, "indptr": indptr, "langs": langs,
            "n_tokens": int(indptr[-1])}


def token_memmap(work: Path, n_tokens: int) -> np.ndarray:
    return np.memmap(work / "tokens.i32", dtype=np.int32, mode="r", shape=(n_tokens,))


def fit_idf(tokens: np.ndarray, vocab_size: int, n_tokens: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Unigram log-probabilities over the cached token ids."""
    counts = np.zeros(vocab_size, dtype=np.int64)
    for start in range(0, n_tokens, 20_000_000):
        block = np.asarray(tokens[start : start + 20_000_000], dtype=np.int64)
        counts += np.bincount(block, minlength=vocab_size)
    observed = np.flatnonzero(counts)
    if len(observed) > IDF_TOP_TOKENS:
        top = np.argpartition(counts[observed], -IDF_TOP_TOKENS)[-IDF_TOP_TOKENS:]
        observed = observed[top]
    order = np.argsort(-counts[observed], kind="stable")
    ids = observed[order].astype(np.int32)
    log_probs = np.log(counts[ids] / max(n_tokens, 1)).astype(np.float32)
    mapping = {int(i): float(p) for i, p in zip(ids, log_probs, strict=True)}
    return mapping, ids, log_probs


# ---------------------------------------------------------------------------
# stage 3 — embed blockwise into a memmap
# ---------------------------------------------------------------------------


def embed_to_memmap(embedder, tokens: np.ndarray, indptr: np.ndarray,
                    work: Path, name: str = "raw.f32") -> np.ndarray:
    n = len(indptr) - 1
    out = np.memmap(work / name, dtype=np.float32, mode="w+", shape=(n, EMBED_DIM))
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        lo, hi = int(indptr[start]), int(indptr[stop])
        local = indptr[start : stop + 1] - indptr[start]
        cache = TokenizedCorpus(
            token_ids=np.asarray(tokens[lo:hi], dtype=np.int32),
            indptr=local.astype(np.int64),
            n_docs=stop - start,
            max_length=512,
        )
        out[start:stop] = embedder.encode_tokenized(cache)
    out.flush()
    return out


def semantic_dedup(raw: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy cosine dedup inside 12-bit sign buckets.

    Approximate by construction: two duplicates whose signs differ in the first
    twelve dimensions land in different buckets and both survive. That is the
    accepted trade — an exact pass is quadratic — and it is stated here rather
    than implied by the word "dedup".
    """
    n = len(raw)
    keep = np.ones(n, dtype=bool)
    buckets: dict[int, list[int]] = defaultdict(list)
    n_bits = min(12, raw.shape[1])
    weights = (1 << np.arange(n_bits)).astype(np.uint16)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        block = np.asarray(raw[start:stop, :n_bits])
        keys = (block > 0).astype(np.uint16).dot(weights)
        for offset, key in enumerate(keys.tolist()):
            buckets[int(key)].append(start + offset)
    for members in buckets.values():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        sub = np.asarray(raw[idx], dtype=np.float32)
        sub /= np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9
        sims = sub @ sub.T
        rows, cols = np.where(np.triu(sims >= threshold, k=1))
        for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
            if keep[idx[a]]:
                keep[idx[b]] = False
    return keep


# ---------------------------------------------------------------------------
# stage 4 — normalization, and the probe test that chooses it
# ---------------------------------------------------------------------------


class LanguageNorm:
    """Mean removal that can be conditioned on the record's language.

    ``lang_means`` is shipped with the artifact and applied by the client using
    the language its own detector already assigned to each record. A record
    whose language is unknown falls back to the global mean, which is exactly
    v2's behaviour — so the worst case of shipping this is v2.
    """

    def __init__(self, global_mean: np.ndarray, lang_means: dict[str, np.ndarray] | None,
                 pca: np.ndarray) -> None:
        self.global_mean = global_mean.astype(np.float32)
        self.lang_means = lang_means or {}
        self.pca = pca.astype(np.float32)

    def apply(self, x: np.ndarray, langs: list[str] | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.lang_means and langs is not None:
            means = np.tile(self.global_mean, (len(x), 1))
            for i, lang in enumerate(langs):
                m = self.lang_means.get(lang)
                if m is not None:
                    means[i] = m
            x = x - means
        else:
            x = x - self.global_mean
        if self.pca.size:
            x = x - (x @ self.pca.T) @ self.pca
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def fit_variants(sample: np.ndarray, sample_langs: list[str]) -> dict[str, LanguageNorm]:
    """Fit the global and per-language normalizations on the same sample."""
    global_mean = sample.mean(axis=0)

    counts = Counter(sample_langs)
    lang_means: dict[str, np.ndarray] = {}
    for lang, count in counts.items():
        if lang == "unknown" or count < MIN_LANG_MEMBERS:
            continue
        mask = np.fromiter((x == lang for x in sample_langs), dtype=bool,
                           count=len(sample_langs))
        lang_means[lang] = sample[mask].mean(axis=0).astype(np.float32)

    def top_pcs(centered: np.ndarray, k: int = 2) -> np.ndarray:
        if len(centered) <= k + 10:
            return np.zeros((0, centered.shape[1]), dtype=np.float32)
        # Randomised SVD: the sample is hundreds of thousands of rows and only
        # the top two directions are wanted.
        from sklearn.utils.extmath import randomized_svd

        _, _, vt = randomized_svd(centered, n_components=k, random_state=SEED)
        return vt[:k].astype(np.float32)

    centered_global = sample - global_mean
    variants = {"global": LanguageNorm(global_mean, None, top_pcs(centered_global))}

    if lang_means:
        centered_lang = sample.copy()
        for i, lang in enumerate(sample_langs):
            centered_lang[i] -= lang_means.get(lang, global_mean)
        variants["per_language"] = LanguageNorm(
            global_mean, lang_means, top_pcs(centered_lang)
        )
    return variants


def load_probes(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_scores(variant: LanguageNorm, centers: np.ndarray, probe_raw: np.ndarray,
                 probes: list[dict], probe_langs: list[str]) -> dict:
    """Cross-language agreement and within-language topic separation.

    Agreement alone is not a target: mapping every language to one cell would
    score perfectly. Separation is the guard, and both are reported.
    """
    projected = variant.apply(probe_raw, probe_langs)
    cells = (projected @ centers.T).argmax(axis=1)

    by_topic: dict[str, list[int]] = defaultdict(list)
    by_lang: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for probe, cell in zip(probes, cells.tolist(), strict=True):
        by_topic[probe["topic"]].append(cell)
        by_lang[probe["lang"]].append((probe["topic"], cell))

    agree_hits = agree_total = 0
    for cells_for_topic in by_topic.values():
        for i in range(len(cells_for_topic)):
            for j in range(i + 1, len(cells_for_topic)):
                agree_total += 1
                agree_hits += int(cells_for_topic[i] == cells_for_topic[j])

    sep_hits = sep_total = 0
    for entries in by_lang.values():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i][0] == entries[j][0]:
                    continue
                sep_total += 1
                sep_hits += int(entries[i][1] != entries[j][1])

    agreement = agree_hits / max(agree_total, 1)
    separation = sep_hits / max(sep_total, 1)
    return {
        "cross_language_agreement": round(agreement, 4),
        "within_language_separation": round(separation, 4),
        "score": round(0.5 * (agreement + separation), 4),
        "pairs": {"agreement": agree_total, "separation": sep_total},
    }


def choose_normalization(variants: dict[str, LanguageNorm], sample: np.ndarray,
                         sample_langs: list[str], probe_raw: np.ndarray,
                         probes: list[dict], probe_langs: list[str],
                         *, separation_tolerance: float = 0.02,
                         force: str | None = None) -> tuple[str, dict]:
    """Fit a coarse map per variant and let the probe set pick the winner.

    The comparison is made at L1 only. It is the cheap half of the pipeline and
    it is where the language-shaped regions appeared, so it answers the question
    that is actually being asked without fitting the full hierarchy twice.
    """
    from sklearn.cluster import MiniBatchKMeans

    results: dict[str, dict] = {}
    for name, variant in variants.items():
        projected = variant.apply(sample, sample_langs)
        km = MiniBatchKMeans(n_clusters=N_L1, random_state=SEED, batch_size=4096,
                             n_init=3, max_iter=200).fit(projected)
        centers = km.cluster_centers_.astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9
        scores = probe_scores(variant, centers, probe_raw, probes, probe_langs)

        # How language-shaped is the map itself: the share of records in the
        # single most language-pure region, and mean purity across regions.
        labels = km.labels_
        purity = []
        for cell in range(N_L1):
            members = [sample_langs[i] for i in np.flatnonzero(labels == cell)]
            if members:
                purity.append(Counter(members).most_common(1)[0][1] / len(members))
        scores["mean_language_purity"] = round(float(np.mean(purity)) if purity else 0.0, 4)
        scores["max_language_purity"] = round(float(np.max(purity)) if purity else 0.0, 4)
        results[name] = scores

    winner = "global"
    if "per_language" in results:
        a, b = results["per_language"], results["global"]
        better_agreement = a["cross_language_agreement"] > b["cross_language_agreement"]
        keeps_separation = (
            a["within_language_separation"]
            >= b["within_language_separation"] - separation_tolerance
        )
        if better_agreement and keeps_separation:
            winner = "per_language"
    results["rule"] = (
        f"per_language ships only if it raises cross-language agreement without "
        f"losing more than {separation_tolerance:.0%} of within-language topic separation"
    )
    if force and force != "auto":
        if force not in variants:
            raise SystemExit(f"--normalization {force} was not fitted (too few "
                             f"records per language?)")
        results["gate_winner"] = winner
        results["forced"] = force
        winner = force
    results["winner"] = winner
    results["separation_tolerance"] = separation_tolerance
    return winner, results


# ---------------------------------------------------------------------------
# stage 5 — hierarchy with a measured number of children
# ---------------------------------------------------------------------------


def choose_child_counts(emb: np.ndarray, l1_labels: np.ndarray, rng) -> tuple[dict, dict]:
    """Silhouette curve per parent, then a budget handed out by marginal gain.

    Every parent starts at ``L2_K_MIN``. The budget is spent one cell at a time
    on whichever parent gains most from its next child, so the 800 cells go
    where there is structure to describe instead of being divided equally
    between a region holding one subject and a region holding twelve.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score

    curves: dict[int, dict[int, float]] = {}
    caps: dict[int, int] = {}
    for parent in range(N_L1):
        members = np.flatnonzero(l1_labels == parent)
        cap = int(np.clip(len(members) // MIN_L2_FIT_MEMBERS, 1, L2_K_MAX))
        caps[parent] = max(1, min(cap, L2_K_MAX))
        if caps[parent] <= 1 or len(members) < L2_K_MIN * MIN_L2_FIT_MEMBERS:
            curves[parent] = {1: 0.0}
            caps[parent] = 1
            continue
        take = members if len(members) <= 20_000 else rng.choice(members, 20_000, replace=False)
        sub = np.asarray(emb[np.sort(take)], dtype=np.float32)
        score_at: dict[int, float] = {}
        scoring = sub if len(sub) <= 4_000 else sub[rng.choice(len(sub), 4_000, replace=False)]
        for k in range(L2_K_MIN, caps[parent] + 1):
            km = MiniBatchKMeans(n_clusters=k, random_state=SEED + parent,
                                 batch_size=2048, n_init=3, max_iter=150).fit(sub)
            labels = km.predict(scoring)
            if len(set(labels.tolist())) < 2:
                score_at[k] = -1.0
                continue
            score_at[k] = float(silhouette_score(scoring, labels, metric="cosine"))
        curves[parent] = score_at
        stamp(f"    parent {parent:>2}: {len(members):>7,} members, "
              f"k*={max(score_at, key=score_at.get)} "
              f"(cap {caps[parent]})")

    allocation = {p: min(L2_K_MIN, caps[p]) for p in range(N_L1)}
    spent = sum(allocation.values())
    while spent < L2_BUDGET:
        best_parent, best_gain = None, 0.0
        for parent, k in allocation.items():
            nxt = k + 1
            if nxt > caps[parent] or nxt not in curves[parent]:
                continue
            gain = curves[parent][nxt] - curves[parent].get(k, -1.0)
            if gain > best_gain:
                best_parent, best_gain = parent, gain
        if best_parent is None:
            break
        allocation[best_parent] += 1
        spent += 1
    return allocation, {str(p): {str(k): round(v, 4) for k, v in c.items()}
                        for p, c in curves.items()}


def fit_children(emb: np.ndarray, l1_labels: np.ndarray, allocation: dict[int, int]):
    from sklearn.cluster import MiniBatchKMeans

    centroids: list[np.ndarray] = []
    parents: list[int] = []
    for parent in range(N_L1):
        members = np.flatnonzero(l1_labels == parent)
        if not len(members):
            continue
        k = max(1, min(allocation.get(parent, 1), len(members)))
        sub = np.asarray(emb[members], dtype=np.float32)
        if k == 1:
            centre = sub.mean(axis=0)
            centroids.append(centre)
            parents.append(parent)
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED + parent,
                             batch_size=min(2048, max(len(sub), 1)),
                             n_init=3, max_iter=200).fit(sub)
        for c in range(k):
            centroids.append(km.cluster_centers_[c])
            parents.append(parent)
    centres = np.vstack(centroids).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-9
    return centres, np.asarray(parents, dtype=np.int32)


def assign_blockwise(emb: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(emb)
    assign = np.empty(n, dtype=np.int32)
    best = np.empty(n, dtype=np.float32)
    for start in range(0, n, BLOCK):
        stop = min(start + BLOCK, n)
        sims = np.asarray(emb[start:stop]) @ centres.T
        idx = sims.argmax(axis=1)
        assign[start:stop] = idx
        best[start:stop] = sims[np.arange(stop - start), idx]
    return assign, best


# ---------------------------------------------------------------------------
# stage 6 — calibration
# ---------------------------------------------------------------------------


def coarse_correction(emb: np.ndarray, l1_centroids: np.ndarray, parents: np.ndarray,
                      assign: np.ndarray, fine_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map an L1-measured distance to the fine distance it implies.

    Distance to a coarse centroid systematically overstates novelty: part of it
    is structure the coarse map cannot see. Without this table every user is
    told their data is unusual. The spec calls it mandatory; v1 and v2 shipped
    without it.
    """
    n_knots = len(COARSE_KNOT_PERCENTILES)
    knots = np.zeros((len(l1_centroids), n_knots), dtype=np.float32)
    expected = np.zeros((len(l1_centroids), n_knots), dtype=np.float32)
    coarse_distance = np.empty(len(assign), dtype=np.float32)
    for start in range(0, len(assign), BLOCK):
        stop = min(start + BLOCK, len(assign))
        block = np.asarray(emb[start:stop])
        own = parents[assign[start:stop]]
        coarse_distance[start:stop] = 1.0 - np.einsum(
            "ij,ij->i", block, l1_centroids[own]
        )
    for parent in range(len(l1_centroids)):
        members = np.flatnonzero(parents[assign] == parent)
        if len(members) < 50:
            knots[parent] = np.linspace(0.0, 1.0, n_knots)
            expected[parent] = np.linspace(0.0, 1.0, n_knots)
            continue
        d1 = coarse_distance[members]
        d2 = fine_distance[members]
        cuts = np.percentile(d1, COARSE_KNOT_PERCENTILES).astype(np.float32)
        knots[parent] = cuts
        order = np.argsort(d1)
        d1s, d2s = d1[order], d2[order]
        positions = np.searchsorted(d1s, cuts)
        for i, pos in enumerate(positions.tolist()):
            lo = max(0, pos - len(d1s) // 20)
            hi = min(len(d1s), pos + len(d1s) // 20 + 1)
            expected[parent, i] = float(np.median(d2s[lo:hi])) if hi > lo else 0.0
    return knots, expected


def distance_references(assign: np.ndarray, parents: np.ndarray, fine_distance: np.ndarray,
                        n_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = np.zeros((n_cells, len(DISTANCE_PERCENTILES)), dtype=np.float32)
    support = np.bincount(assign, minlength=n_cells).astype(np.int32)
    reliable = support >= MIN_CALIBRATION_MEMBERS
    parent_of_record = parents[assign]
    for cell in range(n_cells):
        members = np.flatnonzero(assign == cell)
        if not len(members):
            continue
        dists = fine_distance[members]
        if len(members) < MIN_CALIBRATION_MEMBERS:
            pool = fine_distance[parent_of_record == parents[cell]]
            need = MIN_CALIBRATION_MEMBERS - len(members)
            if len(pool):
                take = np.linspace(0, len(pool) - 1, num=need, dtype=np.int64)
                dists = np.concatenate([dists, np.sort(pool)[take]])
        refs[cell] = np.percentile(dists, DISTANCE_PERCENTILES).astype(np.float32)
    return refs, support, reliable


def cell_prototypes(emb: np.ndarray, assign: np.ndarray, distance: np.ndarray,
                    record_ids: list[str], n_cells: int):
    n_proto = len(PROTOTYPE_PERCENTILES)
    vectors = np.zeros((n_cells, n_proto, EMBED_DIM), dtype=np.float16)
    ids = np.zeros((n_cells, n_proto), dtype="S16")
    distances = np.zeros((n_cells, n_proto), dtype=np.float32)
    for cell in range(n_cells):
        members = np.flatnonzero(assign == cell)
        if not len(members):
            continue
        ordered = members[np.argsort(distance[members])]
        ranks = np.rint(PROTOTYPE_PERCENTILES / 100.0 * (len(ordered) - 1)).astype(np.int64)
        selected = ordered[ranks]
        vectors[cell] = np.asarray(emb[selected]).astype(np.float16)
        ids[cell] = np.asarray([record_ids[i].encode()[:16] for i in selected], dtype="S16")
        distances[cell] = distance[selected]
    return vectors, ids, distances


def cooccurrence_neighbors(cell_source_counts: np.ndarray, top_k: int = COOCCURRENCE_NEIGHBORS):
    occupancy = (cell_source_counts > 0).astype(np.int32)
    co = occupancy @ occupancy.T
    freq = np.diag(co).copy()
    union = freq[:, None] + freq[None, :] - co
    scores = np.divide(co, union, out=np.zeros_like(co, dtype=np.float32), where=union > 0)
    np.fill_diagonal(scores, -1.0)
    k = min(top_k, max(1, scores.shape[1] - 1))
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return (np.take_along_axis(idx, order, axis=1).astype(np.int32),
            np.take_along_axis(vals, order, axis=1).astype(np.float32))


def categorical_counts(assign: np.ndarray, values: list[str], n_cells: int):
    labels = sorted(set(values))
    lookup = {label: i for i, label in enumerate(labels)}
    codes = np.fromiter((lookup[v] for v in values), dtype=np.int32, count=len(values))
    counts = np.zeros((n_cells, len(labels)), dtype=np.int32)
    np.add.at(counts, (assign, codes), 1)
    return labels, counts


# ---------------------------------------------------------------------------
# stage 7 — labels and sibling distinguishability
# ---------------------------------------------------------------------------


def collect_label_samples(records_path: Path, assign: np.ndarray, keep_index: np.ndarray,
                          n_cells: int) -> list[list[str]]:
    """Reservoir-sample texts per cell without holding the corpus in memory."""
    rng = np.random.default_rng(SEED)
    samples: list[list[str]] = [[] for _ in range(n_cells)]
    seen = np.zeros(n_cells, dtype=np.int64)
    position = 0
    kept_lookup = np.full(keep_index.max() + 1 if len(keep_index) else 1, -1, dtype=np.int64)
    kept_lookup[keep_index] = np.arange(len(keep_index))
    for record in iter_records(records_path):
        slot = kept_lookup[position] if position < len(kept_lookup) else -1
        position += 1
        if slot < 0:
            continue
        cell = int(assign[slot])
        seen[cell] += 1
        bucket = samples[cell]
        if len(bucket) < LABEL_SAMPLE_PER_CELL:
            bucket.append(record["text"][:LABEL_SAMPLE_CHARS])
        else:
            j = int(rng.integers(0, seen[cell]))
            if j < LABEL_SAMPLE_PER_CELL:
                bucket[j] = record["text"][:LABEL_SAMPLE_CHARS]
    return samples


def term_lists(samples: list[list[str]], k: int = LABEL_TERMS) -> list[list[str]]:
    df: Counter[str] = Counter()
    per_cell: list[Counter[str]] = []
    for texts in samples:
        tf: Counter[str] = Counter()
        for text in texts:
            local: set[str] = set()
            for word in text.lower().split():
                word = "".join(ch for ch in word if ch.isalpha())
                if len(word) > 3 and word not in STOP_TERMS:
                    tf[word] += 1
                    local.add(word)
            df.update(local)
        per_cell.append(tf)
    total = float(sum(df.values()) or 1)
    out: list[list[str]] = []
    for tf in per_cell:
        if not tf:
            out.append([])
            continue
        scored = sorted(
            ((count * math.log(1.0 + total / (1 + df.get(word, 0))), word)
             for word, count in tf.items()),
            reverse=True,
        )
        out.append([word for _, word in scored[:k]])
    return out


def sibling_distinguishability(terms: list[list[str]], parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Can a report name one child, or only the family?

    If siblings share most of their distinctive terms, then "thin in
    contract drafting" is false precision over "thin in Legal" — the map cannot
    support the finer statement even though it has the finer cell.
    """
    n_parents = int(parents.max()) + 1 if len(parents) else 0
    overlap = np.zeros(n_parents, dtype=np.float32)
    ok = np.zeros(n_parents, dtype=bool)
    for parent in range(n_parents):
        children = np.flatnonzero(parents == parent)
        sets = [set(terms[c]) for c in children if terms[c]]
        if len(sets) < 2:
            overlap[parent] = 0.0
            ok[parent] = len(sets) == 1
            continue
        scores = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                scores.append(len(sets[i] & sets[j]) / max(len(union), 1))
        overlap[parent] = float(np.mean(scores))
        ok[parent] = overlap[parent] <= SIBLING_OVERLAP_MAX
    return ok, overlap


# ---------------------------------------------------------------------------
# stage 8 — release gates
# ---------------------------------------------------------------------------


def stability_check(emb: np.ndarray, rng, rounds: int = 3, sample: int = 150_000) -> dict:
    """Bootstrap-resample, refit, compare. Instability means the map is noise."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score

    n = len(emb)
    holdout_idx = np.sort(rng.choice(n, min(20_000, n), replace=False))
    holdout = np.asarray(emb[holdout_idx], dtype=np.float32)
    labelings = []
    for r in range(rounds):
        pick = np.sort(rng.choice(n, min(sample, n), replace=True))
        fit = np.asarray(emb[pick], dtype=np.float32)
        km = MiniBatchKMeans(n_clusters=N_L1, random_state=SEED + 100 + r,
                             batch_size=4096, n_init=3, max_iter=200).fit(fit)
        labelings.append(km.predict(holdout))
    pairs = [
        adjusted_rand_score(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    return {
        "rounds": rounds,
        "mean_ari": round(float(np.mean(pairs)), 4) if pairs else 0.0,
        "min_ari": round(float(np.min(pairs)), 4) if pairs else 0.0,
        "holdout": int(len(holdout_idx)),
    }


def ood_check(variant: LanguageNorm, centres: np.ndarray, embedder,
              ood: list[dict], threshold: float) -> dict:
    """Content that is genuinely nothing must come back unplaceable."""
    if not ood:
        return {}
    raw = embedder.encode([o["text"] for o in ood])
    projected = variant.apply(raw, ["unknown"] * len(ood))
    best = (projected @ centres.T).max(axis=1)
    return {
        "cutoff": round(float(threshold), 5),
        "cases": [
            {"name": o["name"], "best_cosine": round(float(s), 4),
             "unplaceable": bool(s < threshold)}
            for o, s in zip(ood, best.tolist(), strict=True)
        ],
        "unplaceable_share": round(float((best < threshold).mean()), 4),
    }


def containment_crosswalk(current: np.ndarray, previous: np.ndarray,
                          n_current: int, n_previous: int, previous_version: str) -> dict:
    """Continuity measured by containment, not symmetric Jaccard.

    v2 reported 0 of 480 previous cells surviving, which read as total churn but
    was mostly arithmetic: the cell count went 480 → 1000, so a symmetric
    Jaccard of 0.8 was close to unreachable no matter how stable the geometry
    was. What a reader actually wants to know is whether an old cell's records
    stayed together — as one new cell or as a clean set of children — and that
    is what containment measures.
    """
    pairs = np.zeros((n_current, n_previous), dtype=np.int64)
    np.add.at(pairs, (current.astype(np.int64), previous.astype(np.int64)), 1)
    current_size = pairs.sum(axis=1)
    previous_size = pairs.sum(axis=0)
    majority_parent = pairs.argmax(axis=1)

    survived = split = scattered = retired = 0
    detail: list[dict] = []
    for old in range(n_previous):
        if previous_size[old] == 0:
            retired += 1
            continue
        family = np.flatnonzero(majority_parent == old)
        held = int(pairs[family, old].sum()) if len(family) else 0
        share = held / int(previous_size[old])
        top_share = float(pairs[:, old].max() / previous_size[old])
        if share >= 0.8 and len(family) == 1:
            relation = "unchanged"
            survived += 1
        elif share >= 0.8:
            relation = "clean_split"
            split += 1
        else:
            relation = "scattered"
            scattered += 1
        detail.append({
            "previous_cell_id": old,
            "children": [int(c) for c in family],
            "retained_share": round(share, 4),
            "largest_single_share": round(top_share, 4),
            "relationship": relation,
        })

    total = max(n_previous - retired, 1)
    return {
        "previous_version": previous_version,
        "method": "containment_over_shared_reference_records",
        "summary": {
            "previous_cells": int(n_previous),
            "current_cells": int(n_current),
            "unchanged": survived,
            "clean_split": split,
            "scattered": scattered,
            "retired": retired,
            "continuity_rate": round((survived + split) / total, 4),
        },
        "cells": detail,
    }


# ---------------------------------------------------------------------------


def main() -> int:
    global N_L1, L2_BUDGET, L2_K_MAX, L2_K_MIN, MIN_L2_FIT_MEMBERS, MIN_LANG_MEMBERS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path(".atlas-cache"))
    ap.add_argument("--work", type=Path, default=None,
                    help="Scratch for memmaps (default <cache>/work). Reusable.")
    ap.add_argument("--out", type=Path,
                    default=Path("src/dropoutt/data/atlas/atlas-lite-v3.npz"))
    ap.add_argument("--probes", type=Path,
                    default=Path(__file__).resolve().parent / "atlas_probes.json")
    ap.add_argument("--previous", type=Path,
                    default=Path("src/dropoutt/data/atlas/atlas-lite-v2.npz"))
    ap.add_argument("--labels", type=Path,
                    default=Path("src/dropoutt/data/atlas/l1_labels_v3.json"))
    ap.add_argument("--norm-sample", type=int, default=400_000)
    ap.add_argument("--reuse", action="store_true",
                    help="Reuse extraction/tokenization/embeddings already in --work.")
    ap.add_argument("--no-stability", action="store_true")
    # Geometry is overridable so the pipeline can be exercised end to end on a
    # small corpus. The defaults are the release values.
    ap.add_argument("--l1", type=int, default=N_L1)
    ap.add_argument("--l2-budget", type=int, default=L2_BUDGET)
    ap.add_argument("--l2-max", type=int, default=L2_K_MAX)
    ap.add_argument("--l2-min", type=int, default=L2_K_MIN)
    ap.add_argument("--min-l2-members", type=int, default=MIN_L2_FIT_MEMBERS)
    ap.add_argument("--min-lang-members", type=int, default=MIN_LANG_MEMBERS)
    ap.add_argument("--min-records", type=int, default=50_000)
    ap.add_argument("--normalization", choices=("auto", "global", "per_language"),
                    default="auto",
                    help="Override the probe gate. 'auto' lets the probe set "
                         "decide, which is the point of having one; the override "
                         "is recorded in the artifact next to the score it "
                         "contradicts.")
    ap.add_argument("--separation-tolerance", type=float, default=0.02,
                    help="How much within-language topic separation per-language "
                         "centering may give up in exchange for cross-language "
                         "agreement (default 0.02).")
    args = ap.parse_args()

    N_L1 = args.l1
    MIN_LANG_MEMBERS = args.min_lang_members
    L2_BUDGET = args.l2_budget
    L2_K_MAX = args.l2_max
    L2_K_MIN = args.l2_min
    MIN_L2_FIT_MEMBERS = args.min_l2_members

    work = args.work or (args.cache / "work")
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache / "manifest.json"
    if not manifest_path.exists():
        stamp(f"No corpus cache at {manifest_path}. Run tools/fetch_corpus.py first.")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    timings: dict[str, float] = {}
    wall0 = time.time()

    # -- 1. extract + near-exact dedup -------------------------------------
    records_path = work / "records.jsonl.gz"
    summary_path = work / "extract_summary.json"
    if args.reuse and records_path.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stamp(f"Reusing extraction: {summary['kept']:,} records")
    else:
        stamp(f"Extracting and deduplicating {manifest['totals']['rows']:,} cached rows…")
        t0 = time.time()
        summary = extract_and_dedup(args.cache, manifest, work)
        summary["path"] = str(summary["path"])
        timings["extract_dedup_s"] = time.time() - t0
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        stamp(f"  kept {summary['kept']:,} "
              f"(dropped {summary['dropped_extraction']:,} at extraction, "
              f"{summary['dropped_near_exact']:,} near-exact) "
              f"in {timings['extract_dedup_s']:.0f}s")
    n_records = summary["kept"]
    if n_records < args.min_records:
        stamp(f"Only {n_records:,} records survived; need {args.min_records:,}.")
        return 1
    stamp(f"  formats: {dict(sorted(summary['formats'].items(), key=lambda kv: -kv[1])[:8])}")

    # -- 2. encoder + tokenize --------------------------------------------
    stamp(f"\nLoading encoder {DEFAULT_MODEL} …")
    embedder = load_embedder(DEFAULT_MODEL, out_dim=EMBED_DIM)
    if embedder is None:
        stamp("Could not load the encoder.")
        return 1
    weight_hash = embedder.weight_hash

    token_meta_path = work / "token_meta.json"
    if args.reuse and token_meta_path.exists():
        token_meta = json.loads(token_meta_path.read_text(encoding="utf-8"))
        indptr = np.load(work / "indptr.npy")
        langs = json.loads((work / "langs.json").read_text(encoding="utf-8"))
        stamp(f"Reusing token cache: {token_meta['n_tokens']:,} tokens")
    else:
        stamp("Tokenizing once, to disk (language detection rides along) …")
        t0 = time.time()
        result = tokenize_to_disk(embedder, records_path, work, n_records)
        indptr, langs = result["indptr"], result["langs"]
        token_meta = {"n_tokens": result["n_tokens"]}
        token_meta_path.write_text(json.dumps(token_meta), encoding="utf-8")
        timings["tokenize_s"] = time.time() - t0
        stamp(f"  {result['n_tokens']:,} token occurrences in {timings['tokenize_s']:.0f}s")

    tokens = token_memmap(work, token_meta["n_tokens"])
    vocab_size = int(np.asarray(embedder._model.embedding).shape[0])  # noqa: SLF001

    stamp("Fitting the IDF table from the cached tokens …")
    t0 = time.time()
    token_log_prob, idf_ids, idf_lps = fit_idf(tokens, vocab_size, token_meta["n_tokens"])
    timings["idf_s"] = time.time() - t0
    embedder = embedder.bind_idf(token_log_prob)
    stamp(f"  {len(token_log_prob):,} token types in {timings['idf_s']:.0f}s")

    # -- 3. embed ----------------------------------------------------------
    raw_path = work / "raw.f32"
    if args.reuse and raw_path.exists():
        raw = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n_records, EMBED_DIM))
        stamp(f"Reusing embeddings: {raw.shape}")
    else:
        stamp(f"Embedding {n_records:,} records (SIF, dim={EMBED_DIM}) …")
        t0 = time.time()
        raw = embed_to_memmap(embedder, tokens, indptr, work)
        timings["embed_s"] = time.time() - t0
        stamp(f"  {raw.shape} in {timings['embed_s']:.0f}s "
              f"({n_records / max(timings['embed_s'], 0.001):,.0f} rec/s)")

    # -- 4. semantic dedup -------------------------------------------------
    stamp("\nSemantic dedup (cosine) …")
    t0 = time.time()
    keep = semantic_dedup(raw, SEMANTIC_COSINE)
    keep_index = np.flatnonzero(keep)
    timings["dedup_semantic_s"] = time.time() - t0
    stamp(f"  kept {len(keep_index):,} of {n_records:,} "
          f"({timings['dedup_semantic_s']:.0f}s)")

    # Stream the metadata for the kept rows. Materialising every record to index
    # into it would hold the whole corpus text in memory, which is the thing the
    # on-disk cache exists to avoid.
    wanted = np.zeros(n_records, dtype=bool)
    wanted[keep_index] = True
    axes: list[str] = []
    sources: list[str] = []
    formats: list[str] = []
    record_ids: list[str] = []
    for position, record in enumerate(iter_records(records_path)):
        if position >= n_records or not wanted[position]:
            continue
        axes.append(record["axis"])
        sources.append(record["source"])
        formats.append(record["fmt"])
        record_ids.append(record["id"])
    kept_langs = [langs[i] for i in keep_index]

    kept_raw = np.memmap(work / "kept.f32", dtype=np.float32, mode="w+",
                         shape=(len(keep_index), EMBED_DIM))
    for start in range(0, len(keep_index), BLOCK):
        stop = min(start + BLOCK, len(keep_index))
        kept_raw[start:stop] = raw[keep_index[start:stop]]
    kept_raw.flush()
    n_kept = len(keep_index)

    # -- 5. normalization A/B ---------------------------------------------
    stamp("\nFitting normalization variants and running the probe gate …")
    t0 = time.time()
    sample_idx = np.sort(rng.choice(n_kept, min(args.norm_sample, n_kept), replace=False))
    sample = np.asarray(kept_raw[sample_idx], dtype=np.float32)
    sample_langs = [kept_langs[i] for i in sample_idx]
    variants = fit_variants(sample, sample_langs)

    probe_data = load_probes(args.probes)
    probes = probe_data["probes"]
    from dropoutt.langid import LanguageDetector

    detector = LanguageDetector()
    probe_raw = embedder.encode([p["text"] for p in probes])
    # The client will use its own detector, not a declared label, so the gate is
    # run the same way. A probe the detector misreads is a real cost of the
    # scheme and must show up in the score rather than be assumed away.
    probe_langs = [detector.detect(p["text"]).lang for p in probes]
    detector_agreement = float(np.mean([
        a == b["lang"] for a, b in zip(probe_langs, probes, strict=True)
    ]))

    winner, probe_report = choose_normalization(
        variants, sample, sample_langs, probe_raw, probes, probe_langs,
        separation_tolerance=args.separation_tolerance,
        force=args.normalization,
    )
    probe_report["detector_agreement_with_declared_language"] = round(detector_agreement, 4)
    norm = variants[winner]
    timings["normalization_s"] = time.time() - t0
    stamp(f"  global      {probe_report['global']}")
    if "per_language" in probe_report:
        stamp(f"  per_language {probe_report['per_language']}")
    stamp(f"  → shipping '{winner}' normalization")

    normed = np.memmap(work / "normed.f32", dtype=np.float32, mode="w+",
                       shape=(n_kept, EMBED_DIM))
    for start in range(0, n_kept, BLOCK):
        stop = min(start + BLOCK, n_kept)
        normed[start:stop] = norm.apply(
            np.asarray(kept_raw[start:stop]), kept_langs[start:stop]
        )
    normed.flush()

    # -- 6. hierarchy ------------------------------------------------------
    from sklearn.cluster import MiniBatchKMeans

    stamp(f"\nFitting {N_L1} coarse regions …")
    t0 = time.time()
    fit_idx = np.sort(rng.choice(n_kept, min(500_000, n_kept), replace=False))
    l1 = MiniBatchKMeans(n_clusters=N_L1, random_state=SEED, batch_size=4096,
                         n_init=5, max_iter=300).fit(np.asarray(normed[fit_idx]))
    l1_centroids = l1.cluster_centers_.astype(np.float32)
    l1_centroids /= np.linalg.norm(l1_centroids, axis=1, keepdims=True) + 1e-9
    l1_fit_labels = np.empty(n_kept, dtype=np.int32)
    for start in range(0, n_kept, BLOCK):
        stop = min(start + BLOCK, n_kept)
        l1_fit_labels[start:stop] = (np.asarray(normed[start:stop]) @ l1_centroids.T).argmax(axis=1)
    timings["cluster_l1_s"] = time.time() - t0

    stamp(f"Choosing children per region (k in {L2_K_MIN}..{L2_K_MAX}, budget {L2_BUDGET}) …")
    t0 = time.time()
    allocation, curves = choose_child_counts(normed, l1_fit_labels, rng)
    stamp(f"  allocated {sum(allocation.values())} cells across {N_L1} regions "
          f"(min {min(allocation.values())}, max {max(allocation.values())})")
    centres, parents = fit_children(normed, l1_fit_labels, allocation)
    timings["cluster_l2_s"] = time.time() - t0
    stamp(f"  {centres.shape[0]} fine cells in {timings['cluster_l2_s']:.0f}s")

    # The one assignment that exists. The coarse answer is this cell's parent,
    # so drilling down cannot contradict it.
    assign, best = assign_blockwise(normed, centres)
    l1_assign = parents[assign]
    fine_distance = 1.0 - best
    coarse_disagreement = float((l1_assign != l1_fit_labels).mean())

    # -- 7. calibration ----------------------------------------------------
    stamp("\nCalibrating …")
    off_threshold = float(np.percentile(best, 2.0))
    region_size = np.bincount(assign, minlength=centres.shape[0]).astype(np.int32)
    l1_size = np.bincount(l1_assign, minlength=N_L1).astype(np.int32)
    distance_refs, refs_support, refs_reliable = distance_references(
        assign, parents, fine_distance, centres.shape[0]
    )
    coarse_knots, coarse_expected = coarse_correction(
        normed, l1_centroids, parents, assign, fine_distance
    )
    stamp(f"  off-atlas cutoff (2nd pct cosine): {off_threshold:.4f}")
    stamp(f"  {int(refs_reliable.sum())}/{centres.shape[0]} cells calibrate on "
          f"≥{MIN_CALIBRATION_MEMBERS} direct members")

    sample_n = min(5_000, n_kept)
    sample_sims = np.asarray(normed[np.sort(rng.choice(n_kept, sample_n, replace=False))]) @ centres.T
    k = min(SOFT_K, centres.shape[0])
    part = np.argpartition(-sample_sims, kth=k - 1, axis=1)[:, :k]
    top = np.take_along_axis(sample_sims, part, axis=1)
    logits = top / SOFT_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True) + 1e-9
    soft_mean = float((w > 0.15).sum(axis=1).mean())
    stamp(f"  soft assignment: {soft_mean:.2f} cells above weight 0.15 at T={SOFT_TEMPERATURE}")

    source_labels, cell_source_counts = categorical_counts(assign, sources, centres.shape[0])
    axis_labels, cell_topic_counts = categorical_counts(assign, axes, centres.shape[0])
    language_labels, cell_language_counts = categorical_counts(assign, kept_langs, centres.shape[0])
    cooc_ids, cooc_scores = cooccurrence_neighbors(cell_source_counts)
    proto_vectors, proto_ids, proto_distances = cell_prototypes(
        normed, assign, fine_distance, record_ids, centres.shape[0]
    )

    # -- 8. labels ---------------------------------------------------------
    stamp("\nLabelling …")
    t0 = time.time()
    samples = collect_label_samples(records_path, assign, keep_index, centres.shape[0])
    terms = term_lists(samples)
    region_terms = [", ".join(t) for t in terms]
    parent_samples: list[list[str]] = [[] for _ in range(N_L1)]
    for cell, texts in enumerate(samples):
        parent_samples[int(parents[cell])].extend(texts[:40])
    l1_terms = [", ".join(t) for t in term_lists(parent_samples)]
    l1_labels = list(l1_terms)
    if args.labels.exists():
        curated = json.loads(args.labels.read_text(encoding="utf-8")).get("labels", {})
        for i in range(N_L1):
            if curated.get(str(i)):
                l1_labels[i] = curated[str(i)]
    family_ok, family_overlap = sibling_distinguishability(terms, parents)
    timings["label_s"] = time.time() - t0
    stamp(f"  {int(family_ok.sum())}/{N_L1} families have distinguishable children; "
          f"the rest are reported at parent level")

    # -- 9. release gates --------------------------------------------------
    stamp("\nRelease gates …")
    ood = ood_check(norm, centres, embedder, probe_data.get("ood", []), off_threshold)
    if ood:
        stamp(f"  OOD: {ood['unplaceable_share']:.0%} of control blobs are unplaceable")
    stability = {} if args.no_stability else stability_check(normed, rng)
    if stability:
        stamp(f"  stability: mean ARI {stability['mean_ari']:.3f} over "
              f"{stability['rounds']} bootstrap refits")

    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

    def purity(labels: list[str]) -> dict[str, float]:
        y = np.asarray(labels, dtype=object)
        macro, majority = [], 0
        for cell in np.unique(assign):
            members = y[assign == cell]
            if not len(members):
                continue
            largest = Counter(members.tolist()).most_common(1)[0][1]
            majority += largest
            macro.append(largest / len(members))
        return {"macro": float(np.mean(macro)) if macro else 0.0,
                "micro": majority / max(len(y), 1)}

    axis_purity = purity(axes)
    source_purity = purity(sources)
    language_purity = purity(kept_langs)
    source_ami = float(adjusted_mutual_info_score(sources, assign))
    language_nmi = float(normalized_mutual_info_score(kept_langs, assign))
    non_english = float(np.mean([lang not in ("en", "unknown") for lang in kept_langs]))
    stamp(f"  purity — axis {axis_purity['micro']:.3f}, source {source_purity['micro']:.3f} "
          f"(lower is better), language {language_purity['micro']:.3f}")
    stamp(f"  non-English share of the reference corpus: {non_english:.1%}")

    axis_rows = Counter(axes)
    axis_status = {
        axis: {"rows": axis_rows.get(axis, 0), "floor": floor,
               "meets_floor": axis_rows.get(axis, 0) >= floor}
        for axis, floor in AXIS_FLOORS.items()
    }
    code_langs = Counter(
        s.lang for s in SOURCES if s.axis == "code"
    )
    low_axes = [a for a, v in axis_status.items() if not v["meets_floor"]]
    if low_axes:
        stamp(f"  WARNING: axes below floor — {', '.join(low_axes)}")
    if non_english < NON_ENGLISH_FLOOR:
        stamp(f"  WARNING: non-English share {non_english:.1%} is below "
              f"{NON_ENGLISH_FLOOR:.0%}; the map is an English map")

    # -- 10. lineage -------------------------------------------------------
    crosswalk = None
    if args.previous.exists():
        stamp(f"\nCrosswalk against {args.previous.name} …")
        t0 = time.time()
        previous_atlas = Atlas.load(args.previous)
        previous_embedder = embedder.bind_idf(previous_atlas.token_log_prob)
        previous_assign = np.empty(n_kept, dtype=np.int32)
        for start in range(0, n_kept, BLOCK):
            stop = min(start + BLOCK, n_kept)
            rows = keep_index[start:stop]
            lo, hi = int(indptr[rows[0]]), int(indptr[rows[-1] + 1])
            # Re-pool only the kept rows under the previous IDF table.
            chunk_ids, chunk_ptr = [], [0]
            for r in rows:
                chunk_ids.append(np.asarray(tokens[indptr[r]:indptr[r + 1]], dtype=np.int32))
                chunk_ptr.append(chunk_ptr[-1] + int(indptr[r + 1] - indptr[r]))
            cache_block = TokenizedCorpus(
                token_ids=np.concatenate(chunk_ids) if chunk_ids else np.zeros(0, np.int32),
                indptr=np.asarray(chunk_ptr, dtype=np.int64),
                n_docs=len(rows), max_length=512,
            )
            del lo, hi
            vecs = previous_embedder.encode_tokenized(cache_block)
            _, _, nearest = previous_atlas.assign_full(vecs)
            previous_assign[start:stop] = nearest
        crosswalk = containment_crosswalk(
            assign, previous_assign, centres.shape[0], previous_atlas.n_regions,
            str(previous_atlas.meta.get("version", "unknown")),
        )
        timings["lineage_s"] = time.time() - t0
        stamp(f"  {crosswalk['summary']}")

    # -- 11. write ---------------------------------------------------------
    from sklearn.decomposition import PCA

    coords = PCA(n_components=2, random_state=SEED).fit(
        np.asarray(normed[np.sort(rng.choice(n_kept, min(100_000, n_kept), replace=False))])
    ).transform(centres).astype(np.float32)

    phash = pipeline_hash({
        "encoder_weight_hash": weight_hash,
        "normalization_variant": winner,
        "n_l1": N_L1,
        "l2_budget": L2_BUDGET,
        "atlas_version": VERSION,
    })
    timings["total_s"] = time.time() - wall0

    lang_order = sorted(norm.lang_means)
    lang_means = (
        np.vstack([norm.lang_means[lang] for lang in lang_order]).astype(np.float32)
        if lang_order else np.zeros((0, EMBED_DIM), dtype=np.float32)
    )

    meta = {
        "version": VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_hash": phash,
        "embed_model": DEFAULT_MODEL,
        "embed_dim": EMBED_DIM,
        "encoder_weight_hash": weight_hash,
        "pooling": "sif",
        "sif_a": SIF_A,
        "seed": SEED,
        "n_regions": int(centres.shape[0]),
        "n_l1": N_L1,
        "l2_k_range": [L2_K_MIN, L2_K_MAX],
        "l2_budget": L2_BUDGET,
        "l2_allocation": {str(k): int(v) for k, v in allocation.items()},
        "n_reference_records": int(n_kept),
        "off_atlas_threshold": off_threshold,
        "soft_k": SOFT_K,
        "soft_temperature": SOFT_TEMPERATURE,
        "soft_mean_regions_gt_0.15": soft_mean,
        "normalization": {
            "variant": winner,
            "steps": (["per_language_mean_removal"] if winner == "per_language"
                      else ["mean_removal"]) + ["all_but_the_top", "l2"],
            "pca_k": 2,
            "lang_labels": lang_order,
            "min_language_members": MIN_LANG_MEMBERS,
        },
        "probe_gate": probe_report,
        "ood_check": ood,
        "stability": stability,
        "coarse_fine_disagreement": round(coarse_disagreement, 4),
        "region_purity_by_axis": axis_purity,
        "region_source_purity": source_purity,
        "region_language_purity": language_purity,
        "source_cluster_ami": source_ami,
        "language_cluster_nmi": language_nmi,
        "non_english_share": round(non_english, 4),
        "axis_status": axis_status,
        "code_language_floor": CODE_LANGUAGE_FLOOR,
        "code_languages_configured": len(code_langs),
        "region_terms": region_terms,
        "l1_labels": l1_labels,
        "l1_terms": l1_terms,
        "l1_labels_source": ("curated:" + args.labels.name
                             if l1_labels != l1_terms else "tfidf_terms"),
        "family_sibling_overlap": [round(float(x), 4) for x in family_overlap],
        "sibling_overlap_max": SIBLING_OVERLAP_MAX,
        # The breakdown of the corpus that actually built the map, after both
        # dedup passes — not the pre-dedup count, which describes a corpus the
        # geometry never saw.
        "format_breakdown": dict(Counter(formats)),
        "format_breakdown_before_dedup": summary["formats"],
        "corpus_manifest": {
            "cache_format": manifest.get("cache_format"),
            "settings": manifest.get("settings"),
            "axes": manifest.get("axes"),
            "sources": [
                {k: m.get(k) for k in ("slug", "rows", "target", "complete",
                                       "shard_hash", "note", "error", "card")}
                for m in manifest.get("sources", [])
            ],
        },
        "dedup": {"minhash_jaccard": MINHASH_JACCARD,
                  "minhash_permutations": MINHASH_PERMUTATIONS,
                  "semantic_cosine": SEMANTIC_COSINE,
                  "semantic_is_approximate": "12-bit sign buckets"},
        "calibration": {
            "min_direct_members": MIN_CALIBRATION_MEMBERS,
            "sparse_cell_fallback": "local_plus_l1_parent_residuals",
            "directly_reliable_cells": int(refs_reliable.sum()),
            "distance_percentiles": DISTANCE_PERCENTILES.tolist(),
            "coarse_knot_percentiles": COARSE_KNOT_PERCENTILES.tolist(),
        },
        "prototype_percentiles": PROTOTYPE_PERCENTILES.tolist(),
        "source_labels": source_labels,
        "topic_labels": axis_labels,
        "language_labels": language_labels,
        "cooccurrence_neighbors": COOCCURRENCE_NEIGHBORS,
        "crosswalk": crosswalk,
        "timings_s": timings,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        centroids=centres,
        region_category=parents,
        region_size=region_size,
        l1_centroids=l1_centroids,
        l1_size=l1_size,
        coords=coords,
        norm_mean=norm.global_mean,
        norm_pca=norm.pca,
        norm_lang_means=lang_means,
        idf_token_ids=idf_ids,
        idf_log_probs=idf_lps,
        distance_refs=distance_refs,
        distance_refs_support=refs_support,
        distance_refs_reliable=refs_reliable,
        coarse_knots=coarse_knots,
        coarse_expected=coarse_expected,
        family_distinguishable=family_ok,
        family_sibling_overlap=family_overlap,
        cell_source_counts=cell_source_counts,
        cell_topic_counts=cell_topic_counts,
        cell_language_counts=cell_language_counts,
        cooccurrence_ids=cooc_ids,
        cooccurrence_scores=cooc_scores,
        prototype_vectors=proto_vectors,
        prototype_record_ids=proto_ids,
        prototype_distances=proto_distances,
        probe_coef=np.zeros((0, EMBED_DIM), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        # NB: no allow_pickle kwarg. savez_compressed treats every keyword as
        # an array to save, so passing it stores a stray array literally named
        # "allow_pickle" instead of configuring anything. Pickling is enabled on
        # the load side, which is where the flag actually exists.
        meta=np.array([json.dumps(meta)], dtype=object),
    )
    size_mb = args.out.stat().st_size / 1e6
    stamp(f"\nWrote {args.out} ({size_mb:.2f} MB)")
    if size_mb > MAX_ARTIFACT_MB:
        stamp(f"WARNING: bundle is over the {MAX_ARTIFACT_MB:.0f} MB budget")

    notes_path = args.out.with_name(args.out.stem + "-release-notes.json")
    notes_path.write_text(json.dumps({
        "version": VERSION,
        "pipeline_hash": phash,
        "encoder_weight_hash": weight_hash,
        "records": int(n_kept),
        "cells": int(centres.shape[0]),
        "coarse_regions": N_L1,
        "normalization_winner": winner,
        "probe_gate": probe_report,
        "ood_check": ood,
        "stability": stability,
        "coarse_fine_disagreement": round(coarse_disagreement, 4),
        "axis_status": axis_status,
        "non_english_share": round(non_english, 4),
        "purity": {"axis": axis_purity, "source": source_purity,
                   "language": language_purity},
        "continuity": crosswalk["summary"] if crosswalk else None,
        "l2_allocation": {str(k): int(v) for k, v in allocation.items()},
        "silhouette_curves": curves,
        "timings_s": timings,
    }, indent=2) + "\n", encoding="utf-8")
    stamp(f"Release notes: {notes_path}")

    stamp("\n=== TIMING ===")
    for key, value in timings.items():
        stamp(f"  {key:<22} {value:>8.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
