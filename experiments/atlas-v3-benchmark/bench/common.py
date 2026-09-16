"""Shared harness: products, encoding, placement cache, language detection, metrics.

Every benchmark places the same texts on each product through the exact code
path ``dropoutt atlas`` uses: profile window -> SIF-weighted encoder bound to
the product's own IDF table -> frozen normalisation (per-language centering
when the artifact ships it) -> nearest fine cell, off-atlas cutoff, soft top-k.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
RESULTS = EXP / "results"
RUNS = EXP / "runs"
ATLAS_DIR = REPO / "src" / "dropoutt" / "data" / "atlas"
SCRATCH = Path(os.environ.get(
    "BENCH_SCRATCH",
    "/private/tmp/claude-501/-Users-crokan-Documents-dropoutt-cli/b585583f-4892-462f-a850-64929286d9e7/scratchpad/bench-cache",
))
#: The build's own outputs (USB volume): reservoirs for cell audits, the
#: un-stripped artifact with exemplar texts, and the map this build replaced.
CK512 = Path("/Volumes/ck512/dropoutt-atlas-v2")
RESERVOIRS = {
    "atlas-v3": CK512 / "work-v3-textnorm" / "reservoir.jsonl",
    "atlas-v3-prev": CK512 / "work" / "reservoir.jsonl",
}
EXEMPLAR_ARTIFACT = CK512 / "release-v3-textnorm" / "atlas-v3.npz"
SCRATCH.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

#: Minimum characters for placement, mirrored from dropoutt.runner.
MIN_PLACE_CHARS = 80

#: Products under test. ``atlas-v3`` is the shipped 14 Sep 2026 rebuild (corpus
#: 99687a41, encoder input policy v1, length-weighted fit). ``atlas-v3-prev`` is
#: the map it replaced (corpus 4203fc3a, shipped 12-13 Sep 2026), kept beside the
#: build outputs on the USB volume.
PRODUCTS = {
    "atlas-v3": ATLAS_DIR / "atlas-v3.npz",
    "atlas-v3-prev": CK512 / "release-v3-textnorm" / "previous-shipped-4203fc3a" / "atlas-v3.npz",
    "atlas-v2": ATLAS_DIR / "atlas-v2.npz",
    "atlas-v2-lite": ATLAS_DIR / "atlas-v2-lite.npz",
}
MAIN = "atlas-v3"

_ATLAS: dict[str, object] = {}
_EMB: dict[str, object] = {}


def atlas(name: str = MAIN):
    from dropoutt.atlas.apply import Atlas

    if name not in _ATLAS:
        _ATLAS[name] = Atlas.load(PRODUCTS[name])
    return _ATLAS[name]


def embedder(name: str = MAIN):
    from dropoutt.atlas import load_embedder

    if name not in _EMB:
        a = atlas(name)
        e = load_embedder(a.embed_model, offline=True, out_dim=a.dim)
        if e is None:
            raise RuntimeError("encoder not cached; run `dropoutt fetch`")
        _EMB[name] = a.bind_embedder(e)
    return _EMB[name]


def encode(name: str, texts: list[str]) -> np.ndarray:
    a = atlas(name)
    e = embedder(name)
    profile = a.profile
    return e.encode(
        texts,
        weighted=(profile.pooling == "sif") if profile is not None else None,
        max_chars=profile.max_chars if profile is not None else None,
        max_tokens=profile.max_tokens if profile is not None else 512,
    )


_DETECTOR = None


def detect_langs(texts: list[str]) -> list[str]:
    """Per-record detected language, the way the runner feeds it to assign_all."""
    global _DETECTOR
    from dropoutt.langid import LanguageDetector

    if _DETECTOR is None:
        _DETECTOR = LanguageDetector()
    out = []
    for start in range(0, len(texts), 20000):
        out.extend(r.lang for r in _DETECTOR.detect_many(texts[start:start + 20000]))
    return out


def fingerprint(name: str) -> str:
    """Identity of the artifact file behind a product name.

    Caches are keyed by it as well as by name: a rebuilt or restamped
    ``atlas-v3.npz`` keeps its name, and placements cached against the previous
    file would otherwise be read back as results for the new one.
    """
    stat = PRODUCTS[name].stat() if name in PRODUCTS and PRODUCTS[name].exists() else None
    return f"{stat.st_size}:{stat.st_mtime_ns}" if stat else "absent"


def _key(name: str, texts: list[str], tag: str) -> str:
    h = hashlib.blake2b(digest_size=12)
    h.update(name.encode()); h.update(tag.encode()); h.update(str(len(texts)).encode())
    h.update(fingerprint(name).encode())
    for t in texts[::max(1, len(texts) // 512)]:
        h.update(t[:200].encode("utf-8", "replace"))
    return h.hexdigest()


class Placed:
    """Placement of one text list on one product."""

    def __init__(self, name, emb, best, score, nearest, cat, soft_cells, soft_w, langs):
        self.name = name
        self.emb = emb
        self.best = best            # fine cell or -1
        self.score = score
        self.nearest = nearest
        self.l1 = cat               # parent of nearest cell (placed or not)
        self.soft_cells = soft_cells
        self.soft_w = soft_w
        self.langs = langs

    @property
    def placed(self) -> np.ndarray:
        return self.best >= 0

    @property
    def off_rate(self) -> float:
        return float((~self.placed).mean()) if len(self.best) else 0.0

    @property
    def l1_placed(self) -> np.ndarray:
        """L1 of placed records, -1 where off-atlas."""
        return np.where(self.placed, self.l1, -1)


def place(name: str, texts: list[str], langs: list[str] | None = None, *,
          tag: str = "", use_langs: bool = True, cache: bool = True) -> Placed:
    """Encode + assign, cached on disk by content."""
    if langs is None and use_langs:
        langs = cached_langs(texts, tag=tag)
    key = _key(name, texts, tag + ("|L" if use_langs else "|G"))
    path = SCRATCH / f"place-{key}.npz"
    if cache and path.exists():
        d = np.load(path, allow_pickle=True)
        return _recut(name, Placed(name, d["emb"], d["best"], d["score"], d["nearest"], d["cat"],
                                   d["soft_cells"], d["soft_w"], list(d["langs"])))
    a = atlas(name)
    emb = encode(name, texts)
    r = a.assign_all(emb, langs if use_langs else None)
    p = Placed(name, emb, r.best, r.score, r.nearest, r.categories, r.soft_cells,
               r.soft_weights, langs or ["unknown"] * len(texts))
    if cache:
        np.savez(path, emb=emb, best=r.best, score=r.score, nearest=r.nearest,
                 cat=r.categories, soft_cells=r.soft_cells, soft_w=r.soft_weights,
                 langs=np.array(p.langs))
    return p


def _recut(name: str, p: Placed) -> Placed:
    """Re-apply the product's *current* off-atlas cutoff to a cached placement.

    The cutoff is stamped in the artifact and can change without the centroids
    changing (atlas-v3 was calibrated to 0.3538 after the first run). ``score``
    and ``nearest`` do not depend on it, so the cached placement is exact once
    ``best`` and the soft weights are recomputed from the live threshold.
    """
    thr = atlas(name).off_threshold
    off = p.score < thr
    p.best = np.where(off, -1, p.nearest).astype(p.best.dtype)
    soft_cells = p.soft_cells.copy(); soft_w = p.soft_w.copy()
    soft_cells[off] = -1; soft_w[off] = 0.0
    p.soft_cells, p.soft_w = soft_cells, soft_w
    return p


def cached_langs(texts: list[str], tag: str = "") -> list[str]:
    key = _key("langid", texts, tag)
    path = SCRATCH / f"langs-{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    langs = detect_langs(texts)
    path.write_bytes(pickle.dumps(langs))
    return langs


# ---- metrics -----------------------------------------------------------

def nmi(a, b) -> float:
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))


def ami(a, b) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    return float(adjusted_mutual_info_score(a, b))


def purity(cells, labels) -> float:
    """Share of records whose cell's majority label is their label."""
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for c, l in zip(cells, labels):
        by[c][l] += 1
    return sum(cnt.most_common(1)[0][1] for cnt in by.values()) / max(len(cells), 1)


def majority_map_accuracy(train_cells, train_labels, test_cells, test_labels,
                          fallback=None) -> float:
    """Map each cell to its majority training label; score on held-out rows."""
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for c, l in zip(train_cells, train_labels):
        by[c][l] += 1
    if fallback is None:
        fallback = Counter(train_labels).most_common(1)[0][0]
    table = {c: cnt.most_common(1)[0][0] for c, cnt in by.items()}
    hits = sum(1 for c, l in zip(test_cells, test_labels) if table.get(c, fallback) == l)
    return hits / max(len(test_cells), 1)


def linear_probe_accuracy(x_train, y_train, x_test, y_test, seed=0) -> float:
    """Upper bound: a logistic-regression probe on the same projected vectors."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, C=4.0, n_jobs=-1)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def centroid_probe_accuracy(x_train, y_train, x_test, y_test) -> float:
    """Nearest-class-mean on the same vectors: the cheapest supervised map."""
    labels = sorted(set(y_train))
    idx = {l: i for i, l in enumerate(labels)}
    y = np.array([idx[l] for l in y_train])
    means = np.zeros((len(labels), x_train.shape[1]), dtype=np.float32)
    for i in range(len(labels)):
        m = x_train[y == i].mean(axis=0)
        means[i] = m / (np.linalg.norm(m) + 1e-9)
    pred = (x_test @ means.T).argmax(axis=1)
    yt = np.array([idx.get(l, -1) for l in y_test])
    return float((pred == yt).mean())


def histogram(cells, n) -> np.ndarray:
    cells = np.asarray(cells)
    cells = cells[cells >= 0]
    h = np.bincount(cells, minlength=n).astype(np.float64)
    return h / max(h.sum(), 1.0)


def cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def chance_agreement(hist: np.ndarray) -> float:
    """P(two independent draws from this histogram land in the same bin)."""
    return float((hist ** 2).sum())


# ---- uncertainty ---------------------------------------------------------

def bootstrap_ci(hits, reps: int = 1000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of a mean over per-record 0/1 (or real) values."""
    hits = np.asarray(hits, dtype=np.float64)
    if len(hits) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(hits), size=(reps, len(hits)))
    means = hits[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def paired_bootstrap_delta(hits_a, hits_b, reps: int = 1000, seed: int = 0):
    """CI of mean(a) - mean(b) when both are scored on the same records.

    Pairing removes the record-sampling noise the two maps share, so the
    interval says whether the *maps* differ, not whether the test set does.
    """
    a = np.asarray(hits_a, dtype=np.float64); b = np.asarray(hits_b, dtype=np.float64)
    assert len(a) == len(b)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(d)), "significant": bool(lo > 0 or hi < 0)}


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def retained(acc: float, chance: float, ceiling: float) -> float | None:
    """Share of the ceiling's headroom over chance that the frozen grid keeps."""
    if ceiling is None or ceiling - chance <= 1e-9:
        return None
    return float((acc - chance) / (ceiling - chance))


def majority_map_hits(train_cells, train_labels, test_cells, test_labels, fallback=None) -> np.ndarray:
    """Per-record correctness of the majority-vote reading, for CIs and pairing."""
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for c, l in zip(train_cells, train_labels):
        by[c][l] += 1
    if fallback is None:
        fallback = Counter(train_labels).most_common(1)[0][0]
    table = {c: cnt.most_common(1)[0][0] for c, cnt in by.items()}
    return np.array([table.get(c, fallback) == l for c, l in zip(test_cells, test_labels)], dtype=bool)


def split_halves(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return idx[: n // 2], idx[n // 2:]


def save(name: str, payload: dict) -> Path:
    payload = {"_written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_default))
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def product_info(name: str) -> dict:
    a = atlas(name)
    return {
        "version": a.meta.get("version"),
        "corpus_hash": str(a.meta.get("corpus_hash", ""))[:8],
        "n_reference_records": a.meta.get("n_reference_records"),
        "n_l1": a.n_l1,
        "n_cells": a.n_regions,
        "dim": a.dim,
        "off_threshold": a.off_threshold,
        "per_language_centering": a.uses_language_centering,
        "n_lang_means": len(a.lang_labels),
        "has_l2_names": bool(a.meta.get("region_labels")),
        "has_l1_names": bool(a.meta.get("l1_labels")),
        "encoder_input_policy": (a.meta.get("encoder_input") or {}).get("version"),
        "fit_weights": a.meta.get("fit_weights"),
        "sha256_12": hashlib.sha256(PRODUCTS[name].read_bytes()).hexdigest()[:12],
    }
