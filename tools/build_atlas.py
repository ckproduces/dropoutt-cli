#!/usr/bin/env python3
"""Build atlas-lite-v1: a frozen topic coordinate system.

Pipeline (byte-identical on client and build):

    ingest → detect format → extract text → chunk → dedup → embed
           → normalize → assign to cells → aggregate

Geometry is a two-level k-means hierarchy on cosine distance:

    L1 (lite)  ~40 regions
    L2 (full)  ~12 per L1, ~480 fine cells — exact children of L1

The reference corpus is stratified across code, math, instruction/chat,
legal/finance, scientific, dialogue/forum, structured/tabular, and
multilingual prose. Densest sources are capped so they cannot define the map.

Usage:
    python tools/build_atlas.py --out src/dropoutt/data/atlas/atlas-lite-v1.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dropoutt.atlas.embed import DEFAULT_MODEL, embedding_weight_hash, load as load_embedder
from dropoutt.atlas.extract import extract_from_fields, extract_text
from dropoutt.atlas.normalize import EMBED_DIM, SIF_A, fit_norm
from dropoutt.atlas.pipeline import PIPELINE_VERSION, pipeline_hash

SEED = 20260801
N_L1 = 40
N_L2_PER_L1 = 12
SOFT_K = 5
SOFT_TEMPERATURE = 0.08
MINHASH_JACCARD = 0.8
SEMANTIC_COSINE = 0.95
MIN_CHARS = 80
IDF_TOP_TOKENS = 120_000

# (hf_id, config, split, text_fields, topic_tag, target_samples, language_hint)
# topic_tag is provenance for the manifest and for stratified caps — clustering
# is unsupervised. Sample languages deliberately so EN/Python/JS cannot dominate.
SOURCES: list[tuple[str, str | None, str, tuple[str, ...], str, int, str]] = [
    # Reliable parquet sources first so a hung shard cannot stall the build.
    # -- math (own region, not absorbed into academic) ----------------------
    ("open-web-math/open-web-math", None, "train", ("text",), "math", 8000, "en"),
    ("HuggingFaceTB/finemath", "finemath-3plus", "train", ("text",), "math", 8000, "en"),
    ("openai/gsm8k", "main", "train", ("question", "answer"), "math", 5000, "en"),
    ("EleutherAI/hendrycks_math", "algebra", "train", ("problem", "solution"), "math", 2000, "en"),
    ("EleutherAI/hendrycks_math", "geometry", "train", ("problem", "solution"), "math", 1000, "en"),
    ("microsoft/orca-math-word-problems-200k", None, "train", ("question", "answer"), "math", 6000, "en"),
    # -- instruction / chat (the gap Wikipedia-thinking misses) -------------
    ("OpenAssistant/oasst1", None, "train", ("text",), "instruction", 8000, "multi"),
    ("openbmb/UltraFeedback", None, "train", ("instruction",), "instruction", 8000, "en"),
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", ("prompt",), "instruction", 8000, "en"),
    ("tatsu-lab/alpaca", None, "train", ("instruction", "output"), "instruction", 6000, "en"),
    ("databricks/databricks-dolly-15k", None, "train", ("instruction", "response"), "instruction", 5000, "en"),
    ("Anthropic/hh-rlhf", None, "train", ("chosen",), "instruction", 4000, "en"),
    ("turkish-nlp-suite/InstrucTurca", None, "train", ("Input", "Output"), "instruction", 8000, "tr"),
    ("merve/turkish_instructions", None, "train", ("talimat", "çıktı"), "instruction", 5000, "tr"),
    ("TFLai/Turkish-Alpaca", None, "train", ("instruction", "output"), "instruction", 5000, "tr"),
    # -- code (Stack/StarCoder-class; sample across langs when available) ---
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train", ("instruction", "output"), "code", 5000, "en"),
    ("google-research-datasets/mbpp", "full", "train", ("text", "code"), "code", 400, "en"),
    ("openai/openai_humaneval", None, "test", ("prompt",), "code", 164, "en"),
    ("bigcode/the-stack-smol-xs", None, "train", ("content",), "code", 6000, "code"),
    ("codeparrot/codeparrot-clean-valid", None, "train", ("content",), "code", 3000, "code"),
    # -- legal / finance / admin (Common Corpus-shaped region) --------------
    ("gbharti/finance-alpaca", None, "train", ("instruction", "output"), "legal_finance", 5000, "en"),
    ("albertvillanova/legal_contracts", None, "train", ("text",), "legal_finance", 4000, "en"),
    ("lex_glue", "ecthr_a", "train", ("text",), "legal_finance", 4000, "en"),
    # -- scientific (arXiv, PubMed; peS2o script-loader is retired on Hub) --
    ("CShorten/ML-ArXiv-Papers", None, "train", ("title", "abstract"), "scientific", 8000, "en"),
    ("qiaojin/PubMedQA", "pqa_labeled", "train", ("question", "long_answer"), "scientific", 1000, "en"),
    ("camel-ai/physics", None, "train", ("message_1", "message_2"), "scientific", 3000, "en"),
    ("camel-ai/biology", None, "train", ("message_1", "message_2"), "scientific", 3000, "en"),
    ("bigbio/med_qa", "med_qa_en_source", "train", ("question", "answer"), "scientific", 2000, "en"),
    # -- dialogue / forum (Q&A register) ------------------------------------
    ("HuggingFaceH4/stack-exchange-preferences", None, "train", ("question",), "dialogue", 8000, "en"),
    ("rajpurkar/squad", None, "train", ("context", "question"), "dialogue", 6000, "en"),
    ("tau/commonsense_qa", None, "train", ("question",), "dialogue", 4000, "en"),
    # -- structured / tabular (exercise the extractor) ----------------------
    ("b-mc2/sql-create-context", None, "train", ("question", "answer"), "structured", 4000, "en"),
    ("Salesforce/wikitablequestions", None, "train", ("question", "table"), "structured", 3000, "en"),
    # -- multilingual prose / culture (breadth, not English-only) -----------
    ("wikimedia/wikipedia", "20231101.tr", "train", ("text",), "prose", 8000, "tr"),
    ("wikimedia/wikipedia", "20231101.en", "train", ("text",), "prose", 6000, "en"),
    ("wikimedia/wikipedia", "20231101.ar", "train", ("text",), "prose", 4000, "ar"),
    ("wikimedia/wikipedia", "20231101.de", "train", ("text",), "prose", 4000, "de"),
    ("wikimedia/wikipedia", "20231101.es", "train", ("text",), "prose", 4000, "es"),
    ("wikimedia/wikipedia", "20231101.fr", "train", ("text",), "prose", 4000, "fr"),
    ("wikimedia/wikipedia", "20231101.zh", "train", ("text",), "prose", 4000, "zh"),
    ("wikimedia/wikipedia", "20231101.az", "train", ("text",), "prose", 3000, "az"),
    ("mcemilg/news-cat", None, "train", ("text",), "prose", 4000, "tr"),
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", ("text",), "prose", 5000, "en"),
]


def _call_with_timeout(fn, timeout: float, label: str):
    """Run ``fn()`` in a daemon thread; raise TimeoutError on stall."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            q.put(("ok", fn()))
        except Exception as exc:
            q.put(("err", exc))

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TimeoutError(f"{label} exceeded {timeout:.0f}s")
    kind, payload = q.get_nowait()
    if kind == "err":
        raise payload
    return payload


def _iter_with_timeout(ds, per_source_seconds: float):
    """Iterate a streaming dataset, aborting if the next row blocks too long.

    HuggingFace streaming can hang on a single shard fetch. A worker thread
    plus a join timeout lets us skip that source instead of stalling the build.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=16)
    sentinel = object()

    def worker() -> None:
        try:
            for row in ds:
                q.put(row)
            q.put(sentinel)
        except Exception as exc:
            q.put(exc)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    deadline = time.time() + per_source_seconds
    while time.time() < deadline:
        timeout = min(20.0, max(0.1, deadline - time.time()))
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"no row within {timeout:.0f}s (source budget {per_source_seconds:.0f}s)"
            )
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item
    raise TimeoutError(f"source budget {per_source_seconds:.0f}s exhausted")


def collect(sources, verbose: bool = True, per_source_seconds: float = 90.0):
    """Stream a stratified sample. Failures are recorded, not fatal."""
    import os

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "0")
    from datasets import load_dataset

    texts: list[str] = []
    tags: list[str] = []
    formats: list[str] = []
    source_ids: list[str] = []
    langs: list[str] = []
    manifest: list[dict] = []
    record_ids: list[str] = []

    for hf_id, config, split, fields, tag, target, lang in sources:
        t0 = time.time()
        got = 0
        note = None
        if verbose:
            print(f"  … {hf_id[:50]}", flush=True)
        error = None
        try:
            ds = _call_with_timeout(
                lambda: load_dataset(hf_id, config, split=split, streaming=True),
                timeout=min(45.0, per_source_seconds),
                label=f"load_dataset({hf_id})",
            )
            deadline = t0 + per_source_seconds
            for row in _iter_with_timeout(ds, per_source_seconds):
                if time.time() >= deadline:
                    note = "time budget reached"
                    break
                # Fast path for ordinary prose fields: skip format detection.
                # The detector still runs for structured/code sources and on
                # the client for raw files.
                text, fmt = "", "plain"
                if len(fields) == 1 and isinstance(row.get(fields[0]), str):
                    candidate = row[fields[0]].strip()
                    if len(candidate) >= MIN_CHARS:
                        text, fmt = candidate, "plain"
                if not text:
                    text, fmt = extract_from_fields(row, fields)
                if not text:
                    raw = "\n".join(
                        str(row[f]) for f in fields if f in row and row[f] is not None
                    )
                    text, fmt = extract_text(raw)
                if len(text) < MIN_CHARS:
                    continue
                # Reference corpus: one truncated window per record. The client
                # still ships the full chunker for raw documents.
                piece = text[:2000]
                rid = hashlib.blake2b(
                    f"{hf_id}:{got}:{piece[:200]}".encode("utf-8"), digest_size=8
                ).hexdigest()
                texts.append(piece)
                tags.append(tag)
                formats.append(fmt)
                source_ids.append(hf_id)
                langs.append(lang)
                record_ids.append(rid)
                got += 1
                if verbose and (got in (1, 100, 500) or got % 2000 == 0):
                    print(f"    {got:,} rows…", flush=True)
                if got >= target:
                    break
            if got < target and note is None and time.time() >= deadline:
                note = "time budget reached"
        except Exception as exc:
            error = str(exc)[:200]
            if got == 0 and verbose:
                print(
                    f"  SKIP {hf_id[:42]:<42} {type(exc).__name__}: {str(exc)[:70]}",
                    flush=True,
                )
            elif got > 0:
                note = f"partial: {type(exc).__name__}"
        if got == 0 and error:
            manifest.append({
                "hf_id": hf_id, "config": config, "split": split,
                "topic": tag, "lang": lang, "collected": 0, "error": error,
            })
            continue
        if verbose:
            suffix = f"  ({note})" if note else ""
            print(
                f"  {hf_id[:42]:<42} {tag:<14} {lang:<6} {got:>6,}  "
                f"{time.time() - t0:>5.1f}s{suffix}",
                flush=True,
            )
        entry = {
            "hf_id": hf_id, "config": config, "split": split,
            "topic": tag, "lang": lang, "collected": got, "note": note,
        }
        if error:
            entry["error"] = error
        manifest.append(entry)
    return texts, tags, formats, source_ids, langs, record_ids, manifest


def minhash_dedup(texts: list[str], threshold: float = MINHASH_JACCARD) -> np.ndarray:
    """Near-exact dedup via MinHash LSH over word shingles. Returns keep mask."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        # Fallback: exact normalised-text hash.
        seen: set[str] = set()
        keep = np.ones(len(texts), dtype=bool)
        for i, t in enumerate(texts):
            key = re.sub(r"\s+", " ", t.lower())[:500]
            h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
            if h in seen:
                keep[i] = False
            else:
                seen.add(h)
        return keep

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    keep = np.ones(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        mh = MinHash(num_perm=64)
        words = t.lower().split()
        for shingle in zip(words, words[1:], words[2:]):
            mh.update(" ".join(shingle).encode("utf-8"))
        key = str(i)
        try:
            if lsh.query(mh):
                keep[i] = False
            else:
                lsh.insert(key, mh)
        except ValueError:
            keep[i] = False
    return keep


def semantic_dedup(emb: np.ndarray, threshold: float = SEMANTIC_COSINE) -> np.ndarray:
    """Greedy cosine dedup. O(n²) on small sets; bucketed on large ones."""
    n = len(emb)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    # Bucket by sign of first few dims to avoid full n² on 100k+.
    bits = (emb[:, :8] > 0).astype(np.uint8)
    keys = bits.dot(1 << np.arange(8))
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(keys.tolist()):
        buckets[int(k)].append(i)
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        sub = emb[idxs]
        # Pairwise within bucket
        sims = sub @ sub.T
        for a in range(len(idxs)):
            if not keep[idxs[a]]:
                continue
            for b in range(a + 1, len(idxs)):
                if keep[idxs[b]] and sims[a, b] >= threshold:
                    keep[idxs[b]] = False
    return keep


def top_terms(texts: list[str], k: int = 5, df: Counter[str] | None = None) -> str:
    """Distinctive terms: in-region frequency against document frequency."""
    stop = {
        "that", "this", "with", "from", "have", "will", "your", "what", "which",
        "their", "they", "them", "about", "into", "than", "then", "when", "were",
        "been", "also", "some", "would", "could", "should", "there", "these",
        "olarak", "için", "gibi", "daha", "olan", "ancak", "veya", "değil",
        "return", "function", "import", "class", "const", "print",
    }
    tf: Counter[str] = Counter()
    rng = np.random.default_rng(SEED)
    sample = texts if len(texts) <= 200 else [
        texts[i] for i in rng.choice(len(texts), 200, replace=False)
    ]
    for t in sample:
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop:
                tf[w] += 1
    if not tf:
        return ""
    if df is None:
        return ", ".join(w for w, _ in tf.most_common(k))
    df_total = float(sum(df.values()) or 1)
    scored = [
        (c * np.log(1.0 + df_total / (1 + df.get(w, 0))), w)
        for w, c in tf.items()
    ]
    scored.sort(reverse=True)
    return ", ".join(w for _, w in scored[:k])


def label_regions(
    texts: list[str],
    assign: np.ndarray,
    n_regions: int,
) -> list[str]:
    df: Counter[str] = Counter()
    members: list[list[str]] = [[] for _ in range(n_regions)]
    for t, a in zip(texts, assign, strict=True):
        if 0 <= a < n_regions:
            members[a].append(t)
    # First pass: document frequencies
    for group in members:
        rng = np.random.default_rng(SEED)
        sample = group if len(group) <= 200 else [
            group[i] for i in rng.choice(len(group), 200, replace=False)
        ]
        for t in sample:
            seen: set[str] = set()
            for w in t.lower().split():
                w = "".join(ch for ch in w if ch.isalpha())
                if len(w) > 3:
                    seen.add(w)
            df.update(seen)
    return [top_terms(m, df=df) if m else "" for m in members]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("src/dropoutt/data/atlas/atlas-lite-v1.npz"))
    ap.add_argument("--l1", type=int, default=N_L1)
    ap.add_argument("--l2-per-l1", type=int, default=N_L2_PER_L1)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every target sample size.")
    ap.add_argument("--budget", type=float, default=120.0, help="Seconds per source.")
    ap.add_argument("--timing-log", type=Path, default=Path("src/dropoutt/data/atlas/build-timing.json"))
    args = ap.parse_args()

    timings: dict[str, float] = {}
    wall0 = time.time()

    sources = [
        (h, c, s, f, k, max(50, int(n * args.scale)), lang)
        for h, c, s, f, k, n, lang in SOURCES
    ]
    print(f"Collecting reference corpus from {len(sources)} sources...")
    t0 = time.time()
    texts, tags, formats, source_ids, langs, record_ids, manifest = collect(
        sources, per_source_seconds=args.budget
    )
    timings["collect_s"] = time.time() - t0
    print(
        f"\nCollected {len(texts):,} records across "
        f"{len(set(tags))} topic tags / {len(set(langs))} language hints "
        f"in {timings['collect_s']:.1f}s"
    )
    if len(texts) < 1000:
        print("Too few records collected to build a usable atlas.")
        return 1

    fmt_break = Counter(formats)
    print(f"  format breakdown: {dict(fmt_break.most_common(8))}")

    # -- Dedup (near-exact) ------------------------------------------------
    print("\nNear-exact dedup (MinHash / hash)...")
    t0 = time.time()
    keep = minhash_dedup(texts)
    texts = [t for t, k in zip(texts, keep, strict=True) if k]
    tags = [t for t, k in zip(tags, keep, strict=True) if k]
    formats = [t for t, k in zip(formats, keep, strict=True) if k]
    source_ids = [t for t, k in zip(source_ids, keep, strict=True) if k]
    langs = [t for t, k in zip(langs, keep, strict=True) if k]
    record_ids = [t for t, k in zip(record_ids, keep, strict=True) if k]
    timings["dedup_minhash_s"] = time.time() - t0
    print(f"  kept {len(texts):,} after minhash ({timings['dedup_minhash_s']:.1f}s)")

    # -- Embed (mean first, for IDF fit we need token stats separately) ----
    print(f"\nLoading embedder {DEFAULT_MODEL} ...")
    embedder = load_embedder(DEFAULT_MODEL, out_dim=EMBED_DIM)
    if embedder is None:
        print("Could not load embedder.")
        return 1
    weight_hash = embedder.weight_hash or embedding_weight_hash(DEFAULT_MODEL)

    print("Batch-tokenizing reference corpus once...")
    t0 = time.time()
    tokenized = embedder.tokenize(texts)
    timings["tokenize_s"] = time.time() - t0
    print(
        f"  {tokenized.n_tokens:,} token occurrences in "
        f"{timings['tokenize_s']:.1f}s"
    )

    print("Fitting IDF table from cached token IDs...")
    t0 = time.time()
    token_log_prob, idf_ids, idf_lps = embedder.token_log_prob(
        tokenized, max_tokens=IDF_TOP_TOKENS
    )
    timings["idf_s"] = time.time() - t0
    print(f"  {len(token_log_prob):,} token types in {timings['idf_s']:.1f}s")
    embedder = embedder.bind_idf(token_log_prob)

    print(f"\nEmbedding {len(texts):,} records (SIF-weighted, dim={EMBED_DIM}) ...")
    t0 = time.time()
    emb_raw = embedder.encode_tokenized(tokenized)
    timings["embed_s"] = time.time() - t0
    print(
        f"  {emb_raw.shape} in {timings['embed_s']:.1f}s "
        f"({len(texts) / max(timings['embed_s'], 0.001):,.0f} records/s)"
    )

    # -- Semantic dedup ----------------------------------------------------
    print("\nSemantic dedup (cosine)...")
    t0 = time.time()
    # Temporary L2 for cosine dedup only
    tmp = emb_raw / (np.linalg.norm(emb_raw, axis=1, keepdims=True) + 1e-9)
    keep = semantic_dedup(tmp, SEMANTIC_COSINE)
    emb_raw = emb_raw[keep]
    texts = [t for t, k in zip(texts, keep, strict=True) if k]
    tags = [t for t, k in zip(tags, keep, strict=True) if k]
    formats = [t for t, k in zip(formats, keep, strict=True) if k]
    source_ids = [t for t, k in zip(source_ids, keep, strict=True) if k]
    langs = [t for t, k in zip(langs, keep, strict=True) if k]
    record_ids = [t for t, k in zip(record_ids, keep, strict=True) if k]
    timings["dedup_semantic_s"] = time.time() - t0
    print(f"  kept {len(texts):,} after semantic dedup ({timings['dedup_semantic_s']:.1f}s)")

    # -- Normalize ---------------------------------------------------------
    print("\nFitting normalization (mean + top-2 PCA) ...")
    t0 = time.time()
    norm = fit_norm(emb_raw, pca_k=2, dim=EMBED_DIM)
    emb = norm.apply(emb_raw)
    timings["normalize_s"] = time.time() - t0

    # -- L1 k-means --------------------------------------------------------
    from sklearn.cluster import MiniBatchKMeans

    print(f"\nFitting L1 ({args.l1} regions) ...")
    t0 = time.time()
    l1 = MiniBatchKMeans(
        n_clusters=args.l1, random_state=SEED, batch_size=4096, n_init=3, max_iter=200
    ).fit(emb)
    l1_labels_idx = l1.labels_.astype(np.int32)
    l1_centroids = l1.cluster_centers_.astype(np.float32)
    l1_centroids /= np.linalg.norm(l1_centroids, axis=1, keepdims=True) + 1e-9
    timings["cluster_l1_s"] = time.time() - t0
    print(f"  done in {timings['cluster_l1_s']:.1f}s")

    # -- L2 k-means within each L1 -----------------------------------------
    print(f"\nFitting L2 (~{args.l2_per_l1} per L1) ...")
    t0 = time.time()
    centroids: list[np.ndarray] = []
    region_category: list[int] = []
    l2_assign = np.full(len(emb), -1, dtype=np.int32)
    next_id = 0
    for cat in range(args.l1):
        idx = np.where(l1_labels_idx == cat)[0]
        if len(idx) == 0:
            continue
        k = min(args.l2_per_l1, max(1, len(idx) // 30))
        if len(idx) < k * 5:
            k = max(1, len(idx) // 5)
        sub = emb[idx]
        km = MiniBatchKMeans(
            n_clusters=k, random_state=SEED + cat, batch_size=min(2048, max(len(sub), 1)),
            n_init=3, max_iter=150,
        ).fit(sub)
        for c in range(k):
            centroids.append(km.cluster_centers_[c])
            region_category.append(cat)
            members = idx[km.labels_ == c]
            l2_assign[members] = next_id
            next_id += 1
    C = np.vstack(centroids).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    region_category_arr = np.array(region_category, dtype=np.int32)
    timings["cluster_l2_s"] = time.time() - t0
    print(f"  {C.shape[0]} fine cells across {args.l1} L1 regions in {timings['cluster_l2_s']:.1f}s")

    # -- Labels ------------------------------------------------------------
    print("\nLabelling regions ...")
    t0 = time.time()
    region_terms = label_regions(texts, l2_assign, C.shape[0])
    # L1 labels from members of each L1
    l1_terms = []
    for cat in range(args.l1):
        members = [texts[i] for i in range(len(texts)) if l1_labels_idx[i] == cat]
        l1_terms.append(top_terms(members) if members else f"region-{cat}")
    timings["label_s"] = time.time() - t0

    # -- Calibration -------------------------------------------------------
    print("\nCalibrating off-atlas threshold and distance refs ...")
    sims = emb @ C.T
    best = sims.max(axis=1)
    off_threshold = float(np.percentile(best, 2.0))
    assign = sims.argmax(axis=1)
    region_size = np.bincount(assign, minlength=C.shape[0]).astype(np.int32)
    l1_size = np.bincount(l1_labels_idx, minlength=args.l1).astype(np.int32)

    distance_refs = np.zeros((C.shape[0], 5), dtype=np.float32)
    for r in range(C.shape[0]):
        members = np.where(assign == r)[0]
        if len(members) < 5:
            continue
        dists = 1.0 - sims[members, r]
        distance_refs[r] = np.percentile(dists, [10, 25, 50, 75, 90]).astype(np.float32)

    # Soft-assignment temperature check: typical doc should hold ~2–3 regions
    temp = SOFT_TEMPERATURE
    sample_n = min(2000, len(emb))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(emb), sample_n, replace=False)
    sample_sims = sims[sample_idx]
    k = min(SOFT_K, C.shape[0])
    part = np.argpartition(-sample_sims, kth=k - 1, axis=1)[:, :k]
    top = np.take_along_axis(sample_sims, part, axis=1)
    logits = top / temp
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True) + 1e-9
    meaningful = (w > 0.15).sum(axis=1).mean()
    print(f"  soft-assign mean regions with weight>0.15 at T={temp}: {meaningful:.2f}")
    print(f"  off-atlas threshold (2nd pct cosine): {off_threshold:.3f}")

    # -- 2D projection -----------------------------------------------------
    from sklearn.decomposition import PCA

    pca2 = PCA(n_components=2, random_state=SEED).fit(emb)
    coords = pca2.transform(C).astype(np.float32)

    # -- Topic purity (diagnostic only; clustering is unsupervised) --------
    tag_ids = {t: i for i, t in enumerate(sorted(set(tags)))}
    y_tag = np.array([tag_ids[t] for t in tags], dtype=np.int32)
    purities = []
    for r in range(C.shape[0]):
        members = y_tag[assign == r]
        if len(members) < 5:
            continue
        purities.append(Counter(members.tolist()).most_common(1)[0][1] / len(members))
    purity = float(np.mean(purities)) if purities else 0.0
    print(f"  mean region purity by provenance tag: {purity:.3f}")

    phash = pipeline_hash({"encoder_weight_hash": weight_hash})
    timings["total_s"] = time.time() - wall0
    timings["cluster_total_s"] = timings["cluster_l1_s"] + timings["cluster_l2_s"]
    timings["embed_plus_train_s"] = (
        timings["embed_s"] + timings["normalize_s"] + timings["cluster_total_s"]
    )

    meta = {
        "version": "atlas-lite-v1",
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_hash": phash,
        "embed_model": DEFAULT_MODEL,
        "embed_dim": EMBED_DIM,
        "encoder_weight_hash": weight_hash,
        "pooling": "sif",
        "sif_a": SIF_A,
        "n_regions": int(C.shape[0]),
        "n_l1": int(args.l1),
        "n_l2_per_l1_target": int(args.l2_per_l1),
        "n_reference_records": len(texts),
        "seed": SEED,
        "off_atlas_threshold": off_threshold,
        "soft_k": SOFT_K,
        "soft_temperature": temp,
        "soft_mean_regions_gt_0.15": float(meaningful),
        "region_purity_by_provenance": purity,
        "region_terms": region_terms,
        "l1_labels": l1_terms,
        "format_breakdown": dict(fmt_break),
        "manifest": manifest,
        "timings_s": timings,
        "dedup": {
            "minhash_jaccard": MINHASH_JACCARD,
            "semantic_cosine": SEMANTIC_COSINE,
        },
        "normalization": {"pca_k": 2, "mean_removal": True, "l2": True},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        centroids=C,
        region_category=region_category_arr,
        region_size=region_size,
        l1_centroids=l1_centroids,
        l1_size=l1_size,
        coords=coords,
        norm_mean=norm.mean,
        norm_pca=norm.pca_components,
        idf_token_ids=idf_ids,
        idf_log_probs=idf_lps,
        distance_refs=distance_refs,
        # Empty probe arrays kept so older loaders that require the keys still work.
        probe_coef=np.zeros((0, EMBED_DIM), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        meta=np.array([json.dumps(meta)], dtype=object),
        allow_pickle=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nWrote {args.out} ({size_mb:.2f} MB)")
    if size_mb > 5.0:
        print(f"WARNING: artifact exceeds 5 MB budget ({size_mb:.2f} MB)")

    args.timing_log.parent.mkdir(parents=True, exist_ok=True)
    args.timing_log.write_text(json.dumps({
        "timings_s": timings,
        "n_records": len(texts),
        "n_l1": args.l1,
        "n_l2": int(C.shape[0]),
        "artifact_mb": size_mb,
        "pipeline_hash": phash,
        "encoder_weight_hash": weight_hash,
    }, indent=2) + "\n")
    print(f"Timing log: {args.timing_log}")
    print(
        f"\n=== TIMING SUMMARY ===\n"
        f"  collect:          {timings['collect_s']:.1f}s\n"
        f"  tokenize once:    {timings['tokenize_s']:.1f}s\n"
        f"  idf:              {timings['idf_s']:.1f}s\n"
        f"  embed (SIF):      {timings['embed_s']:.1f}s\n"
        f"  normalize fit:    {timings['normalize_s']:.1f}s\n"
        f"  cluster L1+L2:    {timings['cluster_total_s']:.1f}s\n"
        f"  embed+train:      {timings['embed_plus_train_s']:.1f}s\n"
        f"  total wall:       {timings['total_s']:.1f}s\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
