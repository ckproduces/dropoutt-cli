#!/usr/bin/env python3
"""Build atlas-lite-v2: a frozen topic coordinate system.

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
    python tools/build_atlas.py --out src/dropoutt/data/atlas/atlas-lite-v2.npz
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

from dropoutt.atlas.apply import Atlas
from dropoutt.atlas.embed import DEFAULT_MODEL, embedding_weight_hash
from dropoutt.atlas.embed import load as load_embedder
from dropoutt.atlas.extract import extract_from_fields, extract_text
from dropoutt.atlas.normalize import EMBED_DIM, SIF_A, fit_norm
from dropoutt.atlas.pipeline import (
    PIPELINE_VERSION,
    pipeline_hash,
    population_crosswalk,
)

SEED = 20260801
N_L1 = 50
N_L2_PER_L1 = 20
DEFAULT_SCALE = 4.0
SOFT_K = 5
SOFT_TEMPERATURE = 0.08
MINHASH_JACCARD = 0.8
SEMANTIC_COSINE = 0.95
MIN_CHARS = 80
IDF_TOP_TOKENS = 120_000
MIN_L2_FIT_MEMBERS = 300
MIN_CALIBRATION_MEMBERS = 200
LABEL_TERMS = 12
DISTANCE_PERCENTILES = np.array(
    [1, 2, 5, 10, 15, 20, 25, 35, 50, 65, 75, 80, 85, 90, 95, 98, 99],
    dtype=np.float32,
)
PROTOTYPE_PERCENTILES = np.array([2, 10, 25, 40, 60, 75, 90, 98], dtype=np.float32)
COOCCURRENCE_NEIGHBORS = 16
MIN_ARTIFACT_MB = 3.0
MAX_ARTIFACT_MB = 5.0

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
    # -- code (Stack/StarCoder; configs prevent Python/JS dominance) --------
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train", ("instruction", "output"), "code", 5000, "en"),
    ("google-research-datasets/mbpp", "full", "train", ("text", "code"), "code", 400, "en"),
    ("openai/openai_humaneval", None, "test", ("prompt",), "code", 164, "en"),
    ("codeparrot/codeparrot-clean-valid", None, "train", ("content",), "code", 3000, "code"),
    # Public 0.3% StarCoderData sample. The canonical repository is gated; this
    # mirror preserves its rows and records the parent in SOURCE_PROVENANCE.
    ("codecomplete/starcoderdata_0.003", None, "train", ("text",), "code", 40000, "code:multi"),
    # The Stack's public small corpus must name a config. Leaving it as None
    # invokes the retired dataset script and was why the load-bearing source
    # disappeared from v1.
    ("bigcode/the-stack-smol-xs", "python", "train", ("content",), "code", 1500, "code:python"),
    ("bigcode/the-stack-smol-xs", "javascript", "train", ("content",), "code", 1500, "code:javascript"),
    ("bigcode/the-stack-smol-xs", "java", "train", ("content",), "code", 1500, "code:java"),
    ("bigcode/the-stack-smol-xs", "c", "train", ("content",), "code", 1500, "code:c"),
    ("bigcode/the-stack-smol-xs", "c++", "train", ("content",), "code", 1500, "code:cpp"),
    ("bigcode/the-stack-smol-xs", "go", "train", ("content",), "code", 1500, "code:go"),
    ("bigcode/the-stack-smol-xs", "rust", "train", ("content",), "code", 1500, "code:rust"),
    ("bigcode/the-stack-smol-xs", "typescript", "train", ("content",), "code", 1500, "code:typescript"),
    ("bigcode/the-stack-smol-xs", "c-sharp", "train", ("content",), "code", 1500, "code:csharp"),
    ("bigcode/the-stack-smol-xs", "shell", "train", ("content",), "code", 1500, "code:shell"),
    # -- legal / finance / admin (Common Corpus-shaped region) --------------
    ("gbharti/finance-alpaca", None, "train", ("instruction", "output"), "legal_finance", 5000, "en"),
    ("albertvillanova/legal_contracts", None, "train", ("text",), "legal_finance", 4000, "en"),
    ("coastalcph/lex_glue", "ecthr_a", "train", ("text",), "legal_finance", 4000, "en"),
    # -- scientific (arXiv, PubMed; peS2o script-loader is retired on Hub) --
    ("CShorten/ML-ArXiv-Papers", None, "train", ("title", "abstract"), "scientific", 8000, "en"),
    ("qiaojin/PubMedQA", "pqa_labeled", "train", ("question", "long_answer"), "scientific", 1000, "en"),
    ("BEE-spoke-data/peS2o-100k_en-xlong", None, "train", ("text",), "scientific", 20000, "en"),
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

SOURCE_PROVENANCE = {
    "codecomplete/starcoderdata_0.003": "bigcode/starcoderdata",
}


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
            ) from None
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item
    raise TimeoutError(f"source budget {per_source_seconds:.0f}s exhausted")


def _load_source_dataset(
    load_dataset,
    hf_id: str,
    config: str | None,
    split: str,
):
    """Load a source without invoking retired Hub dataset scripts."""

    if hf_id == "HuggingFaceFW/fineweb" and config == "sample-10BT":
        return _dataset_viewer_rows(hf_id, config, split)

    direct_parquet = {
        ("albertvillanova/legal_contracts", None): (
            "default/partial-train/0000.parquet"
        ),
    }
    direct_path = direct_parquet.get((hf_id, config))
    if direct_path:
        url = (
            f"https://huggingface.co/datasets/{hf_id}/"
            f"resolve/refs%2Fconvert%2Fparquet/{direct_path}"
        )
        return load_dataset(
            "parquet",
            data_files={split: [url]},
            split=split,
            streaming=True,
        )

    if hf_id == "bigcode/the-stack-smol-xs":
        if not config:
            raise ValueError("The Stack source requires an explicit language config")
        from urllib.parse import quote

        subset = quote(config, safe="+")
        url = (
            "https://huggingface.co/datasets/bigcode/the-stack-smol-xs/"
            f"resolve/refs%2Fconvert%2Fparquet/{subset}/{split}/0000.parquet"
        )
        return load_dataset(
            "parquet",
            data_files={split: [url]},
            split=split,
            streaming=True,
        )
    return load_dataset(hf_id, config, split=split, streaming=True)


def _dataset_viewer_rows(
    hf_id: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
):
    """Stream bounded pages from the Hub viewer when 2GB parquet stalls."""

    from urllib.parse import urlencode
    from urllib.request import urlopen

    offset = 0
    while True:
        query = urlencode({
            "dataset": hf_id,
            "config": config,
            "split": split,
            "offset": offset,
            "length": page_size,
        })
        with urlopen(
            f"https://datasets-server.huggingface.co/rows?{query}",
            timeout=30,
        ) as response:
            payload = json.loads(response.read())
        rows = payload.get("rows", [])
        if not rows:
            return
        for item in rows:
            row = item.get("row")
            if isinstance(row, dict):
                yield row
        offset += len(rows)


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
                lambda repo=hf_id, subset=config, part=split: _load_source_dataset(
                    load_dataset, repo, subset, part
                ),
                timeout=min(45.0, per_source_seconds),
                label=f"load_dataset({hf_id})",
            )
            deadline = t0 + per_source_seconds
            for row in _iter_with_timeout(ds, per_source_seconds):
                if time.time() >= deadline:
                    note = "time budget reached"
                    break
                # Fast path for ordinary prose fields. Code is always routed
                # through the shared code extractor; treating a raw source file
                # as plain text makes clusters recover syntax/register instead
                # of the topic discussed by comments and docstrings.
                text, fmt = "", "plain"
                forced_code = (
                    tag == "code"
                    and len(fields) == 1
                    and isinstance(row.get(fields[0]), str)
                )
                if (
                    tag != "code"
                    and len(fields) == 1
                    and isinstance(row.get(fields[0]), str)
                ):
                    candidate = row[fields[0]].strip()
                    if len(candidate) >= MIN_CHARS:
                        text, fmt = candidate, "plain"
                elif forced_code:
                    text, fmt = extract_text(
                        row[fields[0]], detected_format="code"
                    )
                if not text and not forced_code:
                    text, fmt = extract_from_fields(row, fields)
                if not text and not forced_code:
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
                    f"{hf_id}:{config}:{got}:{piece[:200]}".encode(),
                    digest_size=8,
                ).hexdigest()
                texts.append(piece)
                tags.append(tag)
                formats.append(fmt)
                source_ids.append(f"{hf_id}:{config}" if config else hf_id)
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
            entry = {
                "hf_id": hf_id, "config": config, "split": split,
                "topic": tag, "lang": lang, "collected": 0, "error": error,
            }
            if hf_id in SOURCE_PROVENANCE:
                entry["canonical_source"] = SOURCE_PROVENANCE[hf_id]
            manifest.append(entry)
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
        if hf_id in SOURCE_PROVENANCE:
            entry["canonical_source"] = SOURCE_PROVENANCE[hf_id]
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
        for shingle in zip(words, words[1:], words[2:], strict=False):
            mh.update(" ".join(shingle).encode())
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
    """Greedy cosine dedup in small sign buckets.

    The original eight-bit buckets left ~2,300 rows per bucket at 600K
    records, then compared every pair in Python. Twelve bits keep candidate
    groups small and threshold each cosine block in NumPy.
    """
    n = len(emb)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    n_bits = min(12, emb.shape[1])
    bits = (emb[:, :n_bits] > 0).astype(np.uint16)
    keys = bits.dot((1 << np.arange(n_bits)).astype(np.uint16))
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, k in enumerate(keys.tolist()):
        buckets[int(k)].append(i)
    for idxs in buckets.values():
        if len(idxs) < 2:
            continue
        sub = emb[idxs]
        sims = sub @ sub.T
        rows, cols = np.where(np.triu(sims >= threshold, k=1))
        if not rows.size:
            continue
        for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
            if keep[idxs[a]]:
                keep[idxs[b]] = False
    return keep


def top_terms(
    texts: list[str],
    k: int = LABEL_TERMS,
    df: Counter[str] | None = None,
) -> str:
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


def cluster_purity(assignments: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Macro/micro purity, used only as a build diagnostic."""

    y = np.asarray(labels, dtype=object)
    majority = 0
    macro: list[float] = []
    for cell in np.unique(assignments):
        members = y[assignments == cell]
        if not len(members):
            continue
        largest = Counter(members.tolist()).most_common(1)[0][1]
        majority += largest
        macro.append(largest / len(members))
    return {
        "macro": float(np.mean(macro)) if macro else 0.0,
        "micro": majority / max(len(y), 1),
    }


def conditional_source_ami(
    assignments: np.ndarray,
    topics: list[str],
    languages: list[str],
    sources: list[str],
) -> float:
    """Source leakage left after holding topic and language constant."""

    from sklearn.metrics import adjusted_mutual_info_score

    topic_arr = np.asarray(topics, dtype=object)
    language_arr = np.asarray(languages, dtype=object)
    source_arr = np.asarray(sources, dtype=object)
    weighted = 0.0
    eligible = 0
    for topic, language in sorted(set(zip(topics, languages, strict=True))):
        mask = (topic_arr == topic) & (language_arr == language)
        n = int(mask.sum())
        if n < 100 or len(set(source_arr[mask].tolist())) < 2:
            continue
        weighted += n * float(
            adjusted_mutual_info_score(source_arr[mask], assignments[mask])
        )
        eligible += n
    return weighted / max(eligible, 1)


def top_distribution(
    values: list[str],
    members: np.ndarray,
    n: int = 3,
) -> list[dict]:
    counts = Counter(values[i] for i in members)
    return [
        {"value": value, "records": count, "share": round(count / len(members), 4)}
        for value, count in counts.most_common(n)
    ]


def categorical_cell_counts(
    assignments: np.ndarray,
    values: list[str],
    n_cells: int,
) -> tuple[list[str], np.ndarray]:
    labels = sorted(set(values))
    label_to_id = {label: i for i, label in enumerate(labels)}
    codes = np.fromiter(
        (label_to_id[value] for value in values),
        dtype=np.int32,
        count=len(values),
    )
    counts = np.zeros((n_cells, len(labels)), dtype=np.int32)
    np.add.at(counts, (assignments, codes), 1)
    return labels, counts


def cooccurrence_neighbors(
    cell_source_counts: np.ndarray,
    *,
    top_k: int = COOCCURRENCE_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Top cells that co-occur across known source datasets."""

    occupancy = (cell_source_counts > 0).astype(np.int32)
    cooccurrence = occupancy @ occupancy.T
    frequency = np.diag(cooccurrence).copy()
    union = (
        frequency[:, None]
        + frequency[None, :]
        - cooccurrence
    )
    scores = np.divide(
        cooccurrence,
        union,
        out=np.zeros_like(cooccurrence, dtype=np.float32),
        where=union > 0,
    )
    np.fill_diagonal(scores, -1.0)
    k = min(top_k, max(1, scores.shape[1] - 1))
    neighbors = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    neighbor_scores = np.take_along_axis(scores, neighbors, axis=1)
    order = np.argsort(-neighbor_scores, axis=1)
    return (
        np.take_along_axis(neighbors, order, axis=1).astype(np.int32),
        np.take_along_axis(neighbor_scores, order, axis=1).astype(np.float32),
    )


def cell_prototypes(
    embeddings: np.ndarray,
    assignments: np.ndarray,
    distances: np.ndarray,
    record_ids: list[str],
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial prototypes preserve within-cell breadth, not only the center."""

    n_prototypes = len(PROTOTYPE_PERCENTILES)
    vectors = np.zeros(
        (n_cells, n_prototypes, embeddings.shape[1]),
        dtype=np.float16,
    )
    ids = np.zeros((n_cells, n_prototypes), dtype="S16")
    prototype_distances = np.zeros((n_cells, n_prototypes), dtype=np.float32)
    for cell in range(n_cells):
        members = np.where(assignments == cell)[0]
        if not len(members):
            continue
        ordered = members[np.argsort(distances[members])]
        ranks = np.rint(
            PROTOTYPE_PERCENTILES / 100.0 * (len(ordered) - 1)
        ).astype(np.int64)
        selected = ordered[ranks]
        vectors[cell] = embeddings[selected].astype(np.float16)
        ids[cell] = np.asarray(
            [record_ids[i].encode() for i in selected],
            dtype="S16",
        )
        prototype_distances[cell] = distances[selected]
    return vectors, ids, prototype_distances


def region_diagnostics(
    texts: list[str],
    embeddings: np.ndarray,
    l1_assign: np.ndarray,
    l1_centroids: np.ndarray,
    labels: list[str],
    tags: list[str],
    sources: list[str],
    formats: list[str],
    languages: list[str],
) -> list[dict]:
    """Inspectable L1 summaries for topic-vs-register review."""

    out: list[dict] = []
    for cell in range(len(l1_centroids)):
        idx = np.where(l1_assign == cell)[0]
        if not len(idx):
            continue
        nearest = idx[np.argsort(-(embeddings[idx] @ l1_centroids[cell]))[:3]]

        source_counts = Counter(sources[i] for i in idx)
        out.append({
            "l1_id": cell,
            "label": labels[cell],
            "records": len(idx),
            "n_sources": len(source_counts),
            "source_concentration": round(
                source_counts.most_common(1)[0][1] / len(idx), 4
            ),
            "top_topics": top_distribution(tags, idx),
            "top_sources": top_distribution(sources, idx),
            "top_formats": top_distribution(formats, idx),
            "top_languages": top_distribution(languages, idx),
            "exemplars": [
                re.sub(r"\s+", " ", texts[i])[:240]
                for i in nearest
            ],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("src/dropoutt/data/atlas/atlas-lite-v2.npz"))
    ap.add_argument("--l1", type=int, default=N_L1)
    ap.add_argument("--l2-per-l1", type=int, default=N_L2_PER_L1)
    ap.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Multiply every target sample size (default: 4.0, ~600K+ retained records).",
    )
    ap.add_argument("--budget", type=float, default=180.0, help="Seconds per source.")
    ap.add_argument("--timing-log", type=Path, default=Path("src/dropoutt/data/atlas/build-timing.json"))
    ap.add_argument(
        "--diagnostics-log",
        type=Path,
        default=Path("src/dropoutt/data/atlas/build-diagnostics.json"),
    )
    ap.add_argument(
        "--previous",
        type=Path,
        default=Path("src/dropoutt/data/atlas/atlas-lite-v1.npz"),
        help="Previous Atlas used to derive a population crosswalk.",
    )
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
    semantic_keep = keep.copy()
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
        # Do not mint fine cells whose calibration will be based on a few
        # dozen examples. The larger build normally supports all 12 children;
        # sparse L1 regions receive fewer, better-supported children.
        k = min(
            args.l2_per_l1,
            max(1, len(idx) // MIN_L2_FIT_MEMBERS),
        )
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
    assign = np.empty(len(emb), dtype=np.int32)
    best = np.empty(len(emb), dtype=np.float32)
    for start in range(0, len(emb), 20_000):
        stop = min(start + 20_000, len(emb))
        batch_sims = emb[start:stop] @ C.T
        batch_assign = batch_sims.argmax(axis=1)
        assign[start:stop] = batch_assign
        best[start:stop] = batch_sims[
            np.arange(stop - start), batch_assign
        ]
    off_threshold = float(np.percentile(best, 2.0))
    region_size = np.bincount(assign, minlength=C.shape[0]).astype(np.int32)
    l1_size = np.bincount(l1_labels_idx, minlength=args.l1).astype(np.int32)

    distance_refs = np.zeros(
        (C.shape[0], len(DISTANCE_PERCENTILES)),
        dtype=np.float32,
    )
    distance_refs_support = region_size.copy()
    distance_refs_reliable = region_size >= MIN_CALIBRATION_MEMBERS
    assigned_distance = 1.0 - best
    for r in range(C.shape[0]):
        members = np.where(assign == r)[0]
        if len(members) == 0:
            continue
        dists = assigned_distance[members]
        if len(members) < MIN_CALIBRATION_MEMBERS:
            parent = region_category_arr[r]
            parent_pool = assigned_distance[
                region_category_arr[assign] == parent
            ]
            needed = MIN_CALIBRATION_MEMBERS - len(members)
            if len(parent_pool):
                # Deterministic empirical-Bayes fallback: retain every local
                # observation, then borrow residual distances from siblings.
                take = np.linspace(
                    0,
                    len(parent_pool) - 1,
                    num=needed,
                    dtype=np.int64,
                )
                dists = np.concatenate([dists, np.sort(parent_pool)[take]])
        distance_refs[r] = np.percentile(
            dists, DISTANCE_PERCENTILES
        ).astype(np.float32)

    # Soft-assignment temperature check: typical doc should hold ~2–3 regions
    temp = SOFT_TEMPERATURE
    sample_n = min(2000, len(emb))
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(emb), sample_n, replace=False)
    sample_sims = emb[sample_idx] @ C.T
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
    print(
        f"  distance calibration: {int(distance_refs_reliable.sum())}/{len(C)} "
        f"cells have >= {MIN_CALIBRATION_MEMBERS} direct members; sparse cells "
        f"borrow from their L1 parent"
    )

    print("\nBuilding rich cell support tables ...")
    source_labels, cell_source_counts = categorical_cell_counts(
        assign, source_ids, len(C)
    )
    topic_labels, cell_topic_counts = categorical_cell_counts(
        assign, tags, len(C)
    )
    language_labels, cell_language_counts = categorical_cell_counts(
        assign, langs, len(C)
    )
    cooccurrence_ids, cooccurrence_scores = cooccurrence_neighbors(
        cell_source_counts
    )
    prototype_vectors, prototype_record_ids, prototype_distances = cell_prototypes(
        emb,
        assign,
        assigned_distance,
        record_ids,
        len(C),
    )

    # -- Version lineage ---------------------------------------------------
    crosswalk: dict | None = None
    if args.previous.exists():
        print(f"\nDeriving population crosswalk from {args.previous.name} ...")
        t0 = time.time()
        previous_atlas = Atlas.load(args.previous)
        previous_embedder = embedder.bind_idf(previous_atlas.token_log_prob)
        previous_raw = previous_embedder.encode_tokenized(tokenized)
        _, _, previous_nearest = previous_atlas.assign_full(previous_raw)
        previous_nearest = previous_nearest[semantic_keep]
        crosswalk = population_crosswalk(
            assign,
            previous_nearest,
            n_current=C.shape[0],
            n_previous=previous_atlas.n_regions,
            previous_version=str(previous_atlas.meta.get("version", "unknown")),
        )
        timings["lineage_s"] = time.time() - t0
        print(
            f"  {crosswalk['summary']} in {timings['lineage_s']:.1f}s"
        )
        del previous_raw

    # -- 2D projection -----------------------------------------------------
    from sklearn.decomposition import PCA

    pca2 = PCA(n_components=2, random_state=SEED).fit(emb)
    coords = pca2.transform(C).astype(np.float32)

    # -- Topic-vs-source diagnostic ----------------------------------------
    # A high source purity is failure: cells have learned which dataset a row
    # came from. Topic purity is reported separately because the broad tags are
    # intended semantics, though they remain source-level weak labels.
    from sklearn.metrics import normalized_mutual_info_score

    topic_purity = cluster_purity(assign, tags)
    source_purity = cluster_purity(assign, source_ids)
    topic_nmi = float(normalized_mutual_info_score(tags, assign))
    source_nmi = float(normalized_mutual_info_score(source_ids, assign))
    source_ami_conditioned = conditional_source_ami(
        assign, tags, langs, source_ids
    )
    source_counts_per_cell = [
        len(set(np.asarray(source_ids, dtype=object)[assign == r].tolist()))
        for r in range(C.shape[0])
    ]
    single_source_cells = sum(n <= 1 for n in source_counts_per_cell)
    diagnostic_status = "topic-dominant"
    if (
        single_source_cells
        or source_purity["micro"] >= topic_purity["micro"]
        or source_ami_conditioned > 0.25
    ):
        diagnostic_status = "source-dominant-review"
    print(
        "  L2 topic purity "
        f"{topic_purity['macro']:.3f} macro / {topic_purity['micro']:.3f} micro"
    )
    print(
        "  L2 source purity "
        f"{source_purity['macro']:.3f} macro / {source_purity['micro']:.3f} micro "
        "(lower is better)"
    )
    print(
        f"  NMI topic={topic_nmi:.3f}, source={source_nmi:.3f}; "
        f"conditional source AMI={source_ami_conditioned:.3f}; "
        f"{single_source_cells} single-source cells; {diagnostic_status}"
    )

    l1_diagnostics = region_diagnostics(
        texts,
        emb,
        l1_labels_idx,
        l1_centroids,
        l1_terms,
        tags,
        source_ids,
        formats,
        langs,
    )
    print("  L1 inspection sample:")
    for entry in l1_diagnostics[:8]:
        topics = ", ".join(
            f"{x['value']} {x['share']:.0%}" for x in entry["top_topics"][:2]
        )
        print(
            f"    {entry['l1_id']:>2} {entry['label'][:40]:<40} "
            f"sources={entry['n_sources']:<2} top={topics}"
        )

    phash = pipeline_hash({"encoder_weight_hash": weight_hash})
    timings["total_s"] = time.time() - wall0
    timings["cluster_total_s"] = timings["cluster_l1_s"] + timings["cluster_l2_s"]
    timings["embed_plus_train_s"] = (
        timings["embed_s"] + timings["normalize_s"] + timings["cluster_total_s"]
    )

    meta = {
        "version": "atlas-lite-v2",
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
        "region_topic_purity": topic_purity,
        "region_source_purity": source_purity,
        "topic_cluster_nmi": topic_nmi,
        "source_cluster_nmi": source_nmi,
        "source_cluster_ami_conditioned_on_topic_language": source_ami_conditioned,
        "cluster_diagnostic_status": diagnostic_status,
        "single_source_cells": single_source_cells,
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
        "calibration": {
            "min_direct_members": MIN_CALIBRATION_MEMBERS,
            "sparse_cell_fallback": "local_plus_l1_parent_residuals",
            "directly_reliable_cells": int(distance_refs_reliable.sum()),
            "distance_percentiles": DISTANCE_PERCENTILES.tolist(),
        },
        "prototype_percentiles": PROTOTYPE_PERCENTILES.tolist(),
        "source_labels": source_labels,
        "topic_labels": topic_labels,
        "language_labels": language_labels,
        "cooccurrence_neighbors": COOCCURRENCE_NEIGHBORS,
        "crosswalk": crosswalk,
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
        distance_refs_support=distance_refs_support,
        distance_refs_reliable=distance_refs_reliable,
        cell_source_counts=cell_source_counts,
        cell_topic_counts=cell_topic_counts,
        cell_language_counts=cell_language_counts,
        cooccurrence_ids=cooccurrence_ids,
        cooccurrence_scores=cooccurrence_scores,
        prototype_vectors=prototype_vectors,
        prototype_record_ids=prototype_record_ids,
        prototype_distances=prototype_distances,
        # Empty probe arrays kept so older loaders that require the keys still work.
        probe_coef=np.zeros((0, EMBED_DIM), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        meta=np.array([json.dumps(meta)], dtype=object),
        allow_pickle=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nWrote {args.out} ({size_mb:.2f} MB)")
    size_ok = MIN_ARTIFACT_MB <= size_mb <= MAX_ARTIFACT_MB
    if not size_ok:
        print(
            f"ERROR: useful artifact content must be {MIN_ARTIFACT_MB:.0f}–"
            f"{MAX_ARTIFACT_MB:.0f} MB; got {size_mb:.2f} MB"
        )

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

    args.diagnostics_log.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_log.write_text(json.dumps({
        "atlas_version": meta["version"],
        "pipeline_hash": phash,
        "status": diagnostic_status,
        "topic_purity": topic_purity,
        "source_purity": source_purity,
        "topic_cluster_nmi": topic_nmi,
        "source_cluster_nmi": source_nmi,
        "source_cluster_ami_conditioned_on_topic_language": source_ami_conditioned,
        "single_source_cells": single_source_cells,
        "mean_sources_per_cell": float(np.mean(source_counts_per_cell)),
        "calibration_direct_support": {
            "minimum": int(region_size.min()),
            "p10": float(np.percentile(region_size, 10)),
            "median": float(np.median(region_size)),
            "directly_reliable_cells": int(distance_refs_reliable.sum()),
            "total_cells": len(region_size),
        },
        "l1_regions": l1_diagnostics,
    }, indent=2) + "\n")
    print(f"Diagnostics log: {args.diagnostics_log}")
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
    return 0 if size_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
