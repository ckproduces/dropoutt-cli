#!/usr/bin/env python3
"""Build atlas-lite: a frozen coordinate system for comparing datasets.

The atlas is **not** a collection of good datasets. It is a shared coordinate
system, like latitude and longitude, and it contains no notion of quality. Its
only job is to give every dataset the same bins so two fingerprints can be
compared.

Design decisions worth knowing before changing anything here.

**Level 0 is a supervised taxonomy, not clustering.** Roughly 30 hand-designed
topic categories, assigned by a classifier trained on embeddings with labels
bootstrapped from dataset provenance. Clustering would never produce a Turkish
administrative-legal category, because it is small globally and large in the
market this was built for. The taxonomy is also stable across rebuilds, so the
coarse level survives a new atlas version.

**Fine regions are fitted within each category, not globally.** Multilingual
embeddings separate by language before they separate by topic. Fitting k-means
over the whole corpus would spend most of the region budget distinguishing
Turkish from Arabic from Chinese. Conditioning on topic first leaves language
much less room to dominate. We do *not* subtract a language subspace: that
approach conditions the geometry on a language label that is least reliable
exactly where this market needs it most.

**The reference corpus is stratified, not natural.** If it mirrored the real
distribution of the web it would be about 90% English and the Turkish regions
would be useless. Low-resource content is deliberately over-sampled.

Usage:
    python tools/build_atlas.py --out src/dropoutt/data/atlas/atlas-lite-v0.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dropoutt.registry_data import taxonomy  # noqa: E402

EMBED_MODEL = "minishlab/potion-multilingual-128M"
EMBED_DIM = 256
N_REGIONS = 256
SEED = 20260728

# The reference corpus. Every entry is (hf_id, config, split, text_fields,
# taxonomy_key, target_samples). Turkish and regional sources are weighted far
# above their natural share on purpose.
SOURCES: list[tuple[str, str | None, str, tuple[str, ...], str, int]] = [
    # -- general conversation ------------------------------------------
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", ("prompt",), "general_chat", 12000),
    ("tatsu-lab/alpaca", None, "train", ("instruction", "output"), "general_chat", 8000),
    ("databricks/databricks-dolly-15k", None, "train", ("instruction", "response"), "general_chat", 6000),
    # -- code ------------------------------------------------------------
    ("google-research-datasets/mbpp", "full", "train", ("text", "code"), "code_generation", 400),
    ("openai/openai_humaneval", None, "test", ("prompt",), "code_generation", 164),
    ("iamtarun/python_code_instructions_18k_alpaca", None, "train", ("instruction", "output"), "code_generation", 6000),
    ("codeparrot/codeparrot-clean-valid", None, "train", ("content",), "code_explanation", 3000),
    ("b-mc2/sql-create-context", None, "train", ("question", "answer"), "sql_data", 4000),
    # -- mathematics -------------------------------------------------------
    ("openai/gsm8k", "main", "train", ("question", "answer"), "math_arithmetic", 5000),
    ("EleutherAI/hendrycks_math", "algebra", "train", ("problem", "solution"), "math_advanced", 1700),
    ("EleutherAI/hendrycks_math", "geometry", "train", ("problem", "solution"), "math_advanced", 800),
    # -- reasoning / QA ----------------------------------------------------
    ("allenai/ai2_arc", "ARC-Challenge", "train", ("question",), "logic_reasoning", 1100),
    ("tau/commonsense_qa", None, "train", ("question",), "logic_reasoning", 4000),
    ("rajpurkar/squad", None, "train", ("context", "question"), "reading_comprehension", 6000),
    ("truthfulqa/truthful_qa", "generation", "validation", ("question", "best_answer"), "qa_factual", 800),
    # -- science / medicine / law / finance --------------------------------
    ("cais/mmlu", "all", "dev", ("question",), "science_physics", 1500),
    ("qiaojin/PubMedQA", "pqa_labeled", "train", ("question", "long_answer"), "medicine_clinical", 1000),
    ("albertvillanova/legal_contracts", None, "train", ("text",), "law_contracts", 1500),
    ("gbharti/finance-alpaca", None, "train", ("instruction", "output"), "finance_economics", 4000),
    # -- web text ----------------------------------------------------------
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", ("text",), "news_current_events", 8000),
    ("wikimedia/wikipedia", "20231101.en", "train", ("text",), "history_culture", 6000),
    # -- Turkish and regional, deliberately over-weighted ------------------
    ("wikimedia/wikipedia", "20231101.tr", "train", ("text",), "turkish_culture", 14000),
    ("turkish-nlp-suite/InstrucTurca", None, "train", ("Input", "Output"), "general_chat", 16000),
    ("merve/turkish_instructions", None, "train", ("talimat", "çıktı"), "general_chat", 8000),
    ("umarigan/GPTeacher-General-Instruct-tr", None, "train", ("instruction", "response"), "general_chat", 5000),
    ("TFLai/Turkish-Alpaca", None, "train", ("instruction", "output"), "general_chat", 8000),
    ("AYueksel/TurkishMMLU", "All", "test", ("question",), "education_pedagogy", 3000),
    ("mcemilg/news-cat", None, "train", ("text",), "news_current_events", 4000),
    ("ardauzunoglu/tr-wikihow-summ", None, "train", ("text", "summary"), "summarization", 6000),
    ("nli_tr", "snli_tr", "train", ("premise", "hypothesis"), "logic_reasoning", 5000),
    ("wmt16", "tr-en", "train", ("translation",), "translation", 6000),
    ("wikimedia/wikipedia", "20231101.ar", "train", ("text",), "religion_philosophy", 4000),
    ("wikimedia/wikipedia", "20231101.az", "train", ("text",), "turkish_culture", 3000),
    # -- categories that would otherwise be empty --------------------------
    ("Open-Orca/OpenOrca", None, "train", ("question", "response"), "creative_writing", 4000),
    ("cnn_dailymail", "3.0.0", "train", ("article", "highlights"), "summarization", 3000),
    ("Anthropic/hh-rlhf", None, "train", ("chosen",), "safety_refusal", 3000),
    ("glaiveai/glaive-function-calling-v2", None, "train", ("system", "chat"), "tool_use_agentic", 3000),
    ("wikitablequestions", None, "train", ("question", "table"), "structured_data", 2000),
    ("bigbio/med_qa", "med_qa_en_source", "train", ("question", "answer"), "medicine_clinical", 2000),
    ("camel-ai/physics", None, "train", ("message_1", "message_2"), "science_physics", 2500),
    ("camel-ai/biology", None, "train", ("message_1", "message_2"), "science_biology", 2500),
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", ("text",), "education_pedagogy", 3000),
]


def load_embedder():
    """Load potion via a selective download.

    model2vec's ``from_pretrained`` calls ``snapshot_download`` with no
    ``allow_patterns``, so it also pulls a duplicate ONNX copy of the weights
    and roughly doubles the download. Fetching the three files it actually needs
    halves it.
    """
    from huggingface_hub import hf_hub_download
    from model2vec import StaticModel

    local = Path.home() / ".cache" / "dropoutt" / "embedder" / EMBED_MODEL.split("/")[-1]
    local.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        if not (local / name).exists():
            src = hf_hub_download(EMBED_MODEL, name)
            (local / name).write_bytes(Path(src).read_bytes())
    return StaticModel.from_pretrained(str(local))


def extract_text(row: dict, fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for f in fields:
        v = row.get(f)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.extend(str(x) for x in v.values())
        elif isinstance(v, list):
            parts.extend(str(x) for x in v[:4])
    return "\n".join(p for p in parts if p)[:2000]


def collect(sources, verbose: bool = True, per_source_seconds: float = 90.0):
    """Stream a sample from each source.

    Two guards, both learned the hard way. A per-source wall-clock budget, because
    one slow shard on a large corpus will otherwise stall the whole build. And a
    hard skip on failure, because a reference corpus assembled from thirty public
    datasets will always have some that have moved, gone gated, or changed their
    split names.
    """
    from datasets import load_dataset

    texts: list[str] = []
    labels: list[str] = []
    manifest: list[dict] = []

    for hf_id, config, split, fields, tax_key, target in sources:
        t0 = time.time()
        got = 0
        note = None
        try:
            ds = load_dataset(hf_id, config, split=split, streaming=True)
            for row in ds:
                text = extract_text(row, fields)
                if len(text) >= 60:
                    texts.append(text)
                    labels.append(tax_key)
                    got += 1
                    if got >= target:
                        break
                if time.time() - t0 > per_source_seconds:
                    note = "time budget reached"
                    break
        except Exception as exc:
            if verbose:
                print(f"  SKIP {hf_id[:40]:<40} {type(exc).__name__}: {str(exc)[:70]}", flush=True)
            manifest.append({"hf_id": hf_id, "config": config, "split": split,
                             "taxonomy": tax_key, "collected": 0, "error": str(exc)[:200]})
            continue
        if verbose:
            suffix = f"  ({note})" if note else ""
            print(f"  {hf_id[:40]:<40} {tax_key:<22} {got:>6,}  "
                  f"{time.time() - t0:>5.1f}s{suffix}", flush=True)
        manifest.append({"hf_id": hf_id, "config": config, "split": split,
                         "taxonomy": tax_key, "collected": got, "note": note})
    return texts, labels, manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("src/dropoutt/data/atlas/atlas-lite-v0.npz"))
    ap.add_argument("--regions", type=int, default=N_REGIONS)
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply every target sample size.")
    ap.add_argument("--budget", type=float, default=90.0, help="Seconds per source.")
    args = ap.parse_args()

    tax = taxonomy()["categories"]
    key_to_id = {c["key"]: c["id"] for c in tax}

    sources = [(h, c, s, f, k, max(50, int(n * args.scale))) for h, c, s, f, k, n in SOURCES]
    print(f"Collecting reference corpus from {len(sources)} sources...")
    texts, labels, manifest = collect(sources, per_source_seconds=args.budget)
    print(f"\nCollected {len(texts):,} records across "
          f"{len(set(labels))} taxonomy categories")
    if len(texts) < 1000:
        print("Too few records collected to build a usable atlas.")
        return 1

    print(f"\nEmbedding with {EMBED_MODEL} ...")
    model = load_embedder()
    t0 = time.time()
    emb = model.encode(texts, batch_size=1024, max_length=512, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    print(f"  {emb.shape} in {time.time() - t0:.1f}s "
          f"({len(texts) / max(time.time() - t0, 0.001):,.0f} records/s)")

    y = np.array([key_to_id[k] for k in labels], dtype=np.int32)

    # A source that yielded almost nothing leaves a category with a handful of
    # members. Stratified splitting fails on those, and a region fitted to five
    # records is noise anyway, so drop them and say which.
    counts_all = Counter(y.tolist())
    MIN_PER_CATEGORY = 60
    thin = {c for c, n in counts_all.items() if n < MIN_PER_CATEGORY}
    if thin:
        keep = np.array([c not in thin for c in y])
        dropped = {next(k for k, v in key_to_id.items() if v == c): counts_all[c] for c in thin}
        print(f"  dropping {len(thin)} under-populated categories: {dropped}")
        emb, y = emb[keep], y[keep]
        texts = [t for t, k in zip(texts, keep) if k]

    # -- Level 0: supervised taxonomy probe -----------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    print("\nFitting level-0 taxonomy probe ...")
    Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.15, random_state=SEED, stratify=y)
    probe = LogisticRegression(max_iter=1500, C=4.0, n_jobs=-1)
    probe.fit(Xtr, ytr)
    l0_acc = float(probe.score(Xte, yte))
    print(f"  held-out accuracy {l0_acc:.3f} over {len(set(y.tolist()))} categories")

    # -- Level 1: k-means WITHIN each category ---------------------------
    from sklearn.cluster import KMeans

    print(f"\nFitting {args.regions} regions within categories ...")
    present = sorted(set(y.tolist()))
    counts = Counter(y.tolist())
    total = sum(counts.values())
    centroids: list[np.ndarray] = []
    region_category: list[int] = []
    region_terms: list[str] = []

    for cat_id in present:
        share = counts[cat_id] / total
        k = max(1, min(int(round(args.regions * share)), counts[cat_id] // 20 or 1))
        idx = np.where(y == cat_id)[0]
        sub = emb[idx]
        if len(sub) < k * 5:
            k = max(1, len(sub) // 5)
        km = KMeans(n_clusters=k, random_state=SEED, n_init=4).fit(sub)
        for c in range(k):
            centroids.append(km.cluster_centers_[c])
            region_category.append(cat_id)
            members = idx[km.labels_ == c]
            region_terms.append(top_terms([texts[i] for i in members[:150]]))

    C = np.vstack(centroids).astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    print(f"  {C.shape[0]} regions across {len(present)} categories")

    # -- off-atlas threshold ---------------------------------------------
    sims = emb @ C.T
    best = sims.max(axis=1)
    off_threshold = float(np.percentile(best, 2.0))
    print(f"  off-atlas threshold (2nd percentile cosine): {off_threshold:.3f}")

    # -- 2D projection for display ---------------------------------------
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=SEED).fit(emb)
    coords = pca.transform(C).astype(np.float32)

    # -- purity diagnostics ----------------------------------------------
    assign = sims.argmax(axis=1)
    purity_topic = region_purity(assign, y)
    print(f"  mean region purity by taxonomy label: {purity_topic:.3f}")

    # -- the reference distribution ---------------------------------------
    # How the reference corpus itself spread across the regions it produced.
    # v0 computed `assign` here and then discarded it, which cost the atlas the
    # ability to answer the question users actually ask: not "which regions am I
    # in" but "am I over- or under-represented in them relative to a general
    # corpus". Without this, a coverage gap can only be reported as absolute
    # absence. It is a few hundred integers.
    #
    # Read it as a property of *this reference corpus*, which is stratified and
    # deliberately Turkish-weighted, not as a natural population. It is a
    # baseline to compare against, not a target to match.
    region_size = np.bincount(assign, minlength=C.shape[0]).astype(np.int32)
    empty = int((region_size == 0).sum())
    print(f"  reference mass per region: median {int(np.median(region_size))}, "
          f"max {int(region_size.max())}, {empty} region(s) drew nothing")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        centroids=C,
        region_category=np.array(region_category, dtype=np.int32),
        region_size=region_size,
        coords=coords,
        probe_coef=probe.coef_.astype(np.float32),
        probe_intercept=probe.intercept_.astype(np.float32),
        probe_classes=probe.classes_.astype(np.int32),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        meta=np.array([json.dumps({
            "version": "atlas-lite-v0",
            "embed_model": EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "n_regions": int(C.shape[0]),
            "n_reference_records": len(texts),
            "seed": SEED,
            "off_atlas_threshold": off_threshold,
            "l0_holdout_accuracy": l0_acc,
            "region_purity_by_taxonomy": purity_topic,
            "region_terms": region_terms,
            "taxonomy_present": present,
            "manifest": manifest,
        })], dtype=object),
        allow_pickle=True,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")
    return 0


def top_terms(texts: list[str], k: int = 5) -> str:
    """Name a region from its most distinctive words. Deterministic and offline."""
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "you",
        "not", "but", "have", "has", "can", "will", "your", "what", "which",
        "bir", "ve", "bu", "ile", "için", "olarak", "daha", "gibi", "olan", "her",
        "de", "da", "ki", "mi", "en", "çok", "veya", "ise", "ancak",
    }
    counter: Counter[str] = Counter()
    for t in texts:
        for w in t.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop:
                counter[w] += 1
    return ", ".join(w for w, _ in counter.most_common(k))


def region_purity(assign: np.ndarray, labels: np.ndarray) -> float:
    """Mean share of each region occupied by its most common label."""
    purities = []
    for r in np.unique(assign):
        members = labels[assign == r]
        if len(members) < 5:
            continue
        purities.append(Counter(members.tolist()).most_common(1)[0][1] / len(members))
    return float(np.mean(purities)) if purities else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
