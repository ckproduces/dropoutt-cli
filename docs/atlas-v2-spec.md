# Atlas v2 design specification

This document describes the planned **atlas-v2** and **atlas-v2-lite** products.
It supersedes the single-bundle model of `atlas-v1-lite` (one `.npz` with an
internal L1/L2 hierarchy and a tiered report surface).

Both products are **free**. There is **no tier toggle** in the UI — each product
reports at a single resolution (its L2 cells). L1 exists only as a clustering
scaffold at build time.

---

## Summary comparison

| | **atlas-v2** (heavy / high-res) | **atlas-v2-lite** (fast) |
| --- | --- | --- |
| **Role** | Best coverage map; worth the wait | Quick placement; good enough |
| **Encoder file** | `minishlab/potion-multilingual-128M` — one fetch | Same file; slices fewer columns |
| **Quantised storage** | **256 columns** int8 + per-row scale (~126 MB cache) | Same cache |
| **Runtime columns** | **256** | **16** |
| **Pooling** | SIF (`w = a/(a+p)`) + IDF table in artifact | Mean (no IDF at runtime) |
| **Text window** | **4000 chars**, head / mid / tail | **1024 chars**, head / mid / tail |
| **Max tokens** | **1024** | **256** |
| **Normalization** | Per-language mean + top-2 PCA + L2 | Global mean + top-1 PCA + L2 |
| **L1 scaffold** | **256** (k-means, fixed) | **16** (k-means, fixed) |
| **L2 discovery** | Leiden on kNN graph within each L1 | Same algorithm, coarser settings |
| **L2 cell count** | **Unknown until build finishes** | **Unknown until build finishes** |
| **Reference build data** | ~2.1M records post-dedup | ~130k stratified subsample |
| **Artifact (compressed)** | **25–40 MB** | **2–4 MB** |
| **Default runtime sample** | **200,000** | **50,000** |
| **Speed vs lite** | ~**5–10× slower** | Baseline |
| **`atlas_version`** | `atlas-v2` | `atlas-v2-lite` |
| **Cross-product compare** | Refused | Refused |

Sharing one encoder does **not** make the maps the same. They use different
dimensions, normalization, pooling, text windows, and cell partitions. Lite
should agree with full on **coarse subject** (~85–90%) but rarely on the **same
fine cell**.

---

## What the atlas is

A **frozen coordinate system** — like latitude and longitude. It is not a
collection of good datasets and carries no notion of quality. Its job is to give
every dataset the same bins so fingerprints from different machines compare.

Every result carries `atlas_version` + `pipeline_hash`. Fingerprints computed
against different products or pipeline versions **must not** be compared; region
ids do not mean the same thing.

---

## Two products, one encoder

### Why one model

`potion-multilingual-128M` is a **Matryoshka** static embedding: the first 16
columns are a valid 16-dimensional model; the first 256 are the full-resolution
model. Same subword vocabulary, same tokenizer, same HuggingFace download.

| Layer | Shared | Product-specific |
| --- | --- | --- |
| Encoder weights + tokenizer | yes | — |
| Columns used at runtime | — | 256 vs 16 |
| Pooling | same code path | SIF vs mean |
| Char / token limits | same tokenizer | 4000/1024 vs 1024/256 |
| Normalization constants | same code | separate fit per product |
| Centroids / cells | — | separate `.npz` |

Lite is faster because it slices **16 columns**, reads **less text**, and **mean-pools** — not because it loads a different model.

### Encoder delivery (not in the wheel)

| Component | Where | Size |
| --- | --- | --- |
| Atlas map (`.npz`) | Inside the pip wheel | 2–40 MB per product |
| Encoder + tokenizer | Fetched on first use | ~126 MB cached (256-col quant) |

**First run (online):**

1. `load_bundled()` reads the `.npz` from site-packages.
2. `load_embedder(out_dim=atlas.dim)` checks `$DROPOUTT_CACHE/embedder/potion-multilingual-128M/`.
3. If missing: download `config.json`, `model.safetensors`, `tokenizer.json` from HuggingFace.
4. Quantise to **256 columns** int8; delete the float32 safetensors.
5. Subsequent runs read `encoder-int8/codes.npy`, `scale.npy`, `tokenizer.json` only.

**Offline / clusters:**

```bash
dropoutt fetch                    # login node, has network
export DROPOUTT_CACHE=/shared/dropoutt
dropoutt atlas ./data --offline   # compute nodes
```

Reliability: deterministic quantisation (same weights → same `encoder_weight_hash`
on every machine); corrupt downloads self-heal on retry; `--offline` never
 touches the network and fails clearly if the cache is empty.

---

## Encode path

### Text fed vs max tokens

Two limits at two stages:

```
Record text
    ↓   ← char limit (before tokenizer); controls tokenization cost
Tokenizer
    ↓   subword ID list
    ↓   ← max tokens + window selection; controls pool / multiply cost
Pool (SIF or mean)
    ↓
Vector (256-d or 16-d)
    ↓   ← normalization constants from artifact
Placement vs centroids
```

### Three-window token selection

Long records are not represented by the first N tokens only. After the char cap,
tokenize once; if `len(ids) > max_tokens`, take **head / middle / tail** windows
(deterministic split, e.g. 40% / 20% / 40% of the token budget). Reproducible
across machines; no random sampling.

### IDF (atlas-v2 only)

The IDF table stores **log unigram probabilities** for SIF pooling. Tokens in
the table get `w = a / (a + p)`; tokens outside get a conservative fallback.

Ship all observed types (up to a documented cap). On the current v1 build,
120,000 types cover **98.91% of token mass**; v2 targets **≥99%**.

Lite uses **mean pooling** — the IDF table is not used at runtime (may be omitted
from the lite artifact).

---

## Map geometry

### L1 — fixed scaffold

| | atlas-v2 | atlas-v2-lite |
| --- | --- | --- |
| L1 regions | 256 | 16 |
| Method | k-means on normalized embeddings | same |

L1 is not exposed as a user-facing tier. It groups the reference corpus before
fine structure is discovered.

### L2 — Leiden community detection (no fixed cell count)

**Removed from v1:** `L2_BUDGET`, silhouette marginal-gain allocation, fixed
child counts per L1.

**New pipeline**, per L1 region:

```
assign records to L1
    → build kNN graph on embeddings within L1
    → Leiden community detection (recursive splits on large L1s if configured)
    → merge communities smaller than min_community_size
    → one centroid per community = one L2 cell
    → n_regions = total cells across all L1s (written to artifact at end)
```

The **final L2 count is unknown until training completes**. Release notes and
artifact metadata record the outcome.

| Parameter | atlas-v2 | atlas-v2-lite | Effect |
| --- | --- | --- | --- |
| kNN **k** | 30–50 | 10–20 | local neighbourhood density |
| Leiden **γ** (resolution) | 0.12 | 0.08 | more vs fewer communities |
| **Min community size** | 200 | 200 | calibration floor; smaller merges |
| Recursive depth | split communities ≥ 40k | shallow / single pass | finer splits where mass supports it |

### Build metadata (example)

```json
{
  "version": "atlas-v2",
  "n_l1": 256,
  "n_regions": 1847,
  "l2_method": "leiden_knn_recursive",
  "leiden_gamma": 0.12,
  "leiden_k": 40,
  "min_community_size": 200,
  "l1_cell_counts": { "0": 12, "1": 3 }
}
```

---

## Reference corpus

### Fetch and rebuild

| | atlas-v2 | atlas-v2-lite |
| --- | --- | --- |
| **Fetch (pre-dedup)** | ~**2.5–2.7M** rows (full source catalogue) | No separate fetch |
| **After dedup** | ~**2.1–2.3M** retained | ~**130k** stratified subsample |
| **Build input** | all retained records | subsample from same pool |

One `fetch_corpus.py` run serves both products. Lite is a **stratified slice**
(preserve axis and language proportions), not a separate scrape.

### Why these sizes

| Concern | atlas-v2 | atlas-v2-lite |
| --- | --- | --- |
| IDF saturation (≥99% token mass) | needs full ~2.1M fetch | N/A (mean pool) |
| Axis floors (`AXIS_FLOORS`) | met at full fetch | met in subsample |
| Min 200 records / community | Leiden + merge enforces | ~2k avg if ~130k / 64 cells |
| Per-language norm (≥2k / language) | met at full scale | lite uses global norm |

Below ~850k post-dedup, v2 would under-shoot IDF coverage and axis floors.
Below ~13k post-dedup, lite hits the hard calibration floor (64 × 200).

---

## Artifacts (in the wheel)

Rich metadata does not slow placement; it loads once.

| Payload | atlas-v2 | atlas-v2-lite |
| --- | --- | --- |
| Centroids | `n_regions × 256` | `n_regions × 16` |
| Exemplar texts / cell | 64 × 800 chars | 24 × 384 chars |
| Radial prototypes / cell | 32 (float16) | 12 (float16) |
| Distance calibration knots | 50 | 33 |
| Co-occurrence neighbours / cell | 48 | 24 |
| Term vector / cell | top 64 tf-idf weights | top 32 |
| Cell–cell similarity matrix | yes (float16) | no |
| Lite ↔ full crosswalk | yes | yes |

Target compressed sizes: **25–40 MB** (v2), **2–4 MB** (lite).

---

## Runtime

```bash
dropoutt atlas ./data                      # default: atlas-v2-lite
dropoutt atlas ./data --atlas atlas-v2     # heavy map
dropoutt atlas ./data --offline              # encoder from cache
```

Config: `atlas = "atlas-v2-lite"` in `dropoutt.toml`.

| | Default sample |
| --- | --- |
| atlas-v2-lite | 50,000 |
| atlas-v2 | 200,000 |

Fingerprints store `atlas_version`, `pipeline_hash`, and `encoder_weight_hash`.
Cross-product comparison is **refused** (existing behaviour).

---

## Build workflow

```
tools/fetch_corpus.py     →  shared cache (~2.7M rows)
        ↓
dedup + tokenize + embed at 256-d (once, memmap)
        ↓
┌───────────────────────┬────────────────────────────┐
│  build --profile full │  build --profile lite      │
│  norm 256-d           │  truncate 16-d, norm       │
│  L1=256, Leiden       │  L1=16, Leiden (coarse)    │
│  → atlas-v2.npz       │  subsample 130k            │
│                       │  → atlas-v2-lite.npz       │
└───────────────────────┴────────────────────────────┘
```

Client and build share `dropoutt.atlas` (extract, embed, normalize, assign) so
coordinates stay comparable for a given product + pipeline hash.

---

## Quality gates (before ship)

| Check | atlas-v2 | atlas-v2-lite |
| --- | --- | --- |
| IDF token mass | ≥ **99%** | N/A |
| Axis floors | all met post-dedup | met in subsample |
| Non-English share | ≥ 30% | ≥ 30% in subsample |
| Min community size | all cells ≥ 200 (after merge) | same |
| 256-d quant vs float32 | cell placement measured; document drift | 16-d slice stable vs reference |
| Lite vs full (reference corpus) | — | ≥90% coarse-subject agreement |

---

## Relationship to atlas-v1-lite

| | atlas-v1-lite (shipped) | atlas-v2 |
| --- | --- | --- |
| Products | one bundle | two (`atlas-v2`, `atlas-v2-lite`) |
| Embed dims | 128 | 256 / 16 |
| L2 allocation | k-means budget (800 cap → 215 cells) | Leiden (count emergent) |
| User-facing tiers | L1 + L2 in one report | flat cells only |
| Reference records | 2,125,556 | ~2.1M (full), ~130k (lite) |
| IDF token mass | 98.91% at 120k types | target ≥99% |

v1 fingerprints remain valid for v1; v2 is a new coordinate system and a new
release decision.

---

## Implementation notes

- Quantise encoder to **256 columns** once; lite uses `load_embedder(out_dim=16)`,
  full uses `out_dim=256`. Same cache directory.
- Warmup must pass `atlas.dim` instead of hardcoding `EMBED_DIM=128`.
- `pipeline_hash` must include embed profile, Leiden parameters, and `l2_method`.
- Record `n_regions` and Leiden settings in every artifact; do not publish a
  fixed cell count in user-facing docs before a build completes.
- Raise `fetch_corpus` `MAX_CHARS` to **4000** for v2 full build parity with the
  runtime char window (lite build may truncate earlier).
