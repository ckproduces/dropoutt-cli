# dropoutt

**Pre-flight checks for LLM training data.** Point it at a folder. It tells you
what is wrong before you burn a training run finding out.

```bash
pip install -e '.[all]'
dropoutt scan ./data
```

No model, no config, no flags required. Add `--model` or `--target` to unlock
more checks and CI gating. Skipped checks always name the one flag that unlocks
them.

## What it is

A local CLI that:

1. **Scans** training datasets for structural bugs (empty loss masks, broken
   roles, truncation that kills the answer, contamination, PII, language damage).
2. **Fingerprints** the corpus so two datasets can be compared without shipping
   records.
3. **Maps** your data onto a frozen **atlas** — a latent coordinate system built
   from professional public datasets — so you see what you cover, what you miss,
   and what sits off the map.

It runs on your CPU. Optional extras add tokenizers, Parquet, language ID, and
atlas embeddings.

## Install

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[all]'            # or pip install -e '.' for core only
dropoutt doctor                    # what is installed, what each missing piece costs
```

| Extra | Purpose |
| --- | --- |
| *(core)* | Inventory, schema, dedup, overlap, contamination, PII, style |
| `tokenizer` | Exact token counts, chat template, loss mask, packing |
| `lid` | Language identification (938 KB model) |
| `atlas` | Atlas coverage map (~500 MB embedder on first use) |
| `parquet` | `.parquet`, `.arrow`, `.feather`, `.orc` |
| `zstd` | `.zst` compressed input |
| `fast` | `orjson` + Rust MinHash — same results, faster |
| `all` | Everything above |
| `dev` | `all` + pytest |

If `dropoutt` is not on `PATH` (module systems, batch schedulers, some Windows
setups): `python -m dropoutt` does the same thing.

Supported inputs: JSON, JSONL/NDJSON, TXT, Markdown, CSV/TSV, Parquet, Arrow,
Feather, ORC. Text formats may be gzip / bzip2 / xz / zstd compressed.

Works on macOS, Linux, and Windows. Cache defaults to `~/.cache/dropoutt`, or
`%LOCALAPPDATA%\dropoutt` on Windows. Override with `DROPOUTT_CACHE`. See
[docs/portability.md](docs/portability.md) for offline / HPC use.

## Quick start

```bash
dropoutt scan ./my-corpus
# writes .dropoutt/{report.html, report.md, findings.jsonl, fingerprint.json}

dropoutt scan ./my-corpus --model qwen3 --seq-len 4096 --target sft
# unlocks token/mask checks and exit code 10 on blocking findings

dropoutt checks                 # live catalog
dropoutt checks T0-MASK-001     # one check in detail
dropoutt doctor                 # what is installed, what each gap costs
dropoutt fetch                  # pre-download everything --offline needs
```

The report is one self-contained file: no CDN, no web fonts, no network, opens
from `file://`. A scan opens it for you when there is a desktop to open it on,
and quietly does not when there is not — over SSH, in CI, under a batch
scheduler, or with output redirected. `--no-open` or `DROPOUTT_OPEN=0` turns
that off; `DROPOUTT_OPEN=1` forces it, which is what you want with X11
forwarding.

Anything above 24 MB is scanned across processes — 200,000 SFT records in about
20 seconds on a laptop. The result does not depend on how many cores you have:
same findings, same examples, same fingerprint id on one core or on sixteen. Cap
it with `-j` or `DROPOUTT_WORKERS` if you are sharing a node.

**New here?** [docs/getting-started.md](docs/getting-started.md).

## What it catches

Bugs that waste a whole training run without appearing in the logs:

- **Records that train nothing** — empty loss masks from role-name mismatches
  (`from: "gpt"` vs `role: "assistant"`).
- **Truncation that removes the answer** — including cases where the entire
  assistant span falls beyond `--seq-len`.
- **Benchmark contamination** — Tülu 3 rule against bundled hashed 8-gram indices.
- **Files that are not training data** — agent session logs and telemetry that
  look like chat.
- **Directional overlap** — a small set wholly contained in a large one.
- **Language damage** — e.g. Turkish that lost its diacritics (`degil mi`).
- **Atlas shape** — specialised vs broad coverage, missing subject areas, and
  regions of near-identical writing that shingle dedup cannot see.

## Atlas (coverage map)

The atlas is a **frozen topical map** compressed from high-quality public
datasets (258 regions, static multilingual embeddings, CPU-only). Every scan
with the `atlas` extra places a sample of your records on that map and reports:

| Section | What you learn |
| --- | --- |
| **What the map says** | A handful of sentences that clear both a size gate and a significance gate — a subject 8x denser here than the map is built for, an area the map spends a fifth of itself on that you barely reach. Nothing is shown for being true; it is shown for being large *and* true |
| **Where your data piles up** | The five crowded places, named by *your own record* nearest the centre of each, because that is the only description of a neighbourhood that is true by construction |
| **Where you have only a toehold** | The sparsest places you reach. Reaching a place is not covering it, and an occupancy count cannot tell the difference |
| **Shape** | Specialised or broad — right for a single-task set, wrong for a pretraining mixture, and the tool does not know which you are building |
| **Crowding** | One area holding half the corpus whose records are 0.98 alike is one template, not one topic — and shingle dedup cannot see it |
| **Same ground** | Datasets that occupy the same regions even when they share no wording, i.e. merging them adds volume and not coverage |
| **Off the map** | Records unlike the reference geography, with a diagnosis (often length or markup, not “bad data”) |

The atlas's own five-word captions for a region are shown as captions and never
as findings: they are frequency counts over reference records, roughly 40% of
that text is function words shared with other regions, and the subject-area
names were assigned per source dataset rather than per record. What the map is
trusted for is geometry. Details and the full list of what that costs:
[docs/atlas.md](docs/atlas.md).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Completed (findings or not) |
| 1 | Internal error |
| 2 | Usage error |
| 10 | Blocking findings — only when `--target` was declared |

## Check catalog

Identifiers are `T{tier}-{GROUP}-{nnn}` and are **never renumbered**. Mute by id
in `dropoutt.toml`. Full narrative: [docs/checks.md](docs/checks.md). Live list:
`dropoutt checks`.

### Tier 0 — structural (CPU)

| id | What it means |
| --- | --- |
| `T0-SCHEMA-001` | Files are not training data (logs / telemetry) |
| `T0-SCHEMA-002` | One folder mixes several record layouts |
| `T0-SCHEMA-003` | Records failed to parse |
| `T0-SCHEMA-004` | Message content was not a string |
| `T0-SCHEMA-005` | Content sits in keys the layout never reads |
| `T0-FORMAT-001` | Plain-text files are holding structured records |
| `T0-GEN-001` | Generator scaffolding outside the records |
| `T0-REASON-001` | Only some responses carry a reasoning trace |
| `T0-TRUNC-002` | Responses stop at a generation length cap |
| `T0-QUAL-001` | Documents whose lines mostly lack punctuation (corpus) |
| `T0-QUAL-002` | Documents built mostly from very short lines (corpus) |
| `T0-QUAL-003` | Documents repeating their own lines (corpus) |
| `T0-ROLE-001` | Conversation role structure is invalid |
| `T0-ROLE-002` | Role names are not the canonical vocabulary |
| `T0-TMPL-001` | Data is already formatted with a chat template |
| `T0-TMPL-002` | Records fail to render with the target chat template |
| `T0-MASK-001` | Records contribute zero trainable tokens |
| `T0-MASK-002` | Stop token is outside the trainable span |
| `T0-TRUNC-001` | Records exceed the sequence length |
| `T0-PACK-001` | Packing efficiency under concat-and-chunk |
| `T0-ENC-001` | Text encoding is damaged |
| `T0-DUP-001` | Exact and whitespace-identical duplicates |
| `T0-DEGEN-001` | Degenerate responses |

### Tier 1 — statistical

| id | What it means |
| --- | --- |
| `T1-NDUP-001` | Near-duplicate records (MinHash; reports, does not delete) |
| `T1-DUP-002` | Same prompt answered two different ways |
| `T1-OVERLAP-001` | Datasets overlap with each other (directional) |
| `T1-ATLAS-001` | Corpus sits in very few topical regions |
| `T1-ATLAS-002` | A crowded region holds near-identical records |
| `T1-CONTAM-001` | Training data overlaps evaluation benchmarks |
| `T1-LANG-001` | Language composition and detection confidence |
| `T1-LANG-002` | Records deviate from the dataset’s main language |
| `T1-LANG-003` | Script does not match the detected language |
| `T1-PII-001` | Personal data and credentials in training text |
| `T1-IDENT-001` | Assistant identity leakage and refusal boilerplate |
| `T1-STYLE-001` | Formulaic response openings |
| `T1-LIC-001` | Datasets have no recorded licence |

Every finding in this release is labelled `unverified`: no calibration corpus
yet links acting on a finding to a measured change in model quality.

## Progressive disclosure

| What you give | What it unlocks |
| --- | --- |
| nothing | inventory, schema, dedup, overlap, bundled contamination, language, PII, atlas (if installed) |
| `--model` | exact tokens, fertility, truncation, template, loss mask, stop token, packing |
| `--target` | pass-or-fail gating (exit 10) |
| the `atlas` extra | where the corpus sits on the map, and what it misses |

## Documentation

- [Getting started](docs/getting-started.md)
- [CLI reference](docs/cli.md)
- [Check catalog](docs/checks.md)
- [Fingerprint](docs/fingerprint.md)
- [Atlas](docs/atlas.md)
- [Configuration](docs/configuration.md)
- [Portability / offline](docs/portability.md)
- [Limitations](docs/limitations.md)
- [Design rules](docs/design.md)

## What it will not do

- Fail a run whose purpose you never declared (`--target`).
- Tell you to delete data it has not measured for downstream effect.
- Write raw PII values into reports (matches are masked).
- Extract text from PDFs — point it at extracted text instead.

## Develop

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
python -m build && twine check dist/*
```

Lint rules live in `pyproject.toml`, and every rule that is switched off says
why. Two conventions are worth knowing before reading the source:

- **Imports go inside functions** wherever the import is expensive or optional.
  `dropoutt --help` should not pay for numpy, tokenizers and the atlas.
- **Every fast path has a slow one beside it.** `tests/test_fastpaths.py`
  checks the vectorised implementations against the obvious ones they replaced.
  The contamination hashes in particular are frozen — the shipped `.idx` files
  are tables of exactly those numbers, and no benchmark text exists anywhere to
  recompute them from.

## Licence

dropoutt is Apache-2.0.
