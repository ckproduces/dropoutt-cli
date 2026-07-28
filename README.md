# dropoutt

Pre-flight checks and comparable fingerprints for LLM training datasets.

Point it at a folder. It tells you what is wrong with your training data before
you spend a run finding out.

```bash
pip install -e '.[all]'
dropoutt scan ./data
```

No model, no configuration and no flags are required. Supplying them unlocks
more checks, and the report always names the single flag that would unlock each
one it skipped.

**New here?** [docs/getting-started.md](docs/getting-started.md) walks from
installation through reading the output to blocking a bad run in CI.

## What it catches

The checks that matter most are the ones that waste a whole training run without
appearing anywhere in the logs.

- **Records that train nothing.** A record whose label vector is entirely masked
  contributes no gradient, occupies space in the packed block, and is invisible.
  The usual cause is a role name the trainer does not recognise: a ShareGPT
  `from: "gpt"` turn is not `role: "assistant"`, and pipelines that mask on the
  latter silently drop every such record.
- **Truncation that removes the answer.** Not just how many records exceed your
  sequence length, but how many lose their *entire* assistant span, which is a
  different and worse problem.
- **Benchmark contamination.** The Tülu 3 rule, applied against hashed 8-gram
  indices that ship with the package. Removing contamination usually makes your
  reported score go down; that is the point.
- **Files that are not training data at all.** Agent session logs and telemetry
  traces are structurally close enough to chat data that importers ingest them
  happily.
- **Directional dataset overlap.** If a small dataset is wholly contained in a
  large one, the containment is 100% in one direction and 1% in the other. A
  symmetric similarity score hides exactly the case worth acting on.
- **Turkish that lost its diacritics.** Language identification will confidently
  call `degil mi` Turkish, and it is, but it is damaged Turkish that will teach a
  model the wrong orthography.

Full catalog: `dropoutt checks`, or [docs/checks.md](docs/checks.md).

## What it will not do

- **It will not fail a run whose purpose you never declared.** Blocking means
  asserting something is wrong *for a goal*. Without `--target`, findings are
  reported with "would block under sft" and the exit code stays 0.
- **It will not tell you to delete data it has not measured.** FineWeb
  deduplicated Common Crawl across all snapshots, got a corpus that scored below
  their baseline, and found that the data the filter discarded trained a *better*
  model than the data it kept. So this tool reports cluster sizes and lets you
  decide. Every finding in this release is labelled `unverified`, because no
  calibration corpus exists yet.
- **It will not put your secrets in the report.** PII matches are masked before
  they reach any output file, and there is a test that fails if a planted secret
  appears in a generated report.
- **It will not extract text from PDFs.** Use trafilatura or docling and point
  dropoutt at the output. Checking extraction quality is in scope; writing
  another PDF parser is not.

## Progressive disclosure

| what you give it | what it unlocks |
| --- | --- |
| nothing | inventory, schema induction, deduplication, cross-dataset overlap, contamination against bundled benchmarks, language, PII, degeneracy, cross-tokenizer token budget |
| `--model` | exact token counts, fertility, truncation forecast, template render, loss mask, stop token, packing efficiency |
| `--target` | pass-or-fail gating |
| `dropoutt index-eval` | contamination against *your* held-out set, indexed locally |

## Exit codes

| code | meaning |
| --- | --- |
| 0 | completed, findings or not |
| 1 | internal error |
| 2 | usage error |
| 10 | blocking findings, and only when a target profile was declared |

A checker that returns the same code for "found problems" and "crashed" cannot
be used in CI, which is why these are distinct.

## Installation

The core install has one compiled dependency, `tokenizers`, chosen because it is
the one that still ships `manylinux_2_17` and therefore installs on old cluster
images. Everything faster is an optional accelerator with a fallback that
produces the same answers.

Not on PyPI yet, so install from the checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.'              # core
.venv/bin/pip install -e '.[lid]'         # language identification, 938 KB model
.venv/bin/pip install -e '.[tokenizer]'   # exact token counts and template checks
.venv/bin/pip install -e '.[parquet]'     # .parquet input
.venv/bin/pip install -e '.[all]'         # everything
```

Run `dropoutt doctor` to see what is installed and what each missing piece costs.

If `dropoutt` is not on your `PATH` — common under module systems and batch
schedulers — `python -m dropoutt` does the same thing.

## Documentation

- [docs/getting-started.md](docs/getting-started.md) — **start here**: install to first scan
- [docs/index.md](docs/index.md) — quickstart
- [docs/cli.md](docs/cli.md) — every command and flag
- [docs/checks.md](docs/checks.md) — the check catalog
- [docs/fingerprint.md](docs/fingerprint.md) — fingerprint schema and evidence grades
- [docs/atlas.md](docs/atlas.md) — what the atlas is and how it was built
- [docs/configuration.md](docs/configuration.md) — `dropoutt.toml`
- [docs/portability.md](docs/portability.md) — clusters, offline use, fallbacks
- [docs/limitations.md](docs/limitations.md) — what this release does not do
- [docs/design.md](docs/design.md) — the rules and why they exist

Apache-2.0.
