# Changelog

## 0.1.3

### Fixed

- `dropoutt diff` now rejects atlas `.npz` files, unrelated JSON, and malformed
  fingerprints with a command-specific explanation and example. Commands with
  required arguments print full help when run empty.
- Fingerprint `content_hash` now hashes normalized record content and structure.
  It previously hashed only dataset names, byte sizes, and record counts, so
  different same-sized corpora could receive the same fingerprint id.
- Fingerprint ids now include the effective profile, sequence length, tier,
  MinHash preset, and other runtime configuration after CLI overrides.
- Blocking now follows the declared `target`, not the inferred data profile.
- `tier` and `minhash_preset` from `dropoutt.toml` now reach the scanner.
- `eval_sets` now acts as a benchmark-name allowlist and rejects unknown names;
  it was previously parsed and then ignored.
- Malformed TOML is a usage error instead of being silently ignored. Python
  3.10 installs `tomli` so configuration works on every supported version.
- `init` no longer writes into the current directory when its requested path
  does not exist. `index-eval` rejects directories, unsupported extensions,
  empty inputs, and path traversal in benchmark names, and requires `--force`
  before overwriting an index.
- `scan` now rejects paths with no supported data files instead of returning a
  successful empty fingerprint.
- `.bz2` and `.xz` inputs are now actually decompressed. They were classified
  as supported but read as plain text. `.zst` input has an optional dependency
  and produces an install hint instead of gibberish when it is absent.
- Genuine internal failures exit 1 with a concise message. Set
  `DROPOUTT_DEBUG=1` to retain a traceback.
- `fetch` exits 1 when cache preparation is incomplete instead of printing a
  failure and returning success.

### Added

- `--no-evidence` omits record excerpts and source locations from terminal
  output, `findings.jsonl`, and `report.html`.
- The HTML report now uses a minimal, system-sans layout and plots occupied
  atlas regions on the frozen 2D atlas coordinates without scripts or network
  resources.
- Long CLI operations now show their active phase with a terminal spinner.
  Redirected batch output receives stable phase lines and periodic record
  counts.
- Arrow IPC, Feather, and ORC inputs are supported through the existing
  `pyarrow` extra. The model registry adds ten current model options and the
  benchmark registry adds MATH-500, AIME 2024/2025, BigCodeBench, and
  MMLU-Redux 2.0.
- `DROPOUTT_OFFLINE=1` and an existing `HF_HUB_OFFLINE=1` are honored by
  `scan` and `init`. `init` also accepts `--offline`.
- Reports and write-time output now state their confidentiality boundary.
- Fingerprints no longer carry contamination witness paths, and
  `--no-evidence` removes nested witness locations from structured finding data.
  Documentation no longer describes unkeyed private evaluation indices as safe
  to publish. New private indices store the input basename rather than its
  absolute path.

`PIPELINE_VERSION` is bumped because fingerprint identity changed.

## 0.1.2

The atlas was computed on every scan and then displayed nowhere. Coverage went
into `fingerprint.json` and never reached the terminal or the HTML report, and
there was no way to put two datasets side by side, which is the only thing a
shared coordinate system is for.

### Added

- **Atlas coverage in the scan output**, in the terminal and in `report.html`:
  regions occupied, spread against even coverage, off-atlas rate, top categories
  by name, and top regions with their label words. Suppressed coverage prints
  the reason rather than the numbers.
- **`dropoutt diff LEFT RIGHT`** compares two fingerprints across the shared
  atlas and answers "what does LEFT cover that RIGHT does not". Reports
  similarity, shared and new mass, the regions unique to each side, and the
  category mix with deltas.

  Directional, read left against right, for the same reason cross-dataset
  overlap is: a small specialised corpus can sit wholly inside a large one while
  the large one is barely inside it, and a symmetric score hides exactly the
  case worth acting on.

  It refuses rather than guesses. If either side had coverage suppressed, or the
  two fingerprints were built against different atlas versions, it says so and
  stops. It also does not rank datasets: whether new coverage helps depends on
  what you are training.
- **The full region histogram in the coverage facet.** `top_regions` is a
  display head capped at twelve, and comparing on it computed shares over a
  fraction of each corpus. On two real corpora the head covered 88% of one side
  and 36% of the other, overstating novelty as 100% where the true figure was
  62%. Only occupied regions are stored, so this costs a few hundred small
  integers.
- Category ids are resolved to names everywhere they are shown. `"10": 59` is
  not information.

`PIPELINE_VERSION` is bumped because the coverage facet changed.

## 0.1.1

Everything here was found by writing [docs/getting-started.md](docs/getting-started.md)
and running each documented command instead of assuming it worked.

### Fixed

- **The package could not be installed at all.** A `force-include` of
  `src/dropoutt/data` duplicated files that `packages` already carried, and
  hatchling refused to build: *"A second file is being added to the wheel
  archive at the same path"*. Removed; the wheel ships 4.4 MB of data as
  intended.
- **`--offline` reached the network.** `scan()` took no `offline` argument, so
  atlas coverage tried to download a ~500 MB embedding model on a node with no
  egress. `offline` is now threaded to the embedder, and a test fails on any
  socket connection during an offline scan.
- **`--offline` refused to use an already-populated cache.** The tokenizer and
  the chat template both returned empty rather than reading the cache, which
  silently changed every token number and disabled the loss-mask checks. Both
  now resolve locally, and an offline scan produces findings byte-identical to
  an online one.
- **`dropoutt fetch` did not exist**, although skipped-check messages told users
  to run it. Implemented: pre-downloads tokenizers, each model's
  `tokenizer_config.json`, and the atlas embedder into `DROPOUTT_CACHE`.
- **The fingerprint reported bytes on disk as `total_chars`, and `total_words`
  was hardcoded to 0.** Byte counts include JSON syntax and vary with language
  under UTF-8, so the one artifact meant to be comparable across datasets varied
  with file format. On the test fixture this reported 91,976 characters where
  there were 54,577.
- **`index-eval` wrote into the install tree**, failing on read-only
  site-packages, and building a private index silently switched off all ten
  bundled benchmarks. Indices are now written to the cache, and both locations
  are always searched.
- **Install hints lost their extras name.** rich parsed `[tokenizer]` as a style
  tag and deleted it, so users were told to `pip install 'dropoutt'`. All
  data-derived and hint strings are escaped at the render site; a dataset named
  `[red]` can no longer restyle the output either.
- `--quiet` help text described behaviour it does not have.

### Added

- `python -m dropoutt`, for the common case where the venv's `bin/` is not on
  `PATH`.
- [docs/getting-started.md](docs/getting-started.md).
- Honest documentation of what the atlas region labels are, why they read as
  random words, and what the level-0 accuracy figure does not mean. See
  [docs/atlas.md](docs/atlas.md).

## 0.1.0

First release. Tier 0 and Tier 1 checks, a fingerprint, and a first atlas.

### Added

- **27 checks** across two tiers. Tier 0 is structural and needs no model; Tier 1
  is statistical. `dropoutt checks` lists them; `dropoutt checks <id>` explains
  one.
- **Zero-configuration scan.** Discovery, schema induction against ten known
  layouts, detection of files that are not training data at all, hypothesis-based
  profile inference, and a cross-tokenizer token budget when no `--model` is given.
- **Contamination scanning** using the Tülu 3 rule, against 10 bundled hashed
  8-gram indices for permissively licensed benchmarks. `dropoutt index-eval`
  builds an index from your own held-out set locally.
- **`atlas-lite-v0`**: 258 regions over 152,622 reference records, level-0
  taxonomy accuracy 0.864, region purity 0.785. Coverage is suppressed when the
  off-atlas rate exceeds 10%, and is reported per language.
- **Self-contained HTML report**, one file, no server and no CDN.
- **Three-valued exit codes**: 0 completed, 1 internal error, 2 usage, 10
  blocking findings under a declared target.
- Verified static registries: 22 benchmarks with their genuinely scorable eval
  splits, 12 chat-template families with literal delimiters, 37 models including
  the Turkish set, PII patterns with checksums.

### Policies enforced in code and covered by tests

- Nothing blocks a run whose purpose was never declared.
- No check recommends deleting data without a measured effect; every finding in
  this release is `unverified`.
- Cross-dataset overlap is directional.
- PII never reaches the report; a test fails if a planted secret appears.
- Anything that degraded says so.

### Known issues

See [docs/limitations.md](docs/limitations.md). The largest are that language
identification is the main source of false positives, scale above roughly a
million records is unprofiled, and the scan is single-process.
