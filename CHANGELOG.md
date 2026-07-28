# Changelog

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
