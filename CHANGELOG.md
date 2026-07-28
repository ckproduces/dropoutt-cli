# Changelog

## 0.2.0

Publish polish: atlas report storytelling, scan speed, packaging.

### Atlas report

Coverage panels (terminal + HTML) lead with what the atlas is for: what you
cover, what you miss, and what sits off the map. User excerpts name regions
before atlas captions; missing subject areas and off-map diagnosis are
promoted; short-record exclusions and region cohesion appear when relevant.

### Scan performance

- Reuse the CLI discovery walk (no second filesystem scan).
- Parallel layout induction across datasets (thread pool).
- Chunked MinHash for long documents; one-pass surface features; batched
  content hashing.

### Packaging

- `py.typed`, Beta / OS-independent / Typing classifiers, sdist includes,
  GitHub Actions CI on Linux / macOS / Windows.
- Windows cache under `%LOCALAPPDATA%\dropoutt`.
- README rewritten with a full check-code catalog.

## 0.1.5

Three things, all of which came out of running the tool on real data and not
liking the answer.

### A folder of records read as a folder of documents

A scan of 248 `.txt` files reported **251 records** across **251 datasets**, at
one document each, and inferred the `corpus` profile. The files held **6,097 SFT
records**. Every conversational check was skipped — and not skipped visibly, in
the "not checked, and why" list, but never considered, because as far as the
scanner was concerned there were no conversations. Five findings came back where
there were fifteen.

Two independent causes, both fixed:

- **`.txt` and `.md` files are now sniffed rather than assumed.** A new brace-
  balanced scanner (`dropoutt.sniff`) finds every top-level JSON object in a
  file, which collapses line-delimited, blank-separated and ```json-fenced
  framings into one algorithm and makes the text *between* records fall out for
  free. Two rules keep it off real prose: most spans must parse as objects, and
  the spans must cover most of the file's non-whitespace bytes — so a blog post
  quoting one JSON snippet stays a blog post. Reading is incremental, so a
  mislabelled multi-gigabyte dump costs bounded memory.
- **Sharded siblings fold into one dataset.** `responses_0001.txt` … and
  `train-00000-of-00042.parquet` reduce to their family name when at least three
  siblings share it. Naming each shard as its own dataset made every per-dataset
  statistic a statistic about one shard, and made the overlap matrix compare a
  corpus against itself.

### The atlas produced a panel, not an answer

It printed an occupancy count, a spread percentage, and five frequency terms per
region — none of which a reader could act on, and none of which could ever reach
the findings table.

- **Effective region count.** Occupancy counts a region holding one record the
  same as one holding a third of the corpus. The exponential of the region
  entropy says how many *evenly used* regions the corpus is as spread out as, and
  the gap between the two numbers is the size of the tail.
- **Coverage gaps — the part that needed a frozen atlas to exist.** A histogram
  of your own data says what is present. Only a fixed coordinate system can say
  what is *absent*. Subject areas the atlas covers and the corpus does not are
  now listed by name. On the corpus above: no code generation, no reading
  comprehension, no arithmetic, no SQL, nothing on religion or ethics.
- **Regions named by your own records.** The five shipped terms label a region
  *in the reference corpus*. The scan now shows the user's own record sitting
  closest to each busy region's centre, which is what makes the label mean
  anything. Kept in `ctx.stats`, never in `fingerprint.json`.
- **Topical overlap between datasets.** Cosine between each pair's region
  histograms. Two datasets can share no wording and still occupy identical
  ground, which `T1-OVERLAP-001` compares text and cannot see.
- **The map is coloured by subject area** rather than by occupied-or-not, which
  the occupancy metric already said.
- **`tools/build_atlas.py` now records the reference distribution.** v0 computed
  it and threw it away, which is why gaps can only be reported as absence and not
  as under-representation. The loader reads it when present and says nothing when
  it is not; `atlas-lite-v0` is unchanged and still loads.

### Eleven checks added, taking the catalog from 27 to 38

Synthetic SFT data fails fluently: the file parses, the records are well-formed,
the text reads well, and something is still wrong. Every threshold below was
measured on a 6,097-record Turkish generation run before the check was written,
and two of them are deliberately silent on it.

| id | what it caught there |
| --- | --- |
| `T0-FORMAT-001` | 248 text files holding records |
| `T0-SCHEMA-005` | 37 records whose answer sat in a top-level key the layout never reads |
| `T0-REASON-001` | 17.9% of responses opened a `<think>` block and 82.1% did not |
| `T0-GEN-001` | 5 runs of generator scaffolding between records, including a leaked `<thinking_mode>off</thinking_mode>` |
| `T0-TRUNC-002` | nothing — correctly, the corpus is not capped |
| `T1-DUP-002` | nothing — correctly, no prompt is answered two ways |
| `T0-QUAL-001/002/003` | the three FineWeb line-shape filters, at FineWeb's thresholds, for the `corpus` profile |
| `T1-ATLAS-001/002` | topical narrowness and template-shaped crowding |

`T0-SCHEMA-005` is the one worth reading twice. A record can be well-formed and
still lose its answer: a complete assistant turn sitting in a top-level
`assistant` key *beside* `messages` rather than inside it. Every key the trainer
reads is valid, nothing complains, and the conversation ends on a user turn and
trains on nothing.

Both silent checks were made silent by evidence rather than by luck.
`T0-TRUNC-002` first fired on any distribution with few discrete response
lengths; it now requires the top length bucket to hold at least three times as
many records as any bucket within 250 characters beneath it, because a real cap
is a wall and a coarse distribution is not. `T1-ATLAS-001` first reported a
corpus touching 195 regions with an effective spread of 63 as concentrated,
which it is not; it now tests absolute narrowness instead of a ratio that every
long-tailed distribution satisfies.

### Existing checks made more precise

**`T0-DEGEN-001` no longer calls extraction degenerate.** It flagged a response
as copying the prompt whenever one contained the other. On a real corpus that
was wrong 38 times out of 38: every hit was either a multiple-choice answer —
necessarily a substring of the prompt that listed the options — or extractive QA,
where the instruction was literally "quote the answer from the passage". Both are
correct supervision. Containment now has to account for at least 90% of the other
side, so picking one option out of five, or quoting one sentence from a passage,
no longer counts, while a response that simply repeats the prompt still does.

### Compatibility

`PIPELINE_VERSION` moves to 0.1.5. Record counts, dataset counts and the inferred
profile all change for inputs holding records in text files, and the coverage
facet gains keys, so fingerprints from 0.1.4 describe something different and
must not be compared against these.

## 0.1.4

An off-atlas rate above 10% used to discard the entire coverage report and print
a sentence in its place:

```
    withheld 15% of records are off-atlas, above the 10% threshold. This atlas
    does not fit this corpus, so coverage numbers would be misleading and have
    been withheld.
```

That was wrong twice over. Off-atlas records are filtered out of the region
histogram *before* it is counted, so the histogram was never contaminated by them
and withholding it discarded a measurement that was correct for every record it
covered. And the off-atlas set is the most useful thing the atlas produces on a
corpus that does not fit it — it is a list of the records unlike anything in the
reference corpus — which the scan already knew and then threw away.

### Changed

- **Coverage is never withheld.** The histogram is reported whenever anything
  placed. Every share is over the **placed** records and the placed count is
  printed beside it, so the denominator is never implied. The only remaining
  no-numbers case is `none placed`, where the numbers genuinely do not exist.
- **Fit is graded, not passed or failed**: `good` at or below 10% off-atlas,
  `partial` to 35%, `poor` above. The underlying quantity is continuous and a
  corpus at 10.1% is not meaningfully different from one at 9.9%. Ten percent is
  five times the 2% an atlas-distributed corpus sits at by construction, since the
  cutoff is the 2nd percentile of the atlas's own reference records.
- **`dropoutt diff` no longer refuses a corpus with a high off-atlas rate.** It
  prints both placed shares and states which way the bias runs: a partial *right*
  side means regions it appears not to reach may be reached by records it could
  not place, so `New` is an upper bound. It still refuses across atlas versions,
  and for fingerprints written before this release whose histogram is already gone.

### Added

- **Off-atlas records are described rather than counted.** The report gives the
  similarity distribution against the cutoff, the regions they were nearest to
  anyway, the rate broken down by language and by dataset, the furthest excerpts,
  and a one-line reason.
- **The reason is attributed in measured order.** Similarity to a region rises
  steeply with record length: the same English paragraph scores **0.363 truncated
  to 20 characters and 0.787 at 2000**, landing in the same region throughout;
  across a real corpus the correlation between log length and similarity is about
  **0.49**, and the off-atlas rate falls from **33% under 80 characters to 0% above
  150**.

  The order of tests is: whether the records are written like prose at all, then
  length, then whether the off-atlas set is *coherent* rather than scattered, then
  concentration in one dataset or language, then whether they are simply near
  misses at the threshold.

  **Coherence deliberately does not claim a missing subject area.** It sounds like
  it should, and the measurements say otherwise: minified JavaScript scores 0.969
  mean pairwise cosine, HTML boilerplate 0.961, DNA 0.947, base64 0.871, against
  **0.277 for real prose** — and a genuinely missing topic (Ottoman
  endowment-deed vocabulary) scores 0.886, in the middle of the templates. High
  coherence means the records are alike, full stop. What separates a template from
  a subject is how the text is written, so the surface test runs first and the
  coherence sentence stops at what was measured.
- **Whitespace and non-letter share** of the off-atlas set against the placed set,
  which is what actually distinguishes markup, minified code, encoded blobs and
  log lines from a subject the atlas lacks. Measured shares: whitespace 0.158 for
  prose and 0.132 for the missing-topic case, against 0.000 for base64 and DNA and
  0.037 for HTML; non-letter 0.048 for prose and 0.000 for the missing topic,
  against 0.191 for base64 and 0.556 for hex logs.
- `Atlas.assign_full` returns the nearest region alongside the placement.
  `assign` computed it and threw it away, which cost the caller the one thing that
  makes an off-atlas record legible.

`docs/atlas.md` also now states plainly that **off-atlas is not the garbage
detector**. Machine formats usually place, confidently and wrongly, rather than
going off-atlas: on a corpus of 400 records where 100 were base64 and minified
JavaScript, the off-atlas count was zero and the blobs landed in a region labelled
`return, denklemin, array, tdrow, function`. Those records were caught by
`T1-LANG-001`, which flagged exactly 100 of 400. The checks are the instrument for
junk; a coordinate system will happily give nonsense a coordinate.

### Fixed

- **`by_category` counted every record while `region_counts` counted only placed
  ones**, putting two denominators in the same panel so a category share and a
  region share could not be read against each other. Both are now over placed
  records.
- Off-atlas excerpts are kept out of `fingerprint.json`, which is the artifact
  meant to be shareable, and respect `--no-evidence`.
- `docs/atlas.md` described a "33-word stoplist" for region labels. The literal
  has 38 entries and **only 16 of them do anything**: the other 22 are three
  characters or fewer and were already dropped by the length filter one step
  earlier. The labels also use the first 150 members in corpus order, not a
  random sample, which the docs did not say.

### Fixed (packaging and diagnostics)

- **`dropoutt doctor` now names the interpreter it probed.** Every status in that
  table is an import against one specific Python, and the common way to be
  confused by it is to install with a `pip` belonging to a different one. A venv
  created by `uv` ships no `pip`, so an activated shell falls through to whatever
  `pip` is next on `PATH` and installs somewhere the tool cannot see — the package
  installs successfully and the status stays `no`. `doctor` prints the interpreter
  path, and when anything is missing it prints an install command targeting that
  interpreter (`uv pip install` when the running Python has no `pip`).
- **The version had two sources and they drifted.** `pyproject.toml` and
  `src/dropoutt/__init__.py` each carried a literal, so 0.1.4 was tagged in one
  while `dropoutt doctor` reported 0.1.3 from the other. `pyproject.toml` now
  reads the version from the package via `[tool.hatch.version]`; bump
  `__init__.py` only.

`PIPELINE_VERSION` is bumped because both the presence and the denominator of the
coverage facet changed.

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
