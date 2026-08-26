# Command reference

Every command below also works as `python -m dropoutt <command>`. The console
script lands in the environment's `bin/`, which module systems and batch
schedulers often leave off `PATH`; the module form works from any interpreter
that can import the package.

`dropoutt` with no command prints the mark and the help. `scan` and `atlas`,
which take a required argument, print their complete help and a set of examples
when run with no arguments.

Help and version are spelled both ways, because a tool should not answer a
reasonable guess with a usage error:

```bash
dropoutt --help        dropoutt help
dropoutt -h            dropoutt help scan
dropoutt --version     dropoutt version
dropoutt -V
```

The commands below are the whole surface, and it is stable: `dropoutt` follows
semantic versioning from 1.0, so a command or a flag will not change meaning
inside a major version. `atlas` was added in 1.3 and `--no-atlas` removed with
it — the map did not change meaning, it moved to a command of its own.

## `dropoutt scan PATH`

Scan a file or directory.

The CLI shows the active phase while it discovers files, infers layouts, scans
records, finalizes checks, and writes artifacts. In a
terminal this is a spinner; redirected batch output receives stable phase lines
and periodic record counts instead of control sequences.

### What it reads

| extension | format |
| --- | --- |
| `.jsonl`, `.ndjson` | line-delimited JSON. A `.jsonl` that turns out to be one pretty-printed array is read as one. |
| `.json` | a whole document: an array of records, an object of splits, or one object |
| `.parquet` | Apache Parquet, split across workers by row group |
| `.arrow`, `.feather` | Arrow IPC file or stream, and Feather V1 |
| `.orc` | Apache ORC |
| `.csv`, `.tsv` | delimiter detected from the header row |
| `.txt`, `.md` | plain text — unless it turns out to be holding JSON records, which is sniffed rather than assumed |
| `.mds` | MosaicML Streaming shards, decoded against the `index.json` beside them |
| `.tar` | WebDataset shards: consecutive members sharing a basename are one sample |

Text formats may be compressed with gzip, bzip2, xz or zstd. A `.tar.gz` is
handled by the archive reader itself.

Directories that never hold training data are not descended into — `node_modules`,
`site-packages`, `Library`, `.git`, build caches and about twenty others — and
filenames that are always toolchain metadata (`package.json`, `tsconfig.json`,
`CHANGELOG.md`, `requirements.txt`, …) are passed over even though their
extensions match. That is what makes pointing a scan at a home directory finish.

| flag | default | meaning |
| --- | --- | --- |
| `--model`, `-m` | none | target model id, local path, or alias. Unlocks the token-dependent checks. |
| `--profile`, `-p` | `auto` | `sft`, `corpus`, `preference`, or `auto` |
| `--target` | none | declare what you are building. **This is what enables blocking.** |
| `--seq-len` | from model config | training sequence length |
| `--tier` | config, then `1` | highest check tier to run |
| `--out`, `-o` | `<path>/.dropoutt` | output directory |
| `--offline` | off | never touch the network; resolve the model from the cache |
| `--limit` | none | max records per file, for a fast look |
| `--no-html` | off | skip the HTML report |
| `--no-open` | off | do not open the report when the scan finishes |
| `--no-evidence` | off | omit record excerpts and source locations from terminal output, `findings.jsonl`, `report.md`, `report.json`, and `report.html` |
| `--workers`, `-j` | sized from the machine | processes used for the streaming pass |
| `--brief` | off | print the verdict and one line per finding instead of the full report |
| `--quiet`, `-q` | off | suppress the terminal report; only the output path is printed |

```bash
dropoutt scan ./data
dropoutt scan ./data --model qwen3 --seq-len 4096
dropoutt scan ./data --model /scratch/models/qwen3-4b --offline
dropoutt scan ./data --target sft --quiet          # CI gate
dropoutt scan ./data --limit 1000                  # quick look at a huge corpus
dropoutt scan ./data --offline --no-evidence       # VPC-safe network and evidence settings
dropoutt scan ./data -j 4                          # cap the scan at four processes
dropoutt scan ./data --brief                       # verdict and one line per finding
```

Left to itself, `--workers` reads the machine rather than `os.cpu_count()`: CPU
affinity, any cgroup quota, physical cores rather than hyperthread siblings, and
free memory — because peak memory is roughly linear in worker count and a
96-core container with 8 GB is not a 96-worker machine. `dropoutt fetch` prints
what it decided and which limit bound it. An explicit `-j` is obeyed exactly.

### Opening the report

The scan opens `report.html` when it finishes. It does not when there would be
nobody to see it, and it says nothing in that case rather than reporting a
failure — the file is written either way, and the path is printed either way.

The checks, in order: `--no-open`, `--quiet` or `--no-html`, `DROPOUTT_OPEN=0`,
an SSH session (`SSH_CONNECTION`, `SSH_CLIENT`, `SSH_TTY`), CI (`CI`,
`GITHUB_ACTIONS`, `GITLAB_CI` and others), a batch scheduler (`SLURM_JOB_ID`,
`PBS_JOBID`, `LSB_JOBID`, `SGE_TASK_ID`), stdout that is not a terminal, and on
Unix a session with no `DISPLAY` or `WAYLAND_DISPLAY`. `DROPOUTT_OPEN=1` skips
every one of them, which is what you want when X11 forwarding means the SSH
check is wrong.

### Workers

Above 24 MB the streaming pass is split into contiguous shards and run across
processes; below it, and with `-j 1`, it runs in the calling process. The result
is identical either way — same findings, same examples, same fingerprint id —
because a shard reads a slice of the corpus in the order a serial scan would and
the checks are combined afterwards. `DROPOUTT_WORKERS` sets the default for a
whole machine; `-j` overrides it for one run. On a scheduler that allocates you
part of a node, set one of them to your allocation, because the default reads
the machine's core count and not your share of it.

`--model`, `--profile`, `--target`, `--seq-len`, `--tier`, and `--offline` fall
back to `dropoutt.toml` when not given on the command line, and `--seq-len`
falls back again to the model's own `model_max_length`.

`--offline` resolves the tokenizer and the chat template from the cache rather
than refusing to load them, so an offline scan reports the same numbers as an
online one. Populate the cache first with `dropoutt fetch`. If no cached chat
template is found, the run says so before the report rather than counting raw
text silently.

A scan does not draw the coverage map. That is
[`dropoutt atlas`](#dropoutt-atlas-path), and it is a separate command from 1.3
onwards: it is a different question — where the corpus sits rather than what is
wrong with it — and a different cost, because placement runs every sampled record
through a neural encoder that has to be downloaded once. `fingerprint.json` keeps
its `coverage` facet either way, so two fingerprints have the same shape; a scan
fills it with `not computed by scan (run dropoutt atlas)`.

By default, terminal examples, `findings.jsonl`, `report.md`, `report.json`, and `report.html` contain
bounded record excerpts and source locations so findings can be inspected.
Detected PII is masked, but arbitrary proprietary text is not. Use
`--no-evidence` before moving these artifacts outside the dataset's trust
boundary. The fingerprint never contains record excerpts, but it does contain
the scan root, dataset names, aggregate measurements, and stable hashes.

### What gets written

| file | what it is |
| --- | --- |
| `report.html` | the page: one self-contained file, no CDN, opens from `file://` |
| `report.md` | the same content as GitHub-flavoured Markdown, to paste into a PR or a ticket |
| `report.json` | the same content as one JSON document, for a dashboard or a coverage gate |
| `findings.jsonl` | one finding per line, for iterating over problems in a script |
| `fingerprint.json` | the comparable measurement, deliberately free of anything quotable |

The first three carry the same sections, in the same order, and so does the
terminal. That is a property the test suite asserts: a finding that reaches the
page reaches the log, and a section the page has is a section the Markdown file
has. `--no-evidence` is honoured once, where the content is assembled, so it
cannot be honoured in three formats and forgotten in the fourth.

## `dropoutt atlas PATH`

Place a corpus on the atlas and draw where it sits.

The atlas is a frozen coordinate system, not a collection of good datasets: one
map of 215 subregions across 48 subject areas, fitted once on public data, so
two corpora placed on it can be compared and a gap can be named. A typical
coverage plot fits UMAP or k-means on the sample in front of it, which means the
next folder gets a new projection and neighbourhoods stop meaning the same
thing. This command only assigns your records to bins that already exist.

It reads the same files a scan reads, in the same way, and samples the same
records — placement and a scan of the same corpus see the same sample. What it
does with them is different: every sampled record goes through a 128-dimensional
static encoder, which is the one part of the old scan whose cost had nothing to
do with the checks, and which has to be downloaded on first use. `dropoutt fetch`
gets it ahead of time; after that `--offline` works.

| flag | default | meaning |
| --- | --- | --- |
| `--out`, `-o` | `<path>/.dropoutt` | output directory |
| `--offline` | off | never touch the network; resolve the encoder from the cache |
| `--sample` | 200,000 | records to place, or the corpus if it is smaller |
| `--limit` | none | max records per file, for a fast look |
| `--no-html` | off | skip the HTML page |
| `--no-open` | off | do not open the page when the run finishes |
| `--no-evidence` | off | omit record excerpts and source locations from the terminal, `atlas.md`, `atlas.json` and `atlas.html` |
| `--workers`, `-j` | sized from the machine | processes used for the reading pass |
| `--quiet`, `-q` | off | suppress the map; only the output path is printed |

```bash
dropoutt atlas ./data
dropoutt atlas ./data --sample 20000       # a coarser map, sooner
dropoutt atlas ./data --offline            # encoder from the cache
dropoutt atlas ./data --no-evidence -q     # nothing quoted, nothing printed
```

`--sample` is the resolution knob. The default places up to 200,000 records,
which on a large corpus is the difference between a subject area decided by a
hundred records and one decided by thousands. Lower it for a first look; the
shape of the answer arrives long before the last digit of it does.

### What it reports

| section | what you learn |
| --- | --- |
| the map | every subject area, its subregions, and how densely you sit on each. Reach sums `min(1, density)` over subregions, so parity is a full score and over-representation does not count past one |
| what the map says | the handful of sentences that clear both a size gate and a significance gate. Nothing is shown for being true; it is shown for being large *and* true |
| what you have most of | the crowded places, each named by *your own record* nearest its centre — the only description of a neighbourhood that is true by construction |
| what you have least of | the sparsest places you reach. Reaching a place is not covering it, and an occupancy count cannot tell the difference |
| farthest from the map | where you are most over- or under-represented against the reference, and which direction to move |
| off the map | records unlike the reference geography, with a diagnosis. Similarity rises steeply with record length, so a high off-map rate is usually a statement about how short the records are before it is one about their subject |

Two checks run here and nowhere else: `T1-ATLAS-001`, when the corpus covers
very little of the map, and `T1-ATLAS-002`, when one crowded area holds
near-identical records — a template rather than a topic, which shingle-based
duplicate detection cannot see. Neither can fail a build. Concentration is
correct for a specialised dataset and wrong for a pretraining mixture, and the
tool has not been told which one you are building.

### What gets written

| file | what it is |
| --- | --- |
| `atlas.html` | the map, as one self-contained page that opens from `file://` |
| `atlas.md` | the same content as Markdown, to paste into a PR or a ticket |
| `atlas.json` | the same content as one JSON document, schema `dropoutt.atlas/1` |

They are named `atlas.*` rather than `report.*` so one folder can hold a scan and
a map without either overwriting the other.

### When no map can be drawn

Exit code 1, with the reason. Two causes account for almost all of it: the
encoder could not be loaded — run `dropoutt fetch` — or no record was long
enough to place, which needs at least 80 characters of text. A corpus of labels,
ids or one-word rows has no position on a topical map, and an empty page
presented as "no coverage" is the failure this stops.

A corpus whose records are all *far* from everything on the map is not that
case. It exits 0 and says so: that every record you have looks like nothing in
the reference geography is an answer, and often the most interesting one.

## `dropoutt checks [CHECK_ID]`

List the catalog, or explain one check.

```bash
dropoutt checks
dropoutt checks T0-MASK-001
```

The detail view prints the check's tier, cost, severity, confidence, the
profiles it applies to, its requirements, the profiles it blocks under, the fix,
and the reasoning behind it.

## `dropoutt benchmarks`

List the benchmark registry and which contamination indices are available.

Shows the **eval split that is actually scorable**, which is the field most
tools get wrong: HellaSwag, PIQA, WinoGrande and CommonsenseQA all have `test`
splits whose labels are blanked, IFEval's only split is `train`, and GPQA is
gated with no test split at all.

Each row is marked `built` when an index exists, `shippable` when one could be
distributed, or `not distributable` when the licence forbids it.

## `dropoutt models`

List known models with their chat-template family and licence, then the
shorthand aliases. The Turkish models are in the same table.

## `dropoutt fetch`

Pre-download everything a later `--offline` run needs.

| flag | default | meaning |
| --- | --- | --- |
| `--model`, `-m` | none | also fetch this model's tokenizer |
| `--all` | off | fetch the whole comparison panel, not just one tokenizer |

With no `--model` the comparison panel is fetched. With `--model` alone, only
that one tokenizer. With both, that model and the panel.

```bash
dropoutt fetch                       # the comparison panel
dropoutt fetch --model qwen3         # one tokenizer
dropoutt fetch --model qwen3 --all   # that model plus the panel
```

```
  cache: /scratch/$USER/dropoutt

  Tokenizers
    ok      qwen3  Qwen/Qwen3-8B

  Atlas embedding model
    ok  minishlab/potion-multilingual-128M (128 dims, stored int8, 81 MB)

  Bundled in the package, nothing to fetch: the atlas artifact and the
contamination indices.
  Now run scans with --offline. Keep DROPOUTT_CACHE set to this same path.
```

For each tokenizer it also fetches that model's `tokenizer_config.json`, which
carries the chat template. A tokenizer that loads but whose config declares no
chat template is reported as `partial` rather than `ok`, because an offline scan
against it would count raw text and skip the loss-mask checks.

The atlas artifact and the shipped contamination indices are inside the package
and are never downloaded. What this fetches is the atlas embedding model and
tokenizers, which are too large to vendor.

The command exits 1 when a tokenizer, chat template, or atlas embedding model
could not be prepared. This makes the login-node step usable as a batch
precondition instead of requiring someone to inspect colored output.

See [portability.md](portability.md) for the login-node to compute-node
workflow, and for which cache variable holds which file.

## `dropoutt doctor` — removed in 1.1

`doctor` printed a table of which optional components were installed and what
each missing one cost. There are no optional components any more: `pip install
dropoutt` brings all of them, so the table read "yes" on every row and the
command answered a question nobody had.

The part that stayed useful — *which* Python was probed, because the classic way
to be confused by a missing import is to have installed it with a `pip`
belonging to a different interpreter — moved to `dropoutt fetch`, along with the
machine a scan would size itself against:

```
  This environment
    python   /path/to/.venv/bin/python
    cache    ~/.cache/dropoutt
    version  1.1.0
    machine  13 workers (bound by cores) on 14 usable cores · 48 GiB RAM · no GPU
```

If a required package genuinely cannot be imported — a pruned container image, a
partially restored environment — `fetch` names it and prints the repair command
for this interpreter.

## Exit codes

| code | meaning |
| --- | --- |
| 0 | completed, findings or not |
| 1 | internal error |
| 2 | usage error |
| 10 | blocking findings under a declared target |

A checker that returns the same code for "found problems" and "crashed" cannot
be used in CI, which is why 1 is reserved for a genuine failure of the tool.

## Environment variables

| variable | meaning |
| --- | --- |
| `DROPOUTT_CACHE` | cache location; overrides `XDG_CACHE_HOME`. Holds the atlas embedding model and the cached `tokenizer_config.json` files. |
| `DROPOUTT_OFFLINE` | when set to `1`, `true`, `yes`, or `on`, makes `scan` resolve only from local files and caches |
| `DROPOUTT_OPEN` | `0` never opens the finished report; `1` always does, ignoring the SSH and headless checks |
| `DROPOUTT_WORKERS` | default number of processes for the streaming pass; `-j` overrides it for one run |
| `DROPOUTT_NO_GPU` | when truthy, skip accelerator detection entirely — for a node where `nvidia-smi` exists but hangs |
| `DROPOUTT_DEVICE` | force the device the atlas targets (`cpu`, `cuda`, `mps`); overrides detection |
| `DROPOUTT_DEBUG` | when truthy, show a traceback for an internal error instead of the concise exit-1 message |
| `XDG_CACHE_HOME` | standard cache root; used as `$XDG_CACHE_HOME/dropoutt` when `DROPOUTT_CACHE` is unset |
| `HF_HOME` | Hugging Face cache root. The tokenizers themselves are cached here, not under `DROPOUTT_CACHE`. |
| `HF_HUB_OFFLINE` | honored as an offline request; also set internally for the duration of an offline tokenizer load |

`HF_HOME` matters on a cluster: `DROPOUTT_CACHE` alone is not enough to make an
offline run find its tokenizer. See [portability.md](portability.md).
