# Command reference

Every command below also works as `python -m dropoutt <command>`. The console
script lands in the environment's `bin/`, which module systems and batch
schedulers often leave off `PATH`; the module form works from any interpreter
that can import the package.

`dropoutt` with no command prints the mark and the help. `scan`, which takes a
required argument, prints its complete help and a set of examples when run with
no arguments.

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
inside a major version.

## `dropoutt scan PATH`

Scan a file or directory.

The CLI shows the active phase while it discovers files, infers layouts, scans
records, maps atlas coverage, finalizes checks, and writes artifacts. In a
terminal this is a spinner; redirected batch output receives stable phase lines
and periodic record counts instead of control sequences.

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
| `--no-atlas` | off | skip atlas coverage |
| `--no-evidence` | off | omit record excerpts and source locations from terminal output, `findings.jsonl`, `report.md`, and `report.html` |
| `--workers`, `-j` | one per core, less one | processes used for the streaming pass |
| `--quiet`, `-q` | off | suppress the terminal report; only the output path is printed |

```bash
dropoutt scan ./data
dropoutt scan ./data --model qwen3 --seq-len 4096
dropoutt scan ./data --model /scratch/models/qwen3-4b --offline
dropoutt scan ./data --target sft --quiet          # CI gate
dropoutt scan ./data --limit 1000                  # quick look at a huge corpus
dropoutt scan ./data --offline --no-evidence       # VPC-safe network and evidence settings
dropoutt scan ./data -j 4                          # cap the scan at four processes
```

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

Atlas coverage appears in the terminal report and in `report.html`: how many
records were placed and out of how many, the regions occupied, the spread as a
share of even coverage, the top categories and the top regions with their terms.
Every share is over the placed records, and the placed count is printed next to
it. The HTML report also plots occupied regions on the atlas's frozen 2D
coordinates; circle size encodes sampled record count. The level-0 probe accuracy
is printed underneath it.

Records that did not place are described rather than merely counted: the most
likely reason, the similarity distribution against the cutoff, the regions they
were nearest to anyway, the rate broken down by language and by dataset, and the
few furthest excerpts. Similarity to a region rises steeply with record length, so
a high off-atlas rate is usually a statement about how short the records are
before it is one about their subject; the reported reason says which. See
[atlas.md](atlas.md#off-atlas-data).

`--no-atlas` skips the assignment step and the block with it. When coverage is
computed, the full region histogram also goes into `fingerprint.json`, which is
comparable across machines. Record excerpts stay out of `fingerprint.json`, which
is the artifact meant to be shareable.

By default, terminal examples, `findings.jsonl`, `report.md`, and `report.html` contain
bounded record excerpts and source locations so findings can be inspected.
Detected PII is masked, but arbitrary proprietary text is not. Use
`--no-evidence` before moving these artifacts outside the dataset's trust
boundary. The fingerprint never contains record excerpts, but it does contain
the scan root, dataset names, aggregate measurements, and stable hashes.

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
    ok  minishlab/potion-multilingual-128M (256 dims)

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

## `dropoutt doctor`

Show what is installed and what each missing component costs.

```
  component              status    without it                  install
  orjson                 yes       speed only
  tokenizers             yes       exact token counts, chat
                                   template render, loss
                                   mask checks
  pyarrow                yes       reading .parquet files
  rensa                  no        speed only, identical       pip install
                                   clusters                    'dropoutt[fast]'
  fasttext-langdetect    yes       accurate language
                                   identification across
                                   176 languages
  model2vec              yes       atlas coverage

  cache: ~/.cache/dropoutt
  version: 0.1.3
```

The last two lines are the resolved cache directory and the installed version,
which is what you want when a scan on a cluster behaves differently from the
same scan on your laptop.

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
| `DROPOUTT_DEBUG` | when truthy, show a traceback for an internal error instead of the concise exit-1 message |
| `XDG_CACHE_HOME` | standard cache root; used as `$XDG_CACHE_HOME/dropoutt` when `DROPOUTT_CACHE` is unset |
| `HF_HOME` | Hugging Face cache root. The tokenizers themselves are cached here, not under `DROPOUTT_CACHE`. |
| `HF_HUB_OFFLINE` | honored as an offline request; also set internally for the duration of an offline tokenizer load |

`HF_HOME` matters on a cluster: `DROPOUTT_CACHE` alone is not enough to make an
offline run find its tokenizer. See [portability.md](portability.md).
