# Command reference

Every command below also works as `python -m dropoutt <command>`. The console
script lands in the environment's `bin/`, which module systems and batch
schedulers often leave off `PATH`; the module form works from any interpreter
that can import the package.

`dropoutt` with no command prints the help. Commands with required arguments,
such as `scan`, `diff`, and `index-eval`, also print their complete help and an
example when run with no arguments. `dropoutt --version` prints the version and
exits. Those are the only top-level options.

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
| `--no-atlas` | off | skip atlas coverage |
| `--no-evidence` | off | omit record excerpts and source locations from terminal output, `findings.jsonl`, and `report.html` |
| `--quiet`, `-q` | off | suppress the terminal report; only the output path is printed |

```bash
dropoutt scan ./data
dropoutt scan ./data --model qwen3 --seq-len 4096
dropoutt scan ./data --model /scratch/models/qwen3-4b --offline
dropoutt scan ./data --target sft --quiet          # CI gate
dropoutt scan ./data --limit 1000                  # quick look at a huge corpus
dropoutt scan ./data --offline --no-evidence       # VPC-safe network and evidence settings
```

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
what `dropoutt diff` reads. Record excerpts stay out of `fingerprint.json`, which
is the artifact meant to be shareable.

By default, terminal examples, `findings.jsonl`, and `report.html` contain
bounded record excerpts and source locations so findings can be inspected.
Detected PII is masked, but arbitrary proprietary text is not. Use
`--no-evidence` before moving these artifacts outside the dataset's trust
boundary. The fingerprint never contains record excerpts, but it does contain
the scan root, dataset names, aggregate measurements, and stable hashes.

## `dropoutt diff LEFT RIGHT`

Compare two fingerprints against the shared atlas.

| flag | default | meaning |
| --- | --- | --- |
| `--full` | off | list every differing region instead of truncating to the top few |

Both arguments are `fingerprint.json` files written by `dropoutt scan`. The
bundled `atlas-lite-v0.npz` is not an operand. If a binary atlas artifact,
unrelated JSON file, or malformed fingerprint is passed, the command identifies
the mismatch and prints the correct invocation.

The comparison is directional and reads left against right: **what does LEFT
cover that RIGHT does not**. It is not symmetric. A small specialised corpus can
sit wholly inside a large one while the large one is barely inside it, so run it
with the dataset you are considering on the left and the mixture you already
have on the right.

```bash
dropoutt diff ./candidate/.dropoutt/fingerprint.json ./mixture/.dropoutt/fingerprint.json
dropoutt diff ./candidate/.dropoutt/fingerprint.json ./mixture/.dropoutt/fingerprint.json --full
```

```
──────────────────────────────────── dropoutt diff ─────────────────────────────────────
  left  /private/tmp/fp-code
  right /private/tmp/fp-turkish_instructions

  Shape
                 left      right
records         1,500      1,500
datasets            1          1
characters    908,255    568,510

  Atlas comparison
    Similarity   0.02  (1.0 = same distribution over regions)
    Shared       38% of left sits in regions right also occupies
    New          62% of left sits in regions right never reaches

    Only in left — what adding it would bring
      151    12%  import, python, return, data, create
      157    11%  return, function, list, write, given
      149     9%  return, function, write, given, else
      153     8%  list, return, python, function, write
      147     5%  class, return, name, initself, python
      155     5%  import, model, data, python, create
      156     4%  random, import, password, python, generate
      158     2%  import, return, none, class, assert
      and 16 more; --full to list them

    Only in right
       94     5%  yardımcı, nasıl, şekilde, olabilir, sahip
       98     4%  makine, algoritma, öğrenimi, oluşturun, etmek
       89     4%  olduğunu, büyük, film, ortaya, şekilde
        7     4%  cümle, cümlenin, cümleyi, adım, doğru
       92     3%  uygulama, yazılım, kullanıcı, windows, şekilde

    Category mix
category            left    right    delta
code_generation      94%       0%     +94%
general_chat          3%      95%     -92%
turkish_culture       0%       3%      -3%
code_explanation      3%       0%      +3%
  This is geometry, not a recommendation. Whether new coverage helps depends on what you
are training.
```

The Shape table is read straight from the two fingerprints: records, datasets,
characters, and the near- and exact-duplicate rates when both sides recorded
them.

Similarity is the cosine of the two region-mass vectors over their union, so
1.0 means the same distribution across regions and not the same records.
**Shared** is the share of left's placed mass sitting in regions right also
occupies; **New** is the share sitting in regions right never reaches at all.
The two region lists are shares of their own side's placed mass, truncated to
the top 8 and top 5 unless `--full` is given. The category mix table lists only
categories that differ by at least 2 points, again capped at 8 rows without
`--full`.

If the two fingerprints were written by different pipeline versions the run says
so and continues.

When either side placed less than 90% of its records, the run prints the placed
shares under the headline numbers and states which way the bias runs. A partial
**right** side means regions it appears not to reach may be reached by records it
could not place, so `New` is an upper bound. A partial **left** side means the
shares describe only the part that placed. Both are reported; neither stops the
comparison.

The comparison refuses only when it would have to invent an answer:

```
  Atlas comparison
    not comparable left side: coverage status is 'not computed (no atlas available)'
```

That happens when a side's coverage was never computed, when the two fingerprints
were built against different atlas versions, where the region ids do not refer to
the same regions, and when a fingerprint predates 0.1.4 and had its histogram
discarded by the old suppression rule — re-scanning fixes that one. The exit code
is 0 in every case above, including this one. A missing or unreadable fingerprint
file is a usage error and exits 2.

## `dropoutt init [PATH]`

Infer configuration and write `dropoutt.toml`. `PATH` defaults to `.`.

| flag | meaning |
| --- | --- |
| `--model`, `-m` | resolve a model and show the template confirmation |
| `--offline` | resolve model metadata only from local files and caches |
| `--force` | overwrite an existing config |

With `--model` it renders a two-turn probe conversation through the model's chat
template and prints the exact trainable span, so a template or masking mismatch
shows up immediately rather than after a run.

## `dropoutt index-eval PATH --name NAME`

Build a contamination index from your own evaluation set.

| flag | default | meaning |
| --- | --- | --- |
| `--name`, `-n` | required | name for this benchmark |
| `--field`, `-f` | `text` | which field holds the text; falls back to joining all string fields |
| `--force` | off | overwrite an existing private index with the same name |

```bash
dropoutt index-eval ./holdout.jsonl --name internal-eval --field question
```

The index stores hashed 8-grams and the per-instance gram count used as the
coverage denominator rather than raw text. It is not a disclosure-proof
artifact: because hashes are unkeyed, someone with candidate phrases can test
them against the index. Keep private indices inside the evaluation set's trust
boundary.

The output path is `$DROPOUTT_CACHE/contamination`, or the resolved fallback
cache when that variable is unset. The command prints the exact path. Bundled
and private indices are both searched on later scans. Names must be filename-safe
and an existing index is not overwritten without `--force`.

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

## `dropoutt atlas`

Show the bundled atlas and its own quality numbers: artifact hash, embedding
model, region count, reference-record count, the off-atlas cosine cutoff, the
level-0 held-out accuracy and the region purity, how many build sources were
unavailable, and the largest regions with their terms.

The accuracy and purity figures are printed next to any coverage number for a
reason: an atlas whose level-0 probe is weak produces coverage histograms that
look precise and are not.

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
| `DROPOUTT_OFFLINE` | when set to `1`, `true`, `yes`, or `on`, makes `scan` and `init` resolve only from local files and caches |
| `DROPOUTT_DEBUG` | when truthy, show a traceback for an internal error instead of the concise exit-1 message |
| `XDG_CACHE_HOME` | standard cache root; used as `$XDG_CACHE_HOME/dropoutt` when `DROPOUTT_CACHE` is unset |
| `HF_HOME` | Hugging Face cache root. The tokenizers themselves are cached here, not under `DROPOUTT_CACHE`. |
| `HF_HUB_OFFLINE` | honored as an offline request; also set internally for the duration of an offline tokenizer load |

`HF_HOME` matters on a cluster: `DROPOUTT_CACHE` alone is not enough to make an
offline run find its tokenizer. See [portability.md](portability.md).
