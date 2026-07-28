# Command reference

Every command below also works as `python -m dropoutt <command>`. The console
script lands in the environment's `bin/`, which module systems and batch
schedulers often leave off `PATH`; the module form works from any interpreter
that can import the package.

`dropoutt` with no command prints the help. `dropoutt --version` prints the
version and exits. Those are the only top-level options.

## `dropoutt scan PATH`

Scan a file or directory.

| flag | default | meaning |
| --- | --- | --- |
| `--model`, `-m` | none | target model id, local path, or alias. Unlocks the token-dependent checks. |
| `--profile`, `-p` | `auto` | `sft`, `corpus`, `preference`, or `auto` |
| `--target` | none | declare what you are building. **This is what enables blocking.** |
| `--seq-len` | from model config | training sequence length |
| `--tier` | `1` | highest check tier to run |
| `--out`, `-o` | `<path>/.dropoutt` | output directory |
| `--offline` | off | never touch the network; resolve the model from the cache |
| `--limit` | none | max records per file, for a fast look |
| `--no-html` | off | skip the HTML report |
| `--no-atlas` | off | skip atlas coverage |
| `--quiet`, `-q` | off | suppress the terminal report; only the output path is printed |

```bash
dropoutt scan ./data
dropoutt scan ./data --model qwen3 --seq-len 4096
dropoutt scan ./data --model /scratch/models/qwen3-4b --offline
dropoutt scan ./data --target sft --quiet          # CI gate
dropoutt scan ./data --limit 1000                  # quick look at a huge corpus
```

`--model`, `--profile`, `--target`, `--seq-len` and `--offline` fall back to
`dropoutt.toml` when not given on the command line, and `--seq-len` falls back
again to the model's own `model_max_length`.

`--offline` resolves the tokenizer and the chat template from the cache rather
than refusing to load them, so an offline scan reports the same numbers as an
online one. Populate the cache first with `dropoutt fetch`. If no cached chat
template is found, the run says so before the report rather than counting raw
text silently.

## `dropoutt init [PATH]`

Infer configuration and write `dropoutt.toml`. `PATH` defaults to `.`.

| flag | meaning |
| --- | --- |
| `--model`, `-m` | resolve a model and show the template confirmation |
| `--force` | overwrite an existing config |

With `--model` it renders a two-turn probe conversation through the model's chat
template and prints the exact trainable span, so a template or masking mismatch
shows up immediately rather than after a run. `init` has no `--offline` flag and
resolves the model against the Hub.

## `dropoutt index-eval PATH --name NAME`

Build a contamination index from your own evaluation set.

| flag | default | meaning |
| --- | --- | --- |
| `--name`, `-n` | required | name for this benchmark |
| `--field`, `-f` | `text` | which field holds the text; falls back to joining all string fields |

```bash
dropoutt index-eval ./holdout.jsonl --name internal-eval --field question
```

The index stores hashed 8-grams and the per-instance gram count used as the
coverage denominator. The text is not stored and cannot be recovered, so the
index is safe to keep next to your data or commit.

The output path is not configurable. The command prints where it wrote the
index, which on a fresh machine is the installed package's own
`data/contamination` directory rather than `DROPOUTT_CACHE`. See
[portability.md](portability.md).

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
  version: 0.1.0
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
| `XDG_CACHE_HOME` | standard cache root; used as `$XDG_CACHE_HOME/dropoutt` when `DROPOUTT_CACHE` is unset |
| `HF_HOME` | Hugging Face cache root. The tokenizers themselves are cached here, not under `DROPOUTT_CACHE`. |
| `HF_HUB_OFFLINE` | set to `1` internally for the duration of an offline tokenizer load, so the Hub client resolves from the cache and raises instead of connecting |

`HF_HOME` matters on a cluster: `DROPOUTT_CACHE` alone is not enough to make an
offline run find its tokenizer. See [portability.md](portability.md).
