# Command reference

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
| `--offline` | off | never touch the network |
| `--limit` | none | max records per file, for a fast look |
| `--no-html` | off | skip the HTML report |
| `--quiet`, `-q` | off | suppress the terminal report |

```bash
dropoutt scan ./data
dropoutt scan ./data --model qwen3 --seq-len 4096
dropoutt scan ./data --model /scratch/models/qwen3-4b --offline
dropoutt scan ./data --target sft --quiet          # CI gate
dropoutt scan ./data --limit 1000                  # quick look at a huge corpus
```

## `dropoutt init [PATH]`

Infer configuration and write `dropoutt.toml`.

| flag | meaning |
| --- | --- |
| `--model`, `-m` | resolve a model and show the template confirmation |
| `--force` | overwrite an existing config |

With `--model` it renders two records and prints the exact trainable span, so a
template or masking mismatch shows up immediately rather than after a run.

## `dropoutt index-eval PATH --name NAME`

Build a contamination index from your own evaluation set.

| flag | default | meaning |
| --- | --- | --- |
| `--name`, `-n` | required | name for this benchmark |
| `--field`, `-f` | `text` | which field holds the text; falls back to joining all string fields |

```bash
dropoutt index-eval ./holdout.jsonl --name internal-eval --field question
```

The index stores hashed 8-grams and instance sizes. The text is not stored and
cannot be recovered, so the index is safe to keep next to your data or commit.

## `dropoutt checks [CHECK_ID]`

List the catalog, or explain one check.

```bash
dropoutt checks
dropoutt checks T0-MASK-001
```

The detail view prints the check's tier, cost, severity, requirements, the
profiles it blocks under, the fix, and the reasoning behind it.

## `dropoutt benchmarks`

List the benchmark registry and which contamination indices are available.

Shows the **eval split that is actually scorable**, which is the field most
tools get wrong: HellaSwag, PIQA, WinoGrande and CommonsenseQA all have `test`
splits whose labels are blanked, IFEval's only split is `train`, and GPQA is
gated with no test split at all.

## `dropoutt models`

List known models, their chat-template family, and shorthand aliases. Includes
the verified Turkish set.

## `dropoutt doctor`

Show what is installed and what each missing component costs.

```
component             status   without it                                install
orjson                yes      speed only
tokenizers            yes      exact token counts, template, loss mask
fasttext-langdetect   no       accurate language identification          pip install 'dropoutt[lid]'
model2vec             no       atlas coverage                            pip install 'dropoutt[atlas]'
```

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
| `DROPOUTT_CACHE` | cache location; overrides `XDG_CACHE_HOME` |
| `XDG_CACHE_HOME` | standard cache root |
| `HF_HUB_OFFLINE` | respected by the Hub client |
