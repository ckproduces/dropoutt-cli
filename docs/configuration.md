# Configuration

The principle is: infer first, confirm once, keep it in version control. Nobody
completes a preferences form, so the tool does not ask for one.

## What is inferred

| setting | inferred from |
| --- | --- |
| chat template, special tokens, EOS, BOS | the target model's `tokenizer_config.json`. The model card *is* the configuration. |
| sequence length | the model config's `model_max_length` |
| record layout | sniffed from the data and matched against a library of known layouts |
| language mix | measured from the data, then proposed as a target |
| training stage | inferred from the layouts: preference triples imply DPO, bare text implies continued pretraining, message lists imply SFT |
| loss-masking policy | **not** inferable from data. It is a training-script decision. |

## `dropoutt.toml`

Lives beside your data-preparation code so it is reviewed in pull requests and
shared across the team, rather than living in one person's shell history. This
is the pattern ruff, eslint and dbt use.

```toml
# dropoutt configuration.
# Every key is optional. A flag on the command line always wins over the file.

[scan]
model = "Qwen/Qwen3-8B"       # resolved from the Hub
profile = "sft"               # inferred from 83 dataset(s)
# target = "sft"              # uncomment to let findings fail the run
seq_len = 4096                # from the model config
tier = 1
minhash_preset = "fineweb"
# offline = true
# eval_sets = ["gsm8k", "internal-eval"]

[mute]
# Check ids to silence, with a reason. Muting is a decision worth reviewing.
checks = []
```

### Fields

| field | meaning |
| --- | --- |
| `model` | Hub id, local path, or a shorthand alias such as `qwen3` |
| `profile` | `sft`, `corpus`, `preference`, or `auto` |
| `target` | **declaring this enables blocking.** Absent means nothing can fail a run. |
| `seq_len` | training sequence length, for the truncation forecast |
| `tier` | highest check tier to run |
| `minhash_preset` | `fineweb` or `hf-neardedup`; the report always states which |
| `offline` | never access the network during `scan` or `atlas`; resolve models and the atlas encoder from local files and caches |
| `eval_sets` | optional allowlist of bundled or locally indexed benchmark names; absent means use every available index |
| `mute.checks` | check ids to silence |

Command-line flags override the file.

## Aliases

```bash
dropoutt scan ./data --model qwen3      # -> Qwen/Qwen3-8B
dropoutt scan ./data --model trendyol   # -> Trendyol/Trendyol-LLM-8B-T1
dropoutt scan ./data --model kumru      # -> vngrs-ai/Kumru-2B
```

`dropoutt models` lists everything known, including the verified Turkish set.

## Declaring a target

Without `target`, findings read "would block under sft" and the exit code stays
0. With it, blocking findings return exit code 10.

This is deliberate. Blocking asserts that something is wrong *for a goal*, and
without a declared goal there is no basis for the assertion. A checker that
fails your build on an assumption it made by itself gets removed once and never
comes back.
