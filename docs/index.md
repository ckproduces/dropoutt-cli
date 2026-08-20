# Quickstart

```bash
pip install dropoutt
dropoutt scan ./data
```

That is the whole first run. No model, no config, no flags.

For the longer version — installing, reading the output finding by finding, and
blocking a bad run in CI — read [getting-started.md](getting-started.md). This
page is the summary.

## What a zero-configuration run produces

```
──────────────────────────── dropoutt scan ─────────────────────────────
  Discovered  6 files, 5 datasets, 0.1 MB
  Formats     jsonl 6
  Empty files 1

  Schema induction
    chatml                 1 dataset(s)
    alpaca                 1 dataset(s)
    sharegpt               1 dataset(s)
    text                   1 dataset(s)

  Not training data
    logs                     records carry type='queue-operation'

  Best guess at what you are building
    Stage      sft
    Language   tr 97%, unknown 2%, en 0%
    Confidence: medium. Set it explicitly in dropoutt.toml if this is wrong.

  Findings
  T0-ROLE-002    ●   60   non-canonical role names: 'human' (60), 'gpt' (60)
  T0-SCHEMA-002  ●    1   alpaca_mix: alpaca 91%, prompt_completion 9%
  T1-PII-001     ●    1   Email address (1), OpenAI API key (1)
  T1-OVERLAP-001 ●    2   17% of 'alpaca_mix' records also appear in 'good_chat'
  ...

  Token budget (estimated, no --model given)
    Llama-3.1      ~ 0.02M tokens   1.75 tok/word
    Qwen3          ~ 0.02M tokens   2.14 tok/word  (+22%)

  Not checked, and why
    Records contribute zero trainable tokens    needs the target model's template
      → pass --model
    Training data overlaps evaluation benchmarks  no benchmark indices found
      → reinstall dropoutt; the benchmark indices ship in the package

  No blocking verdict issued: no target declared.
```

Four things are doing the work there.

**Schema induction, not schema configuration.** Records are sampled and matched
against known layouts. The *mixture* is itself a finding: three layouts in one
folder is almost always a collection bug, and it will silently break a
preparation script written for one of them.

**"Not training data" detection.** Agent session logs and telemetry traces get
detected as what they are rather than forced into a chat layout.

**The missing model becomes a feature.** With no `--model`, refusing to count
tokens would be correct and useless. Counting under several tokenizers and
showing the spread answers a question you did not know to ask — and for Turkish
the spread is large enough to matter to a real decision.

**"Not checked, and why" is a capability statement**, not an apology. Each line
names the single flag that would unlock it.

## Unlocking more

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096
```

Adds exact token counts, fertility, the truncation forecast, chat-template
rendering, loss-mask validation and packing efficiency.

```bash
dropoutt scan ./data --model qwen3 --target sft
```

Adds contamination against *your* held-out set. The index stores hashed 8-grams,
not raw records, and is built locally. Keep it inside the held-out set's trust
boundary because unkeyed hashes can be tested against candidate phrases.

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft
```

Enables blocking. Exit code 10 on blocking findings.

## Outputs

Every scan writes four files to `.dropoutt/`:

| file | what it is |
| --- | --- |
| `report.html` | one self-contained file, no server, no CDN. Contains excerpts and paths unless the scan used `--no-evidence`. |
| `report.md` | the same content as text, for a pull-request comment or a CI log |
| `report.json` | the same content as one JSON document, for a dashboard or a coverage gate |
| `fingerprint.json` | the comparable description of your dataset |
| `findings.jsonl` | one record per finding, for scripting |

Use `--no-evidence` before exporting reports from the dataset trust boundary.
Aggregate metadata, dataset names, the scan root, and hashes still remain.

## Next

- [cli.md](cli.md) — every command
- [checks.md](checks.md) — the catalog
- [configuration.md](configuration.md) — `dropoutt.toml`
- [portability.md](portability.md) — clusters and offline use
- [design.md](design.md) — why the rules are what they are
- [limitations.md](limitations.md) — what this release does not do
