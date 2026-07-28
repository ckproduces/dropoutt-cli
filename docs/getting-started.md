# Getting started

A walkthrough from nothing installed to a scan you can act on. Every command
here was run against the real package and every block of output is real output,
not an illustration.

If you read one section, read [Reading the output](#reading-the-output).

---

## 1. Install

`dropoutt` is not on PyPI yet, so install it from the checkout.

```bash
cd ~/Documents/dropoutt-cli
python3 -m venv .venv
.venv/bin/pip install -e '.[all]'
```

`-e` is an editable install: the `dropoutt` command points back at the source
tree, so edits take effect without reinstalling. Drop it for a normal install.

Check it worked:

```bash
.venv/bin/dropoutt --version
```

```
0.1.3
```

Either activate the environment or use the full path. The rest of this document
writes `dropoutt`, assuming you ran `source .venv/bin/activate`.

### If `dropoutt` is not found

The console script lives in `.venv/bin/`, which module systems and batch
schedulers often strip from `PATH`. This always works instead:

```bash
python -m dropoutt --version
```

### What the extras cost you

`[all]` pulls everything. On a constrained machine, install the core and add
what you need:

| install | size | what you lose without it |
| --- | --- | --- |
| `pip install -e .` | small | nothing structural — all Tier 0 checks run |
| `'.[tokenizer]'` | moderate | exact token counts, chat-template rendering, loss-mask checks, packing |
| `'.[lid]'` | 938 KB model | accurate language ID across 176 languages |
| `'.[atlas]'` | ~500 MB model on first use | atlas coverage |
| `'.[parquet]'` | 35–50 MB | reading `.parquet`, `.arrow`, `.feather`, and `.orc` files |
| `'.[fast]'` | small | speed only, identical results |

Ask the tool rather than guessing:

```bash
dropoutt doctor
```

```
  component              status    without it                  install
  orjson                 yes       speed only
  tokenizers             yes       exact token counts, chat
                                   template render, loss
                                   mask checks
  pyarrow                yes       reading Parquet, Arrow, Feather, and ORC
  rensa                  no        speed only, identical       pip install
                                   clusters                    'dropoutt[fast]'
  fasttext-langdetect    yes       accurate language
                                   identification across
                                   176 languages
  model2vec              yes       atlas coverage

  cache: /Users/you/.cache/dropoutt
  version: 0.1.3
```

Nothing here is required. A missing component removes checks; it never produces
a wrong answer silently. Whatever is skipped is listed at the end of every scan
with the flag that unlocks it.

---

## 2. First run

Point it at a folder. No flags, no config, no model.

```bash
dropoutt scan ./data
```

Try it on the bundled fixture, which is deliberately broken in nine different
ways:

```bash
dropoutt scan tests/fixtures/messy
```

Real output, trimmed in the middle:

```
──────────────────────────── dropoutt scan ─────────────────────────────
  Discovered  6 files, 5 datasets, 0.1 MB
  Formats     jsonl 6
  Empty files 1

  Schema induction
    alpaca                 1 dataset(s)
    chatml                 1 dataset(s)
    text                   1 dataset(s)
    sharegpt               1 dataset(s)

  Not training data
    logs                          records carry type='queue-operation'

  Best guess at what you are building
    Stage      sft
    Language   tr 97%, unknown 2%, en 0%
    Confidence: medium. Confirm with `dropoutt init`.

  Findings
check                  count    detail
T0-ROLE-001       ●        9    1 with no assistant turn; 1 with consecutive
                                same-role turns; 7 with an empty message; 1
                                ending on a user turn  would block under sft
T0-ROLE-002       ●       60    non-canonical role names: 'human' (60),
                                'gpt' (60)  would block under sft
T0-SCHEMA-001     ●        1    logs: records carry type='queue-operation'
                                would block under sft, corpus, preference
T1-IDENT-001      ●        1    1 records claim to be another vendor's
                                assistant  would block under sft
T1-PII-001        ●        1    Email address (1), OpenAI API key (1)
                                would block under sft, corpus, preference
T0-DEGEN-001      ●      133    2 trivial, 131 looping responses
T0-DUP-001        ●        6    6 redundant copies across 1 exact clusters
                                (largest cluster 7 copies)
T0-SCHEMA-002     ●        1    alpaca_mix: alpaca 91%, prompt_completion 9%
T0-SCHEMA-003     ●        1    1 of 332 records could not be parsed as JSON
T1-LANG-004       ●        1    1 records (0.3%) read as Turkish but contain
                                none of the Turkish-specific characters
T1-NDUP-001       ●       23    23 redundant records across 11 clusters at
                                Jaccard >= 0.75 (largest cluster 8)
T1-OVERLAP-001    ●        2    17% of 'alpaca_mix' records (10 of 60) also
                                appear in 'good_chat'

  Token budget (estimated, no --model given)
    Llama-3.1      ~    0.02M tokens   1.75 tok/word
    Gemma-2        ~    0.02M tokens   1.78 tok/word  (+2%)
    Qwen3          ~    0.02M tokens   2.14 tok/word  (+22%)
    SmolLM2        ~    0.03M tokens   3.11 tok/word  (+78%)
    Mistral-v0.3   ~    0.03M tokens   3.11 tok/word  (+78%)

  Not checked, and why
    Records contribute zero trainable tokens    needs the target model's
                                                chat template
      → pass --model
    Packing efficiency under concat-and-chunk   needs a tokenizer
      → pass --model, e.g. --model Qwen/Qwen3-8B

  All findings in this build are unverified: no measured effect size is
  attached to acting on them. They are structural observations, not
  predictions.
  No blocking verdict issued: no target declared.
  332 records in 1.6s

  wrote tests/fixtures/messy/.dropoutt/fingerprint.json, findings.jsonl,
  report.html
```

Scanning writes a `.dropoutt/` folder next to your data. Use `--out` to put it
elsewhere.

---

## 3. Reading the output

The screen is one argument, top to bottom.

### Discovered

What was found on disk. If the file count is wrong, the rest is wrong — check
your path before reading further.

### Schema induction

Records are sampled and matched against known layouts. Nothing was configured;
this was inferred.

**The mixture is itself the finding.** Four layouts in one folder is almost
always a collection bug. A preparation script written for one of them will
silently mishandle the rest.

### Not training data

Agent session logs, telemetry, and rollout traces are structurally close enough
to chat data that importers ingest them without complaint. They are detected and
excluded from content checks, because forcing a log record through a chat layout
produces confident nonsense.

### Best guess at what you are building

A hypothesis, labelled as one. `dropoutt init` writes it down so you can correct
it.

### Findings

One row per check that fired. Read three columns:

- **the dot** — red blocking, yellow warning, cyan informational
- **count** — how many records, or `-` for a whole-dataset observation
- **detail** — the measurement, plus `would block under sft` if this would fail
  a run once you declare a target

Rows are sorted by severity, not by count. 133 degenerate responses sits below a
single leaked API key on purpose.

### Token budget

With no `--model`, refusing to count tokens would be correct and useless.
Counting under several tokenizers and showing the spread answers a question you
did not think to ask.

Look at the fixture: the same Turkish text is **78% more tokens** under Mistral
than under Llama-3.1. That is a real difference in what a run costs, and for
non-English corpora it is routinely this large.

### Atlas coverage

Where your records land on `atlas-lite-v0`, a shared coordinate system. Covered
in [section 7](#7-comparing-two-datasets).

### Not checked, and why

Not an apology — a capability statement. Every line names the one flag that
unlocks it. This is the map of what you get next.

---

## 4. Unlocking more

Four steps, each adding checks. Stop wherever the answers stop being worth the
setup.

### Step 1 — name the model

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096
```

`--model` takes a Hugging Face id, a local path, or a shorthand from
`dropoutt models`. This unlocks the checks that need a tokenizer and the model's
own chat template:

```
T0-MASK-001       ●        8    8 records (2.7%) have an entirely masked label
                                vector and will train nothing
                                would block under sft
T0-PACK-001       ●        -    22,805 tokens fill 5 blocks of 4096, 2,325
                                tokens land in a residual tail that
                                concat-and-chunk pipelines usually discard;
                                55.2% of all tokens are trainable
```

`T0-MASK-001` is the one to care about. Eight records will occupy space in the
packed blocks and contribute nothing to any gradient. Nothing in a training log
would ever tell you.

### Step 2 — check against your own eval set

The package ships hashed 8-gram indices for ten permissively licensed
benchmarks. Your own held-out set matters more:

```bash
dropoutt index-eval ./my_holdout.jsonl --name my-eval --field question
dropoutt scan ./data --model qwen3
```

The index stores hashed 8-grams rather than raw text. Keep it beside the held-out
data: unkeyed hashes can still be tested against candidate phrases, so the index
is not safe to publish.

Contamination uses the Tülu 3 rule: an eval instance counts as contaminated when
more than 50% of its tokens are covered by 8-gram matches against a single
training instance; the training set counts as contaminated when more than 2% of
an eval's instances match.

Expect removing contamination to make your reported score go **down**. That is
the point.

### Step 3 — write the config down

```bash
dropoutt init ./data
```

```
  Wrote ./data/dropoutt.toml
  Detected profile: sft from 5 dataset(s)
  Edit the file to declare a target if you want findings to block.
```

```toml
[scan]
# model = "Qwen/Qwen3-8B"
profile = "sft"  # inferred from 5 dataset(s)
# target = "sft"  # uncomment to let findings fail the run
tier = 1
minhash_preset = "fineweb"

[mute]
# Check ids to silence, with a reason. Muting is a decision worth reviewing.
checks = []
```

Everything is inferred and every value is editable. See
[configuration.md](configuration.md).

### Step 4 — declare a target and let it block

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft
```

```
  6 blocking finding(s) under target profile sft
```

```bash
echo $?     # 10
```

**Nothing blocks until you declare a target.** Blocking means asserting that
something is wrong *for a goal*, and without a stated goal the tool has no
standing to fail your run. Until then, findings read `would block under sft` and
the exit code stays 0.

---

## 5. Using it in CI

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft --quiet
```

| code | meaning |
| --- | --- |
| 0 | completed, findings or not |
| 1 | internal error |
| 2 | usage error |
| 10 | blocking findings, and only when a target was declared |

10 is separate from 1 so that "your data has problems" and "the tool crashed"
are distinguishable. A checker that returns the same code for both cannot be
trusted in a pipeline.

```yaml
- name: Check training data
  run: dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft --quiet
```

---

## 6. The three output files

Every scan writes to `.dropoutt/`:

| file | what it is |
| --- | --- |
| `report.html` | one self-contained file — no server, no CDN. Contains excerpts and paths unless the scan used `--no-evidence`. |
| `fingerprint.json` | the comparable description of the dataset |
| `findings.jsonl` | one JSON object per finding, for scripting |

**`report.html` may contain your data.** PII values are masked before they reach
it — there is a test that fails if a planted secret appears in a generated
report — but excerpts of your text are in there. Treat it as you would treat the
dataset.

Scripting against findings:

```bash
jq -r 'select(.severity=="blocking") | "\(.check_id)\t\(.count)\t\(.detail)"' \
  .dropoutt/findings.jsonl
```

The fingerprint groups measurements into facets, each carrying an evidence
grade so you know how much weight it holds:

```json
{
  "fingerprint_id": "fp_0a64b170ccb89872fac5606475b3a356",
  "schema_version": "fp-v0.1",
  "facets": {
    "shape": {
      "values": { "records": 332, "total_chars": 54577, "total_words": 8725 },
      "evidence_grade": "descriptive"
    },
    "redundancy": {
      "values": { "near_duplicate_rate": 0.07165, "largest_cluster": 8 },
      "evidence_grade": "conditional"
    }
  }
}
```

`evidence_grade` is the honest part. `descriptive` means the number has no good
or bad direction. `conditional` means acting on it helps only under conditions
the tool cannot verify for you. Full schema in
[fingerprint.md](fingerprint.md).

---

## 7. Comparing two datasets

Every scan places your records on `atlas-lite-v0`, a shared coordinate system,
and prints where they landed:

```
  Atlas coverage (atlas-lite-v0)
    Regions      24 of 258 occupied
    Spread       40% of even coverage  (specialised)
    Off-atlas    0.0%
    Top categories
      general_chat            46%
      code_explanation        31%
```

That describes one corpus. The question worth asking involves two — **what does
this dataset cover that the one I already have does not?**

```bash
dropoutt scan ./candidate --out ./fp/candidate
dropoutt scan ./have      --out ./fp/have
dropoutt diff ./fp/candidate/fingerprint.json ./fp/have/fingerprint.json
```

```
    Similarity   0.02  (1.0 = same distribution over regions)
    Shared       38% of left sits in regions right also occupies
    New          62% of left sits in regions right never reaches

    Only in left — what adding it would bring
      151    12%  import, python, return, data, create
      157    11%  return, function, list, write, given
```

Read it left against right. It is directional on purpose: a small specialised
corpus can sit wholly inside a large one while the large one is barely inside
it. Swap the arguments for the other question.

If either side had too many off-atlas records, `diff` refuses instead of
producing a confident-looking number from two unreliable ones.

Details, including what the five label words are and are not, in
[atlas.md](atlas.md).

---

## 8. Browsing what it knows

These need no data and no network:

```bash
dropoutt checks               # all 27 checks: id, tier, title, what each needs
dropoutt checks T0-MASK-001   # one check in full
dropoutt models               # known models, templates, licences
dropoutt benchmarks           # benchmarks available for contamination scanning
dropoutt atlas                # the atlas and its own quality numbers
```

`dropoutt checks <id>` is the one to reach for when a finding is unclear:

```
  T0-MASK-001  Records contribute zero trainable tokens
  tier 0 · tokenizer · blocking · unverified
  profiles: sft, preference
  requires: chat_template, tokenizer
  blocks under: sft

  Fix  Drop these records, or fix the role names so the assistant span is
       recognised.

  The most expensive silent failure in supervised fine-tuning. A record whose
  label vector is entirely ignore-index contributes nothing to any gradient,
  costs tokens in the packed block, and appears nowhere in the training logs.
  The usual cause is a role name the template does not recognise.
```

Every check carries a `Fix`.

---

## 9. What to do about the common findings

| finding | what it means | what to do |
| --- | --- | --- |
| `T0-MASK-001` | records train nothing at all | fix the role names, or drop the records. Never ignore this one. |
| `T0-ROLE-002` | roles are `human`/`gpt`, not `user`/`assistant` | map them before training. This is the ShareGPT trap: pipelines that mask on `assistant` drop every such record in silence. |
| `T0-SCHEMA-001` | the folder holds logs, not training data | exclude those files |
| `T0-SCHEMA-002` | one folder, several layouts | usually a collection bug; check where the odd files came from |
| `T1-PII-001` | credentials or personal data in the text | remove before training. A leaked key is memorised and can be emitted. |
| `T0-TRUNC-001` | records exceed `--seq-len` | raise the length or truncate deliberately. Check the separate count for records that lose their *whole* assistant span — those are worse than truncated. |
| `T1-OVERLAP-001` | two datasets contain the same records | direction matters. 100% one way and 1% the other means full containment. |
| `T1-NDUP-001` | near-duplicate clusters | **do not reflexively delete.** See below. |
| `T1-LANG-004` | Turkish that lost its diacritics | fix the encoding upstream, or drop. `degil` teaches wrong orthography. |

### On deduplication

The tool reports cluster sizes and does not tell you to delete them, for a
measured reason.

FineWeb deduplicated Common Crawl globally across 96 snapshots. The result
scored **below** their own baseline. When they looked, the data the filter had
*discarded* trained a better model than the data it kept. The fix was per-
snapshot deduplication, which produced 20T tokens instead of 4T.

More filtering is not better. Every finding in this release is labelled
`unverified` because no calibration corpus exists yet, and the tool says so on
every run rather than implying a confidence it has not earned.

---

## 10. Running on a cluster

Two constraints usually apply: no network on compute nodes, and a read-only
home.

```bash
# on the login node, where there is network
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
export HF_HOME=/scratch/$USER/hf
dropoutt fetch --all

# on the compute node
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
export HF_HOME=/scratch/$USER/hf
python -m dropoutt scan /scratch/$USER/data --model qwen3 --offline
```

`--offline` never touches the network. Missing cached components are reported as
skipped or degraded rather than triggering a download. `DROPOUTT_CACHE` honours
`XDG_CACHE_HOME` and falls back to a temp directory when home is not writable.

Full detail in [portability.md](portability.md).

---

## 11. Troubleshooting

**`command not found: dropoutt`** — the venv's `bin/` is not on `PATH`. Use
`python -m dropoutt`.

**`A second file is being added to the wheel archive`** — you are on a checkout
predating 0.1.1. Pull, then reinstall.

**Everything says "language unknown"** — `fasttext-langdetect` is missing, so
the pure-Python fallback ran. It is much weaker and the output says so. Install
`'.[lid]'`.

**"atlas is present but its embedding model could not be loaded"** — install
`'.[atlas]'`. The embedding model downloads once, about 500 MB.

**Token counts look wrong and say `character-ratio`** — no tokenizer backend.
Install `'.[tokenizer]'`.

**A check is firing on something you accept** — mute it with a reason:

```toml
[mute]
checks = ["T1-LIC-001"]  # internal data, licences tracked elsewhere
```

**A scan on a large corpus is slow** — the scan is single-process in this
release. Use `--limit` to sample per file while iterating:

```bash
dropoutt scan ./data --limit 5000
```

---

## Next

- [cli.md](cli.md) — every command and flag
- [checks.md](checks.md) — the full catalog
- [atlas.md](atlas.md) — what the atlas is, and how much to trust its labels
- [fingerprint.md](fingerprint.md) — schema and evidence grades
- [limitations.md](limitations.md) — what this release does not do
