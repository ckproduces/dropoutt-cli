# Research notes, August 2026

Five questions, answered with measurements taken on this machine against this
codebase rather than with estimates. Everything below states what was measured,
on what, and what it does not establish.

Where a number is given, it came from a run. Where a proposal is untested, it
says so.

**Test bench.** A 14-core Apple M4 Pro with 48 GB of memory, macOS, CPython
3.12. Two corpora:

- **`synthetic-480k`** — 480,000 conversational JSONL records across eight
  datasets, 377 MB, half English and half Turkish, generated with a fixed seed.
  Used for whole-scan timing and memory.
- **`atlas-6k`** — 6,000 real documents drawn from the atlas's own reference
  material, median 800 characters, eight-plus languages and four scripts. Used
  wherever the question is about accuracy, because synthetic text is easier than
  real text and would flatter every answer here.

---

## 1. Can `dropoutt scan` be made faster?

### Where the time goes now

`synthetic-480k`, thirteen workers, everything enabled:

| phase | wall clock | share |
| --- | ---: | ---: |
| discovery and index loading | 0.24 s | <1% |
| layout induction | 0.07 s | <1% |
| **the streaming scan pass** | **26.0 s** | **69%** |
| **atlas coverage** | **10.9 s** | **29%** |
| finalize | 0.8 s | 2% |
| total | 37.8 s | |

That is 12,700 records/second and about 10 MB/s. Two phases matter and nothing
else does.

### Inside the scan pass

Profiled on one core over 30,000 records:

| what | share of the pass |
| --- | ---: |
| language identification (`py3langid.instance2fv` and friends) | ~29% |
| regex gating for the content checks | ~6% |
| MinHash signing and shingling | ~11% |
| the record digest, normalisation and document construction | ~7% |
| everything else, spread thin | remainder |

Language identification is the single largest item, and the reason is
structural: `instance2fv` is a Python loop over **every byte** of the text,
stepping a DFA one character at a time. A 700-byte record is 700 dictionary
lookups.

### Levers, measured

**Feed the classifier less text.** Language is a bag-of-n-grams decision and
converges fast. Measured on `atlas-6k`, agreement with the answer the same
model gives on 2,000 characters:

| head characters | agrees with the 2,000-char answer | µs per call |
| ---: | ---: | ---: |
| 128 | 97.18% | 72 |
| 256 | 98.70% | 90 |
| 512 | 99.47% | 126 |
| 768 | 99.95% | 162 |
| 1024 | 100.00% | 161 |

End to end on `synthetic-480k`, though, cutting the head from 2,000 to 768
characters bought **3%** of total scan time, because that corpus's records are
already about 700 characters and the cap never binds. It is worth 2.5× *on the
classification call* and therefore matters only on corpora of long documents.

**Verdict: not shipped.** Changing the reported language of 0.05% of records to
buy 3% on one corpus shape is a bad trade, and the honest description of the
lever is "helps long-document corpora, does nothing for short-record ones". The
constant is `LanguageDetector.HEAD_CHARS` for anyone who wants to make that
trade deliberately.

**The real fix for language identification** is to stop stepping the DFA in
Python. Three options, none prototyped:

1. Reconstruct the n-gram → feature-index map from the automaton once at load,
   then count features with `np.unique` over a strided view of the byte array.
   Vectorises the whole thing. The work is in proving the reconstruction is
   exact.
2. Identify language on a *sample* per dataset and assign the majority to the
   rest. Cheap, and wrong exactly where the language check is most useful —
   corpora that are a mixture. Rejected.
3. Run identification only where a result is used: the atlas sample, the budget
   sample, and the deviation check. That is currently every record; a bounded
   sample would need the deviation check restated as a rate with an interval.

**Shipped in this release, and measured:**

- The atlas built the same 200,000 × 212 similarity matrix three times —
  `assign_full`, `categorize` and `soft_assign` each began by computing it.
  Doing it once is **34% faster** for that step (0.60 s → 0.39 s) and removes
  two similarity matrices from peak memory.
- The LSH band index was fourteen Python dictionaries per document. As a dense
  array of 64-bit band hashes recovered by sorting, it is roughly thirty times
  smaller and the second phase got quicker with it.
- The exact-duplicate tally moved from `dict[int, int]` to an array counted once
  with `np.unique`.

### Inside the atlas phase

10.9 s for 200,000 records, and it is tokenization and SIF pooling — the
similarity matrix is 0.4 s of it. Two candidate levers:

**Cap the tokens per record.** Measured on `atlas-6k`, agreement with placement
at 512 tokens:

| max tokens | encode time (24k docs) | same cell | same subject area |
| ---: | ---: | ---: | ---: |
| 512 | 1.58 s | 100.00% | 100.00% |
| 384 | 1.48 s | 99.39% | 99.51% |
| 256 | 1.40 s | 97.91% | 98.41% |
| 192 | 1.37 s | 92.07% | 94.17% |
| 128 | 1.25 s | 73.42% | 80.02% |

**Verdict: rejected.** Placement degrades fast below 256 tokens and the encode
time barely moves, because the cost is tokenizing the text that is *there*, not
the cap. Two percent of records changing cell to buy 11% of one phase is not a
trade worth making.

**Shrink the sample.** The atlas sample is 200,000 records for a 212-cell
histogram — roughly 900 observations per cell. The statistics do not need that;
the constant was raised from 20,000 because that under-sampled large corpora,
and the right rule is probably "scale with cells" (say 500 per cell, so ~106,000
for this atlas) rather than a flat number. This would halve the atlas phase.
Untested, and it changes reported coverage, so it needs a comparison run first.

### Levers not yet pulled

- **The scan pass is pure Python.** The largest available win is not a
  micro-optimisation, it is moving the per-record inner loop — normalise,
  digest, shingle, gate — into vectorised batches over columns of records rather
  than one dictionary-shaped `Document` at a time. That is a rewrite of
  `run_shard`, not a patch.
- **`.jsonl` parsing** already uses orjson. Reading is not the bottleneck: the
  pass moves 10 MB/s while the disk does hundreds.
- **GPU.** Nothing in the scan pass is GPU work. The one arithmetic step, the
  sparse-dense multiply behind the embeddings, now runs on a device when torch
  and a GPU are both present — worth seconds on a large sample, not minutes.

---

## 2. A cheaper, smaller, equally accurate multilingual encoder

### What is shipped now

`minishlab/potion-multilingual-128M`: a model2vec static model, a 500,353 × 256
float32 lookup table, **488.6 MB** on disk plus a 17.8 MB bge-m3 tokenizer. It
scores 47.31 on MMTEB, which is 90.86% of LaBSE, over 101 languages. Loading it
costs about 1.3 GB resident, because model2vec makes a second copy on the way
in.

The atlas is a **128-dimensional** space. The potion models are Matryoshka, so
the first 128 columns *are* the 128-dimensional model. **Half of that 488 MB is
downloaded, held in memory, and never read.**

### The measurement

Rather than ask which other model to use, the more useful question is how much
of the one we have is actually needed. Measured on `atlas-6k` by the only
criterion that matters — does a record still land in the same cell of the
atlas:

| table | size | same cell | same subject area |
| --- | ---: | ---: | ---: |
| shipped: 500,353 × 256 fp32 | 489 MiB | — (baseline) | — |
| first 128 columns, fp32 | 244 MiB | **100.00%** | **100.00%** |
| first 128 columns, float16 | 122 MiB | **99.98%** | 100.00% |
| first 128 columns, int8 per-row scale | 63 MiB | **99.63%** | 99.73% |

And the vocabulary is mostly unused. Over 36,000 real documents (6.8 M tokens):

| rows kept | share of token occurrences covered | size at 128-d fp32 |
| ---: | ---: | ---: |
| 20,000 | 86.8% | 10 MiB |
| 50,000 | 95.6% | 24 MiB |
| 100,000 | 99.8% | 49 MiB |
| 200,000 | 100.0% | 98 MiB |

Only 105,400 of the 500,353 rows were touched at all — 21% of the vocabulary.

### Recommendation

**Do not change models. Ship a compressed build of the one we have.**

A 128-column, int8-quantised table with a pruned vocabulary is **under 40 MB**
against 489 MB today — a twelve-fold reduction in download and in resident
memory — for 99.6% identical cell placement and 99.7% identical subject area.
Nothing about the atlas has to be rebuilt, because the coordinate system is
unchanged: the same clusters, the same centroids, the same reference sizes.

The work is: quantise and prune once, publish the artifact, load it, and record
the quantisation in the atlas identity block so a coverage number measured
against the compressed encoder is never silently compared with one measured
against the full encoder. That last part is not optional — the report already
carries `encoder_weight_hash` for exactly this reason.

Two things to be careful about, one of which already bit:

- **Normalisation width.** The model config sets `normalize: true`, so
  `StaticModel.encode` L2-normalises over whatever width the table has. Loading
  128 columns normalises over 128; loading 256 and truncating afterwards, which
  is what `Embedder.encode` does, does not. The two differ by up to 0.057 per
  component. A compressed build has to make truncation and normalisation the
  same operation in both paths first. There is a note in `atlas/embed.py` where
  this was tried and reverted.
- **The long tail of languages.** Pruning the vocabulary by frequency in the
  *reference* corpus prunes it in favour of the languages that corpus contains.
  A user scanning a language the reference corpus barely has would lose more
  than 0.2%. Prune by frequency *and* keep every row above a floor in any
  script, or measure per-language before shipping.

### The alternative, for the record

`sentence-transformers/static-similarity-mrl-multilingual-v1`: 105,879 × 1024,
Matryoshka down to 32 dimensions, Apache 2.0, 51 languages. At 128 dimensions
that is 54 MB — comparable to the compressed build above, and it is better at
retrieval and STS while potion is better at classification and clustering, which
is what an atlas is.

It supports half as many languages. For a tool that exists partly to be honest
about Turkish, Azerbaijani and Arabic, that trade is not obviously worth making
to reach a size the current model reaches by being compressed. **Reject for
now**; revisit if the compressed build turns out worse than measured here.

---

## 3. Measuring more than semantic topic

Coverage against the atlas answers "what is this corpus about". A person
training a model wants several other things, and every one of them is a
*distribution over records* that the scan is already in a position to measure.

The key structural observation: **the scan already embeds up to 200,000 records
into a 128-dimensional space.** Anything that can be read off that embedding
with a linear probe costs one 128 × k matrix multiply for the whole corpus —
microseconds — and ships as a few kilobytes of weights. The atlas already
carries such a probe (`probe_coef`, `probe_intercept`) and already reports its
holdout accuracy, so the precedent for "a probe, with its accuracy stated" is in
the codebase.

That suggests a facet system: the atlas is one facet, and each new one declares
its method, its output, and what it does not establish.

### Facets that need no model at all

Cheap, exact, and computable in the existing streaming pass:

| facet | what it measures | why a trainer cares |
| --- | --- | --- |
| **Sentence shape** | sentences per record, words per sentence, the distribution rather than the mean | A corpus of uniform 12-word sentences teaches uniform 12-word answers |
| **Response opening entropy** | entropy of the first three tokens of every assistant turn | The sharpest single detector of distillation collapse. If 40% of answers begin "Sure! Here's", the model will too |
| **Turn structure** | turns per conversation, prompt:answer length ratio, multi-turn share | An SFT set that is 98% single-turn will not hold a conversation |
| **Register and surface** | question / imperative / statement share, punctuation profile, list-vs-prose share, code-fence share, markdown-structure share | "Every answer is a bulleted list" is a real and common defect |
| **Lexical richness** | type-token ratio and hapax rate, per dataset, length-normalised | Distinguishes a large corpus from a large corpus of the same thing |
| **Numeric and entity density** | digits, dates, URLs, identifiers per record | Predicts what the model will hallucinate confidently |

None of these needs a model, a download or a second pass. They are the same
shape as the checks that already exist and would slot into the same catalog.
**This is the cheapest large increase in what dropoutt reports, and it should
come first.**

### Facets that need a probe over the existing embedding

| facet | how | what it costs |
| --- | --- | --- |
| **Sentiment / emotional tone** | logistic regression over the 128-d embedding, trained on a public multilingual sentiment set | ~1 KB of weights, no extra pass |
| **Intent / task type** | the same, over a taxonomy: summarise, translate, generate code, extract, classify, roleplay, reason, refuse | ~2 KB |
| **Formality / register** | the same, binary | ~1 KB |
| **Refusal and safety-shape** | high-precision pattern gate first, probe second — refusals have strong lexical markers in every language | negligible |

Each one needs labelled training data, a holdout, and a stated accuracy. The
rule the atlas already follows applies: **a probe's label is a caption, not a
claim**, and a facet whose holdout accuracy is 0.7 must say 0.7 next to every
number it produces.

### What this cannot become

A "quality score". Every facet above is a description, and the project's own
design rule — nothing recommends deleting data without a measured effect —
means none of them may be summed into a number that ranks corpora. The value is
in the distributions and in the comparison against a reference, which is exactly
what the atlas does for topic.

---

## 4. Benchmark comparison: overlap, and estimating a score

Two halves, with very different answers.

### The overlap half: yes, and most of it exists

`dropoutt` already ships MinHash + exact contamination indices for ten
benchmarks and already reports per-benchmark contamination with witnesses. What
is missing is the ability to point it at a benchmark file the user supplies:

```
dropoutt scan ./data --benchmark ./mmlu_test.jsonl
```

The machinery — `contamination.py`, the index format, the merge across shards —
is all there. This is a CLI flag, an index builder over a supplied file, and a
report section. It is the smallest useful version of the feature and it should
be built.

Beyond exact and near-duplicate overlap, three measurements are worth more and
are all reachable with what the scan already computes:

1. **Topical support.** Embed the benchmark's items with the same encoder, place
   them on the same atlas, and compare their cell histogram with the training
   data's. "Your data reaches 62% of the cells this benchmark's questions land
   in, and is 5× under-represented in the three cells holding 30% of them" is a
   real, checkable statement about whether the corpus can teach the task.
2. **Nearest-neighbour support per item.** For every benchmark item, the
   similarity to its nearest training record and the number of records within a
   radius. Items with no support are items the model has nothing to learn from.
   Reported as a distribution, plus the list of unsupported items.
3. **Contamination as a floor.** Items whose text appears in the training data
   are items the benchmark can no longer measure. That is a floor on a score
   obtained by memorisation, and it is the most actionable number of the three.

All three are honest, all three are computable, and all three answer the
question a user is really asking.

### The score half: no, and it should not be shipped

Estimating a benchmark score from training data alone is not something anyone
can currently do with defensible accuracy. Scaling laws predict loss from
compute and data volume; they do not predict task accuracy, and the mapping from
loss to downstream accuracy is model-family-specific, emergent in places, and
requires training runs to calibrate. Data-composition work reports *relative*
effects of mixture changes measured by training models, not absolute predictions
from data statistics.

Shipping a predicted score would also break the project's own contract. The
report says, in every format, that findings are structural observations with no
measured effect on model quality attached. A number labelled "estimated MMLU:
54.2" is the opposite of that, and the first user who trains a model and scores
41 stops trusting everything else in the report.

**Recommendation:** build the overlap feature, build the three measurements
above, and name the section something that promises what it delivers —
"coverage of this benchmark", not "estimated score". If a score estimate is
wanted later, the only defensible route is empirical: collect
(corpus fingerprint, model, benchmark score) triples from real training runs and
fit a predictor whose error bars are published. That is a data-collection
programme, not a feature.

---

## 5. Making atlas coverage understandable on the first read

A/B testers did not understand the coverage section first time. Reading it as a
stranger, the reason is legible.

### What goes wrong

**It asks for three concepts before showing anything familiar.** The grid's
central number is *density relative to a reference corpus*, which requires
knowing that a reference corpus exists, what is in it, and why a ratio against
it is the interesting quantity. None of that is established before the grid
appears.

**Two different percentages sit in adjacent columns.** "Share" is a share of
your data. "Density" is a ratio against the map. They are visually alike and
mean opposite things; a reader who conflates them draws exactly the wrong
conclusion about what to add.

**The cells are anonymous.** The design rule is that a place is named by the
reader's own record before it is named by a caption — and the grid, which is the
lead visual, is 212 unlabelled squares of ratios. The rule is followed
everywhere except the first thing anyone looks at.

**Nothing tells the reader what to do.** A grid of ratios is a measurement. The
question a reader has is "what should I add".

### Proposed sequence

1. **One sentence, in their units.** *"Your data covers 47 of the 212
   neighbourhoods on the map. It is concentrated: a quarter of it sits in one
   place."* No ratios, no reference corpus, no vocabulary to learn.
2. **One coverage bar.** A single horizontal bar of 212 segments, filled for
   reached and empty for not. It answers "how much of the map do I touch" in
   one glance and needs no legend.
3. **A diverging bar chart of the extremes, labelled with the reader's own
   records.** Top five over-represented above the line, top five
   under-represented below, each row captioned with one of *their* records from
   that place and an explicit instruction — *cut* or *grow*. Diverging bars are
   read instantly and correctly by people who have never seen one before, which
   is not true of a density grid.
4. **The density explained by example, once, in a caption next to the first
   number that uses it.** *"Films: 25% of your data is here. The reference
   corpus is 17% films. That is 1.5× as dense."* Three concrete numbers beat
   any definition of a ratio.
5. **The full grid, behind a disclosure, for the reader who wants it.** It is
   the right artifact for someone auditing the whole map and the wrong one for
   someone meeting it for the first time.

### Two smaller changes worth making regardless

- **Separate the two percentages visually.** Share-of-yours and density-vs-map
  should not be adjacent right-aligned columns in the same table. Put density in
  a different register — a chip, a bar, a colour — so they cannot be read as the
  same kind of number.
- **Say what the reference corpus is, in one line, above the first comparison.**
  The atlas has a version, a size and a composition, and all of it currently
  lives in a footer.

### The proposal, drawn

`docs/coverage-redesign.html` is the sequence above as working mockups, built
from a real scan — 1,498 records of real multilingual text, 1,464 placed, on
`atlas-v1-lite`. Open it in a browser. Every number in it came out of
`report.json`, including the 212 per-neighbourhood densities behind the coverage
bar and the full 48-row grid behind the disclosure.

### How to know whether it worked

The A/B test needs a comprehension task, not a preference question. Give the
reader the section and ask: *"name one subject area you should add more of, and
one you have too much of."* Measure the share who answer correctly and the time
they take. That is the thing the section exists to make possible, and it is the
only measurement that would have caught the current design.
