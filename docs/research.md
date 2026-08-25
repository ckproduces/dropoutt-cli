# Research notes, August 2026

Seven questions, answered with measurements taken on this machine against this
codebase rather than with estimates. Everything below states what was measured,
on what, and what it does not establish.

Where a number is given, it came from a run. Where a proposal is untested, it
says so. Sections 1 and 2 were written before the work they describe and have
been rewritten around what shipped; section 5 records a proposal that was
rejected.

**Test bench.** A 14-core Apple M4 Pro with 48 GB of memory, macOS, CPython
3.12, thirteen workers unless a measurement says "serial". Three corpora:

- **`atlas-118k`** — 118,616 real records, 150 MB across four JSONL shards,
  drawn from the atlas's own reference material: 73% English, 6.7% Turkish, and
  two per cent each of Spanish, French, German, Russian, Chinese, Japanese and
  Arabic. Median record 1,080 characters. Used for whole-scan timing, memory
  and fingerprint comparison.
- **`atlas-9k`** — 9,000 held-out documents from the same material, eight-plus
  languages and four scripts. Used wherever the question is about accuracy,
  because synthetic text is easier than real text and would flatter every answer
  here.
- Public benchmark splits where a facet needed labels it could not get from a
  corpus: TweetEval sentiment and emotion, and the Pavlick–Tetreault formality
  scores. Named where they are used.

---

## 1. Can `dropoutt scan` be made faster?

**Yes. It is 1.28× faster end to end and 1.49× faster in the streaming pass, and
the largest single item in that pass went from 41% of it to 21%.** Everything
below is what was measured, including two levers that were tried and rejected
and one bug the work uncovered.

### Where the time went, and where it goes now

`atlas-118k`, thirteen workers, everything enabled:

| | 1.1 | 1.2 |
| --- | ---: | ---: |
| the streaming scan pass | 8.97 s | **6.01 s** |
| atlas coverage | 15.6 s | **13.1 s** |
| the scan, as `elapsed_seconds` reports it | 23.6 s | **18.8 s** |
| the whole command, wall clock | 28.3 s | **22.3 s** |
| the whole command, CPU | 194 s | **143 s** |
| peak resident, parent | 3.39 GB | **3.00 GB** |

The scan pass on one core, which is where the per-record cost is visible without
thirteen workers hiding it:

| | 1.1 | 1.2 |
| --- | ---: | ---: |
| everything | 58.31 s | 41.94 s |
| the same, with language identification off | 34.34 s | 33.19 s |
| **language identification** | **23.97 s** | **8.75 s** |
| tier-0 checks alone | 5.74 s | 5.26 s |
| tier-1 checks (near-duplicates, PII, identity, style) | 28.9 s | 28.4 s |

### The scan pass is a batch at a time now

`run_shard` used to be one dictionary-shaped `Document` at a time all the way
down: one classification, one script decision, one numpy call per array
operation, per record. It reads :data:`parallel.RECORD_BATCH` records at a time
now and computes the shared features by column — language, script, sizes, sample
keys — then hands the same batch to the checks through `Check.observe_batch`.

The default `observe_batch` is the loop it replaced, error handling included, so
a check that has nothing to gain from a batch does not have to know one exists
and behaves exactly as before. Two checks override it: the near-duplicate pair,
which share one MinHash store and are handed the same batch back to back, so the
store now recognises a batch it has already taken.

**The fingerprint is byte-identical.** `atlas-118k` produces
`fp_48989bca302c9564878a18f8e2702cf5` before and after, so `PIPELINE_VERSION`
was *not* bumped: a version that moves invalidates every stored fingerprint, and
this one measures the same numbers.

### Language identification: the automaton, read as arithmetic

py3langid spends nearly all of its time in `instance2fv`, a Python loop over
every byte of the text: a dict lookup and a list extend per byte, so a
1,277-byte record is 1,277 of each. It was 41% of the streaming pass — larger
than deduplication and larger than every content check together.

The loop is not vectorisable as written, because the state after byte *i*
depends on the state after byte *i-1*. What is vectorisable is what the loop
*computes*. The automaton is Aho-Corasick over byte n-grams: after reading a
prefix it sits in the state naming the longest suffix of that prefix which is a
trie node, and that state's output set is every pattern ending at that position.
So the feature vector is exactly

    fv[f] = the number of times the byte string of feature f occurs in the text

— a bag of n-grams, countable with array arithmetic over a whole batch at once,
with no automaton at all.

Recovering the byte string of each feature is a breadth-first search of the
transition table: a trie node at depth *d* is reachable in *d* steps and no
fewer, so level-order traversal from the root discovers every state at a depth
equal to the length of its string, and the path spells that string out. A
feature is emitted by a state exactly when its pattern is a suffix of that
state's string, so the shortest string among the states emitting a feature *is*
that feature's pattern.

**The reconstruction is verified, not assumed.** For all 9,118 states of the
shipped model, the output set equals exactly the set of features whose recovered
pattern is a suffix of that state's string — checked exhaustively, at load, in
about a tenth of a second, with the model refusing to build the fast path when
it fails. That check is what makes the substitution legitimate rather than
plausible; `tests/test_ngram_langid.py` runs it and then compares feature
vectors against `instance2fv` directly.

The shipped model turns out to be 7,480 features over byte n-grams of length one
to four: 53 single bytes, 2,694 pairs, 2,021 triples and 2,712 quadruples, each
with a unique byte string. One- and two-byte grams get a direct index table; the
longer two get a sorted lookup. Only 13% of positions in real text hit anything.

Measured on `atlas-9k`, mean 1,277 bytes per document:

| | µs per document | labels differing | largest probability difference |
| --- | ---: | ---: | ---: |
| py3langid `classify` | 170.2 | — | — |
| this module | **49.3** | **0 of 9,000** | 6e-5 |

Two smaller things fell out of the same work. py3langid renormalises the whole
97-class log-probability vector into a distribution — an O(C²) pass of 9,409
exponentials — and then reads one entry of it. That entry is
`1 / Σ_j exp(pd_j − pd_max)`, the same sum over the same values in the same
order, so it is the same float computed once instead of 97 times. And
`dominant_script` became `dominant_scripts`, one pass of array arithmetic for
the batch instead of five numpy calls per record.

### The bug this found: a dense multiply cost the scan its workers

The first version of the batched classifier used a dense
`(records × features) @ (features × classes)` multiply. On this machine the scan
got **slower** — 23.6 s to 53.1 s — and the report said `parallel scan
unavailable (BrokenProcessPool), ran on one core`.

The cause is not in this package. On macOS numpy links against Accelerate, whose
`gemm` dispatches through libdispatch, and **a process that has called it cannot
be forked and then call it again**: the child dies of SIGSEGV before raising
anything Python can catch. Isolated:

| operation in a forked child, after the parent did a `gemm` | outcome |
| --- | --- |
| `gemv` (a vector times a matrix) | fine |
| SciPy sparse times dense | fine |
| `einsum` | fine |
| `gemm` (a matrix times a matrix) | **SIGSEGV** |

Which is why py3langid's own `np.dot` had never tripped it: a feature vector
times the class matrix is a `gemv`.

Phrasing the multiply sparsely fixes it and is the honest shape anyway — six or
seven hundred n-gram occurrences land in a row 7,480 wide — and produces
identical numbers. **The rule this leaves behind: nothing on the scan path may
reach a BLAS `gemm`,** because the scan path runs in forked workers.
`tests/test_fork_safety.py` holds the line, and was checked by reintroducing the
dense multiply and watching it fail.

### The second bug: the device path was unreachable

`GPU_MIN_DOCS` was 20,000 while the encode path chunks at `ENCODE_CHUNK =
16,384`, so no batch could ever reach the threshold. The accelerated
sparse-dense multiply shipped in 1.1 as code that could not run. The threshold
is 4,096 now and a test keeps the two in order.

### Levers measured and rejected

**Feed the classifier less text.** Language is a bag-of-n-grams decision and
converges fast; on `atlas-6k`, 512 characters agreed with the 2,000-character
answer 99.47% of the time. End to end it bought 3%, because records are already
about 700 characters and the cap rarely binds. Changing the reported language of
0.05% of records to buy 3% on one corpus shape is a bad trade. The constant is
`LanguageDetector.HEAD_CHARS` for anyone who wants to make it deliberately.

**Drop SciPy and do the multiplies in numpy.** SciPy is 99 MB of the install for
what is essentially two calls. Both were rewritten as a sorted gather plus
`np.add.reduceat` and measured:

| multiply | SciPy | numpy only | transient |
| --- | ---: | ---: | --- |
| classifier, per document | 48.4 µs | 145.8 µs | 60 MB per 256-record batch |
| atlas pooling, 9,000 documents | 0.33 s | 1.40 s | 149 MB per 1,024 documents |

Three to four times slower and an order of magnitude more transient memory. The
99 MB stays. See section 7.

**Cap the tokens per record in the atlas.** Measured on `atlas-6k`, placement
degrades fast below 256 tokens (97.9% same cell at 256, 73.4% at 128) while
encode time barely moves, because the cost is tokenizing the text that is
*there*, not the cap. Rejected.

### What is left, and what it would cost

The tier-1 checks are now the largest item in the streaming pass at 28.4 s
serial — MinHash shingling and signing, and the PII, identity and style regex
families. The shingle hash is BLAKE2b per word, memoised, and 19.8 million
lookups on this corpus; a vectorised polynomial hash would be materially faster
and would change every MinHash signature, which means every near-duplicate
count. That is a coordinate change for the headline dedup number, not an
optimisation, and it needs a version bump and a comparison run before anyone
takes it.

Batch-level gating for the regex families was prototyped and abandoned: testing
a gate token against the whole batch costs the same scan as testing it against
each record, because a substring search stops at the first hit, so the batch
version adds a pass rather than removing one. The per-record gate is already the
optimisation.

The atlas phase is now 69% of a scan, and 66% of *that* is tokenization, already
running about ten-way parallel inside `tokenizers`. The remaining lever there is
the sample size: 118,616 records for a 212-cell histogram is roughly 560
observations per cell, and the statistics do not need that. It would halve the
phase and it changes reported coverage, so it needs a comparison run first.

---

## 2. A cheaper, smaller, equally accurate multilingual encoder

**Shipped. The encoder is the same model, stored the way it is used: the first
128 columns, one byte per weight, with a per-row scale. 489 MB became 63 MB and
99.74% of records land in the same cell.**

### What the problem was

`minishlab/potion-multilingual-128M` is a 500,353 × 256 float32 lookup table —
488.6 MB on disk, about 1.3 GB resident once model2vec had copied it in. The
atlas is a **128-dimensional** space and the potion models are Matryoshka, so
the first 128 columns *are* the 128-dimensional model. Half of that table was
downloaded, held in memory, and never read. The half that was read was carried
at four bytes a weight for a coordinate system whose cells are nowhere near that
fine.

### The measurement

The only criterion that matters is whether a record still lands in the same cell
of the atlas. Measured on `atlas-9k` through the shipped `atlas-v1-lite`, with
the real IDF table and the real normalisation:

| table | size | same cell | same subject area |
| --- | ---: | ---: | ---: |
| 500,353 × 256 float32 | 489 MiB | — (baseline) | — |
| first 128 columns, float32 | 244 MiB | 100.00% | 100.00% |
| first 128 columns, int8 with a per-row scale | **63 MiB** | **99.74%** | **99.79%** |

Per row rather than per table because the rows of a static embedding differ in
magnitude by orders of magnitude — a frequent subword and a rare one are not on
the same scale — and one global scale would spend the whole int8 range on the
largest rows and quantise the rest to noise. The median row's largest error is
0.39% of that row's peak.

On a whole scan of `atlas-118k`, against the uncompressed encoder: 115,884
records placed becomes 115,883, effective reach 158.305 becomes 158.275, and all
212 subregions are still touched.

### What shipped

| | 1.1 | 1.2 |
| --- | ---: | ---: |
| encoder cache on disk | 507 MB | **81 MB** (63 MB encoder, 18 MB tokenizer) |
| loading it | 1.51 s | **0.58 s** |
| peak resident, atlas phase | 2.30 GB | **1.62 GB** |
| atlas phase, 118k records | 15.5 s | **13.1 s** |

There is **no second, uncompressed build**. Shipping both would mean two
coordinate systems with one name and a coverage number whose meaning depended on
which one a user happened to have.

The conversion runs once, on the machine, from the published weights, and
deletes the original afterwards. It is deterministic — a fixed slice, a fixed
scale, round-half-to-even — so every install derives byte-identical codes from
byte-identical weights, and `encoder_weight_hash` stays a statement about the
coordinate system rather than a per-machine accident. Checked rather than
asserted: a fresh download converted under CPython 3.14, in a separate venv with
its own numpy build, produced the same `2d0eb24bce48…`. The report now carries
both that hash and `encoder_built_with`, the hash of the float32 weights the
atlas was *fitted* on, because since 1.2 the two differ by construction.

Three things fell out of it:

- **model2vec is gone.** Every part of it this package used was a table lookup
  and a pooled average; it copied the whole table on the way in and pulled
  joblib, tqdm and rich behind it. The safetensors file is read directly — an
  eight-byte header length, a JSON header, a raw buffer, about twenty lines.
- **The normalisation width stopped being ambiguous.** model2vec's `encode`
  L2-normalises over whatever width the table has, so loading 256 columns and
  truncating gave a different vector from loading 128 — by up to 0.057 per
  component, which is why an earlier attempt at this was reverted. A table that
  is 128 wide has one answer.
- **The table is memory-mapped.** A scan touches about a fifth of the
  vocabulary; the rest is never paged in.

### What was not done, and why

**The vocabulary was not pruned.** Only 105,400 of the 500,353 rows are ever
touched on real text — 21% — and pruning to the top 100,000 would reach 99.8% of
token occurrences at 49 MiB. It is not done because pruning by frequency in the
*reference* corpus prunes in favour of the languages that corpus contains, and a
user scanning a language it barely holds would lose more than the 0.2% headline
suggests. Sixty-three megabytes is small enough that the trade is not worth
making blind.

**The model was not changed.** `sentence-transformers/static-similarity-mrl-multilingual-v1`
is 105,879 × 1024, Matryoshka to 32 dimensions, Apache 2.0 — 54 MB at 128
dimensions, comparable to the compressed build, better at retrieval and STS
while potion is better at classification and clustering, which is what an atlas
is. It supports 51 languages against potion's 101. For a tool that exists partly
to be honest about Turkish, Azerbaijani and Arabic, halving the language
coverage to reach a size the current model reaches by being compressed is not a
trade worth making.

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
| **Sentiment / emotional tone** | logistic regression over the 128-d embedding | 1.5 KB of weights, no extra pass |
| **Intent / task type** | the same, over a task taxonomy | 6 KB |
| **Formality / register** | the same, binary | 0.5 KB |
| **Refusal and safety-shape** | high-precision pattern gate first, probe second — refusals have strong lexical markers in every language | negligible |

These were estimates when this section was written. **Section 6 measures all
three of them**, against public benchmarks and against a published transformer
on the same test set, and the answer is not the same for every facet.

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

**Status: proposed, reviewed, and rejected. The shipped report design stands.**
The mockups that accompanied this section have been deleted rather than left in
the tree to be mistaken for a plan.

The finding that prompted it was real — A/B testers did not understand the
coverage section on a first read — and the reading of *why* still holds, so it
is kept here. What was rejected is the redesign, not the diagnosis.

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
lead visual, is 212 unlabelled squares of ratios.

**Nothing tells the reader what to do.** A grid of ratios is a measurement. The
question a reader has is "what should I add".

### What was proposed

A five-step sequence replacing the grid as the lead visual: one plain sentence
in the reader's own units, a single 212-segment coverage bar, a diverging bar
chart of the five most over- and under-represented areas captioned with the
reader's own records, density explained once by example, and the full grid moved
behind a disclosure.

### Why it was rejected

The report has one design and it is the one in `report.html`. Four renderings —
page, markdown, JSON, terminal — are held to be equivalents of each other, and
the proposal restructures the lead visual of only one of them; carrying it
properly means re-cutting all four around a shape that has not been shown to
read better, because the comprehension test below was never run. Rewriting the
section that the whole report is organised around, on the strength of an
unmeasured hypothesis, trades a known design for an unknown one.

### The part of it that survives

Two changes are worth making inside the current design, and neither disturbs it:

- **Separate the two percentages visually.** Share-of-yours and density-vs-map
  should not be adjacent right-aligned columns in the same table. Put density in
  a different register — a chip, a bar, a colour — so they cannot be read as the
  same kind of number.
- **Say what the reference corpus is, in one line, above the first comparison.**
  The atlas has a version, a size and a composition, and all of it currently
  lives in a footer.

### How to know whether any of it worked

The A/B test needs a comprehension task, not a preference question. Give the
reader the section and ask: *"name one subject area you should add more of, and
one you have too much of."* Measure the share who answer correctly and the time
they take. That is the thing the section exists to make possible, and it is the
only measurement that would settle the question the redesign was guessing at.

---

## 6. Sentiment, intent and formality: train one, borrow one, or probe the one we have?

Three candidate columns — *sentiment / emotional tone*, *intent / task type*,
*formality / register* — and three ways to get them. The question was framed as
train-from-scratch against use-a-public-model. There is a third option this
codebase is unusually well placed to take, and it changes the answer for two of
the three columns.

**The three options**

1. **Train from scratch.** A classifier per facet, our labels, our taxonomy.
2. **Use a public model.** Download a fine-tuned transformer and run it.
3. **Probe the embedding already computed.** The scan already embeds the sample
   into the atlas's 128-dimensional space. A logistic regression on top of that
   is a few kilobytes of coefficients and one matrix multiply for the whole
   corpus.

### What each one actually scores

Measured here, on the public test splits, with the atlas encoder as shipped —
128 columns, int8. The transformer is `cardiffnlp/twitter-roberta-base-sentiment-latest`
run as int8 ONNX, which is the model fine-tuned *on this benchmark's own
training split*, so it is close to a ceiling rather than a fair generic
baseline. TF-IDF is a bag of 1–2 grams with logistic regression, included
because it is what "no model at all" looks like.

| facet | test set | majority | **atlas probe** | tf-idf | published transformer |
| --- | --- | ---: | ---: | ---: | ---: |
| Sentiment (3 classes) | TweetEval sentiment, 12,284 | 48.3% | **60.6%** / 0.586 F1 | 58.9% / 0.554 | **72.5%** / 0.725 F1 |
| Emotion (4 classes) | TweetEval emotion, 1,421 | 39.3% | **70.9%** / 0.651 F1 | 62.6% / 0.514 | not run |
| Formality (binary) | Pavlick–Tetreault, 2,000 | 49.1% | **75.1%** / 0.752 F1 | 75.8% / 0.767 | 85.2% English, 79.4% overall (reported) |
| Task type (11 classes) | 2,396 held-out records, labelled by dataset of origin | 9.4% | **93.5%** / 0.933 F1 | 97.6% / 0.975 | none exists for this taxonomy |

Accuracy first, macro-F1 second. The formality transformer figure is
`s-nlp/xlmr_formality_classifier`'s own reported accuracy on XFORMAL, not a run
of ours, and covers four languages.

### What each one costs

Also measured, on the same 12,284 documents and the same machine:

| | throughput | added install | added download |
| --- | ---: | ---: | ---: |
| transformer, int8 ONNX, 128 tokens | **185 docs/s** | onnxruntime, 23 MB | 126 MB per model |
| atlas embedding (already paid by coverage) | 69,175 docs/s | none | none |
| the probe on top of it | ~10⁸ docs/s | none | ~6 KB per facet |

**That is the number that decides it.** A scan of `atlas-118k` takes 19 seconds
today. One transformer facet over the same records is 641 seconds — the scan
becomes 34× slower to add one column. A distilled six-layer model would be
perhaps three times quicker and still turn a twenty-second scan into a
three-minute one. And it is per facet: three columns is three passes.

The install cost is smaller than expected and should not be the argument.
onnxruntime resolves from wheels on every supported interpreter and platform
except macOS on CPython 3.14, and torch resolves everywhere — so the rule that
removed fastText does not, on today's index, remove these. Torch's 527 MB Linux
wheel would nearly triple the install; onnxruntime's 23 MB would not.

### The multilingual test, which is where the probe earns its place

A probe trained on English, tested zero-shot on Turkish. Encyclopedic prose
against instruction-following text, English Wikipedia and Alpaca for training,
Turkish Wikipedia and InstrucTurca for testing — 1,200 records each, no Turkish
in training at all:

| | in-language (English) | zero-shot (Turkish) |
| --- | ---: | ---: |
| atlas probe | 97.4% | **80.3%** |
| tf-idf | — | **50.4%** (chance) |

The static multilingual encoder puts the two languages in a shared space, so one
set of 128 coefficients transfers. A bag of words cannot transfer at all. This
is the property that matters for a tool whose whole point is being honest about
Turkish and Arabic, and it is the one thing on this page that neither a
per-language public model nor a from-scratch English classifier gives you: the
best public formality model covers four languages, and the best public intent
models cover a voice-assistant taxonomy.

### Recommendation, per column

**Task type / intent — probe, and it is not close.** 93.5% from six kilobytes,
transferring across languages, on the facet where no public model has the right
taxonomy anyway. Public intent models classify utterances into voice-assistant
categories (MASSIVE's 60 intents over 51 languages); nobody publishes "is this
record a math word problem, a summarisation instruction, or a dialogue turn",
which is the taxonomy a person building a training set needs. This is the column
where "train from scratch" was the only alternative, and the probe removes the
need. **Ship it, with its holdout accuracy printed beside every number.**

Two cautions on that 93.5%: the labels are dataset-of-origin, which is a proxy
for task and not the same thing, and TF-IDF beats the probe in-language (97.6%),
so on a monolingual English corpus the embedding is not what is doing the work.
The probe's advantage is cross-lingual, and the honest facet is a probe whose
*reported* accuracy comes from a properly labelled multilingual set that does
not exist yet. Building that set is the real work.

**Formality / register — probe, with the gap stated.** 75.1% against a reported
85.2% for a four-language transformer. Fifteen points below, free, and covers
every language the encoder does. A register distribution is a descriptive
column, not a gate: the cost of being wrong on one record in four is a slightly
blurred histogram, not a wrong decision about data. State the accuracy.

**Sentiment / emotional tone — do not ship any of the three.** The probe reaches
60.6% on a three-class problem where chance is 48.3%. That is twelve points of
signal, and a sentiment column that is right three times in five is worse than
no column, because a reader will act on it. The transformer reaches 72.5% and
costs the scan 34× its runtime. And sentiment is the facet where the taxonomy
itself is weakest for the use case: "what is the emotional tone of this training
corpus" is a question with a clear answer for product reviews and no clear
answer for a mixture of code, maths and dialogue, which is what a training set
usually is. Emotion at 70.9% over four classes is more promising than sentiment
and is English-only in every public dataset worth training on.

**Train from scratch — no, for all three.** Training inherits the *inference*
cost of a public model without inheriting its training data, and inference cost
is what decides this. The only column where a public model does not exist with
the right taxonomy is task type, and that is exactly the column the probe
already handles. If a from-scratch model is ever built it should be built as a
better *probe head* over the existing embedding — more capacity than a linear
layer, still microseconds per corpus — not as a separate encoder.

### What would change this answer

An ONNX-exported distilled multilingual classifier small enough to run at, say,
5,000 documents a second would move sentiment and formality into range: at that
rate one facet over `atlas-118k` is 24 seconds rather than 641. Nothing published
is close today at the accuracy that would justify it. The other thing that would
change it is running the facet over the *sample* rather than the corpus — the
atlas sample is already the unit coverage is measured on, and a transformer over
20,000 sampled records is 108 seconds. That is still five times the whole scan,
but it is the shape of a `--facets` flag rather than a default.

---

## 7. Pareto: smaller, faster, more accurate, less memory

Four axes that mostly trade against each other. What follows is where this
package actually sits on each, measured, and which moves are available.

### Where the size is

A clean install on CPython 3.14, macOS arm64:

| | installed | share |
| --- | ---: | ---: |
| pyarrow | 127 MB | 40% |
| scipy | 99 MB | 31% |
| numpy | 34 MB | 11% |
| pygments (via rich) | 10 MB | 3% |
| **dropoutt itself** | **10 MB** | 3% |
| tokenizers | 9 MB | 3% |
| hf_xet + huggingface_hub | 15 MB | 5% |
| everything else | 17 MB | 5% |
| **total** | **321 MB** | |

Plus a first-run download that used to be 507 MB and is now 81 MB.

The wheel this project publishes is **5.3 MB**, 8 MB installed of which is data:
7 MB of contamination indices and 2 MB of atlas. Ninety-seven per cent of an
install is other people's code.

**pyarrow, 127 MB, and 28 MB of it is unreachable.** `libarrow_flight` is 23 MB
of RPC and `libarrow_substrait` 5 MB of query plans; a data-quality tool touches
neither. pyarrow publishes no slim wheel, and reading Parquet without it is not
a thing to hand-roll. Nothing to do from here, but it is worth knowing that the
single largest component of the install is 22% dead weight for every consumer
who only wants to read a file.

**scipy, 99 MB, for two calls — and it stays.** Both were rewritten in pure
numpy and measured: 3× slower on the classifier, 4.2× slower on atlas pooling,
with transients of 60 MB and 149 MB respectively where SciPy's CSR kernel
streams. Trading a third of the install for a third of the speed is the wrong
direction on this frontier.

**huggingface_hub cannot be dropped**, because `tokenizers` depends on it. A
hand-rolled downloader for the two files this package fetches would save nothing.

### Where the Python is

"Less dependent on Python methods" is the right instinct and the measurements
say where it pays and where it does not. The scan pass, serial, one core:

| | 1.1 | 1.2 | what moved it |
| --- | ---: | ---: | --- |
| language identification | 23.97 s | 8.75 s | the automaton, read as n-gram counts |
| tier-1 checks | 28.9 s | 28.4 s | unchanged |
| tier-0 checks | 5.74 s | 5.26 s | batched shared features |

What is left is genuinely per-record string work, and numpy has nothing to say
about most of it:

- **MinHash shingling.** 19.8 million memoised BLAKE2b word hashes on this
  corpus. Vectorising means replacing BLAKE2b with a polynomial hash, which
  changes every signature and therefore every near-duplicate count. That is a
  coordinate change, not an optimisation.
- **Normalisation.** NFC, a Turkish-aware case fold, a regex substitution and a
  split — four passes over each record's text, in CPython, per record.
- **The regex families.** Already gated by derived substring tests, which is the
  optimisation; batching the gates was prototyped and does not help, because a
  substring search stops at the first hit and a batch scan therefore costs what
  the per-record scans cost.

The remaining structural win is the one already taken: do the *shared* work by
column, and leave the irreducibly per-record work alone.

### Where the memory is

`atlas-118k`, thirteen workers: peak resident went 3.39 GB → 3.00 GB, and the
atlas phase in a single process 2.30 GB → 1.62 GB. The encoder accounts for most
of that — 1.3 GB resident became about 150 MB, memory-mapped, of which a scan
touches a fifth.

The remaining ceilings are deliberate and stated in their findings rather than
silent: the near-duplicate index stops at a memory-derived capacity, the
contradiction table likewise, and the sample caps are sized from
`hardware.plan()`. The one unbounded term left is the atlas sample itself, which
is capped by count rather than by bytes.

### Where the accuracy is

Three levers, all measured, none free:

| lever | accuracy cost | speed or size gain |
| --- | --- | --- |
| int8 128-column encoder **(taken)** | 0.26% of records change cell | 489 MB → 63 MB, 2.6× on load |
| language head 2,000 → 512 chars (rejected) | 0.53% of records change language | 3% of a scan |
| atlas tokens 512 → 256 (rejected) | 2.1% of records change cell | 11% of one phase |
| vocabulary pruning to 100k rows (rejected) | 0.2% of token occurrences, unevenly by language | 63 MB → 49 MB |

The pattern is consistent: this pipeline is on a steep part of the curve for
anything that shortens the *input*, and on a flat part for anything that
compresses the *model*. Compress the model; do not truncate the text.

### The frontier, stated plainly

- **Size.** 321 MB of dependencies, of which 226 MB is pyarrow and scipy and
  both are load-bearing. The available move was the model cache, and it has been
  made: 507 MB → 81 MB. Nothing else on this list is worth what removing it
  costs.
- **Speed.** 1.28× end to end this release. The next 20% is in the tier-1 checks
  and costs a version bump to the near-duplicate coordinate system.
- **Accuracy.** No accuracy was traded for speed in this release. The one thing
  traded for size — the encoder — moves 0.26% of records by one cell, and the
  report now names the encoder that measured it.
- **Memory.** Bounded rather than merely smaller. Every ceiling says so in the
  finding it affects.
