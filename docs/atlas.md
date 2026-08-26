# The atlas

## What it is

A **coordinate system**, like latitude and longitude. It contains no notion of
quality and it is not a collection of good datasets. Its only job is to give
every dataset the same bins, so that two fingerprints computed on different
machines can be compared.

## Why a frozen one

Every existing tool refits a UMAP or t-SNE projection separately for each
dataset. That produces pictures that cannot be compared with each other. Freeze
the projection once and a user learns the geography of the map a single time,
then reads any dataset as a heatmap over familiar ground.

The cost is low: once embeddings exist, assignment is one matrix multiply.

## How it is built

`tools/build_atlas.py`, fully reproducible, writing a manifest of exactly which
sources were used and which were unavailable. Client and build share one
pipeline library under `dropoutt.atlas` — extraction, chunking, embedding,
normalization — so coordinates stay comparable.

```
ingest → detect format → extract text → chunk → dedup → embed
       → normalize → assign to cells → aggregate
```

1. **Sample a stratified reference corpus** across code, math, instruction/chat,
   legal/finance, scientific, dialogue/forum, structured/tabular, and
   multilingual prose. The v2 build collected 786,180 records from 48 working
   source/configuration pairs: FineWeb, Wikipedia in six languages,
   StarCoderData plus ten explicit The Stack language configs, OpenWebMath,
   FineMath, peS2o, arXiv, PubMed, OpenAssistant, UltraFeedback, Stack Exchange,
   contracts, ECHR case law, finance, and SQL/tabular material. Per-source caps
   keep English, Python, and the densest shards from defining the geometry.
2. **Format-aware extraction.** JSON/CSV/HTML/markdown/code are reduced to
   natural-language content before embedding. `detected_format` is metadata, not
   vector content — otherwise static embeddings collapse into a fake "structured
   data" cluster.
3. **Dedup.** Near-exact MinHash over word shingles, then semantic cosine on
   temporary L2 vectors. Both thresholds are recorded in the manifest.
4. **Embed with `potion-multilingual-128M`.** The fast tokenizer runs once.
   Its flat token-ID cache fits both the unigram table and a CSR document/token
   matrix; one sparse-dense multiply pools all documents with
   `w = a/(a+p)`, `a=1e-3`. The embedding table is subset to observed tokens.
   No per-record Python pooling loop, no torch, no model2vec.
5. **Freeze normalization.** Mean removal, drop top-2 principal components
   (all-but-the-top), L2. Constants ship in the artifact; the client applies
   them and never refits.
6. **Fit a two-level k-means hierarchy** on cosine distance, top-down so lite is
   an exact coarsening of full: 50 L1 regions and 20 children each (1,000 L2
   cells). L2 count is support-gated; a build cannot create a child for every
   300 reference members it does not have.
7. **Label and calibrate.** Each cell carries 12 distinctive terms, 17 distance
   quantiles, eight radial prototype vectors, source/topic/language support,
   and its 16 strongest source-level co-occurrence neighbors. Cells below 200
   direct calibration members retain their local observations and borrow
   residuals from siblings under the same L1 parent.

Every result carries `atlas_version` + `pipeline_hash`. Encoder weights stay on
disk — 63 MB since 1.2, quantised on first use from the 489 MB published file,
which is then deleted — and the artifact stores their content hash, not the
weights.

### Measured v2 build time

Measured on the release machine over 736,966 records after both dedup passes:

| stage | wall time |
| --- | ---: |
| source collection | 1,229.3 s |
| MinHash dedup | 15.3 s |
| tokenize once | 44.4 s |
| fit token probabilities from cached IDs | 1.6 s |
| **SIF sparse embedding** | **23.0 s (32,904 records/s)** |
| semantic dedup | 11.2 s |
| normalization fit | 3.2 s |
| **L1 + L2 clustering** | **5.9 s** |
| labels | 35.5 s |
| v1 population crosswalk | 22.8 s |
| **embedding + normalization + clustering** | **32.1 s** |
| **total wall** | **1,406.6 s (23.4 min)** |

Tokenization, probability fitting, and embedding together took 69.0 seconds.
The geometry training itself (normalization plus both clustering levels) took
9.1 seconds. Collection, not model compute, remains the dominant build cost.

## Putting your data on it

Coverage is drawn by `dropoutt atlas`, which is its own command from 1.3 rather
than a section of the scan report. It writes `atlas.html`, `atlas.md` and
`atlas.json` beside the scan's artifacts.

```bash
dropoutt atlas ./my-corpus
dropoutt atlas ./my-corpus --sample 20000   # a coarser map, sooner
```

Splitting it out was not tidying. Placement runs every sampled record through a
neural encoder — the one part of a scan whose cost had nothing to do with which
checks were enabled — and it needs an 81 MB model that a scan otherwise has no
use for. It also answers a different question. A scan asks what would break a
training run and gives you a list to act on; the map asks where the corpus sits,
and the answer is right or wrong only against a goal the tool has not been told.
Those two things sharing an exit code and a report was the mistake.

`fingerprint.json` still carries a `coverage` facet either way, so two
fingerprints have the same shape and can be compared. A scan fills it with
`not computed by scan (run dropoutt atlas)`.

```
  ◧◨ Where this corpus sits on the map
     atlas-v1-lite

  4,000 records   ·   1 dataset   ·   en 100%

    Effective coverage 26.9 of 215 (29 subregions hold any records) (mixed)
    3,969 of 4,000 sampled records placed · 31 off the map (0.8%) · 0 too short to place

    Density is your share of a subject area against the map's own. 1.0× is parity.
  subject area                              share    density    reach
  Film and television                       22.5%       9.2×      2/5
  Creative writing and fiction              17.7%       6.6×      4/4
  Web development troubleshooting           12.8%       3.9×      1/6
  Digit and counting puzzles                10.7%       6.6×      2/4
  Open-source licensing and file headers     9.0%       3.1×      2/6
    7 further areas reached; 31 of the map's 48 subject areas never reached

    22% of your data sits in a single place on the map
      877 records land there, 0.54 alike on average. Below 0.85 that is a
      subject, not a template: they say the same kind of thing in enough
      different ways to be worth keeping.
      "Known for his Hollywood blockbusters with complex storytelling, Nolan…"

    Film and television — 9.7× denser here than on the map
      The map spends 5 of its 215 places on that subject; 23% of your placed
      records land there. That is what a specialist corpus looks like, and it is
      only a problem if you meant to build a general one.

    Of the 29 places you reach, 9 hold 3% of your data between them
      Real presence in 20 places, a toehold in the rest. An occupancy count
      reads a place holding one record the same as one holding a third of the
      corpus, which is how a narrow corpus comes to look broad.
```

Trimmed: the run also prints what you have most and least of, where you are
farthest from the map in both directions, and the off-map diagnosis.

| line | how to read it |
| --- | --- |
| Effective coverage | two numbers, because occupancy alone is unreadable. The count in brackets says how many subregions hold *any* records, which reads a subregion holding one record the same as one holding a third of the corpus. Effective coverage sums `min(1, density)` over subregions: parity is a full score, thinner coverage a fraction, and over-representation does not count past one. The gap between them is the size of the tail. |
| Placed / off the map / too short | every share below is over the placed records, and all three counts are printed so you can see what the shares are not about. Placement needs at least 80 characters; below that an embedding is noise. |
| Density | your share of a subject area against *the map's* share of the same one. 1.0× is parity. This is the number a histogram of your own data cannot give you: a histogram says what is present, and it takes a fixed coordinate system to say what is absent or thin. |
| Reach | `min(1, density)` summed over that area's subregions, against how many it has. `2/5` means you cover two subregions' worth of five, however unevenly your records are spread across them. |
| What the map says | sentences that clear both a size gate and a significance gate. Nothing appears for being true; it appears for being large *and* true. |
| the quoted record | your own record sitting closest to a region's centre — the only description of a neighbourhood that is true by construction. The atlas's own five-word captions describe the reference corpus, not yours. Suppressed by `--no-evidence`. |
| Off the map | records too far from every centroid to place. Described, never grounds for withholding the rest. |

Nothing here is a verdict. A specialised corpus *should* be concentrated and a
pretraining mixture should not, and the tool has not been told which you are
building.

When a scan covers more than one dataset, a further section reports the cosine
between each pair's region histograms. Two datasets can share no wording and
still occupy the same ground, which is what "we added a third source and gained
no new coverage" looks like from the outside; `T1-OVERLAP-001` compares text and
cannot see it.

### What the atlas still cannot tell you

`atlas-v1-lite` stores `region_size` / `l1_size` for the reference mass, so gaps
can be reported as under-representation against the stratified baseline, not
only as absolute absence. Read that baseline as a property of *this* reference
corpus (topic- and language-capped on purpose), not as a natural population.

If the map cannot be drawn at all, `dropoutt atlas` exits 1 and says why. The
usual cause is an encoder that is not in the cache and cannot be downloaded: run
`dropoutt fetch` first, then `--offline` works everywhere.

## Comparing two corpora

One corpus on the map is a description. Two is a decision, and that is the
question the atlas exists to answer: **what does this dataset cover that the one
I already have does not?**

Place both and compare what came back. `atlas.json` carries the full region
histogram under `atlas.subject_areas[].cells`, which is what makes two runs
comparable at all:

```bash
dropoutt atlas ./candidate  --out ./maps/candidate
dropoutt atlas ./have       --out ./maps/have
```

What that comparison looks like, Python code instructions against Turkish
general instructions:

```
  Atlas comparison
    Similarity   0.02  (1.0 = same distribution over regions)
    Shared       38% of left sits in regions right also occupies
    New          62% of left sits in regions right never reaches

    Only in left — what adding it would bring
      151    12%  import, python, return, data, create
      157    11%  return, function, list, write, given
      149     9%  return, function, write, given, else

    Only in right
       94     5%  yardımcı, nasıl, şekilde, olabilir, sahip
       98     4%  makine, algoritma, öğrenimi, oluşturun, etmek

    Category mix
category            left    right    delta
code_generation      94%       0%     +94%
general_chat          3%      95%     -92%
```

**The comparison is directional**, read left against right, for the same reason
cross-dataset overlap is. A small specialised corpus can sit wholly inside a
large one while the large one is barely inside it; a symmetric score hides
exactly the case worth acting on. Swap the arguments to ask the other question.

**A partial side is carried, not refused.** Every number is over the records each
side actually placed, and both placed shares are printed. A high off-atlas rate on
the **right** side biases novelty in one direction only: regions the right side
appears not to reach may in fact be reached by records it could not place, so the
`New` figure is an upper bound. `diff` says so rather than refusing:

```
  note the right side placed only 62% of its records, so regions it appears not
  to reach may be reached by records it could not place. Read 81% new as an
  upper bound
```

It still refuses in two cases: when the two fingerprints were computed against
different atlas versions, where region ids do not refer to the same regions, and
when one fingerprint was written before 0.1.4 with its histogram already
discarded. Re-scanning fixes the second.

**It does not rank datasets.** `New 62%` is geometry. Whether new coverage helps
depends on what you are training, which the tool does not know.

## Region labels: the five words

> The measurements in this section and in
> [Reading the quality numbers](#reading-the-quality-numbers) were taken on the
> 258-region build that preceded `atlas-v1-lite`. They are kept because the
> failure modes they describe are properties of the *method*, which has not
> changed, and because a measurement is worth more than a description of one.
> The shipped artifact has 215 regions and twelve label slots each; the one
> number re-measured against it is noted below.

Each region prints with five words next to it:

```
  0  film, movie, films, filmi, best
167  select, where, count, show, order
```

**These words are a caption, not a rule.** No record is ever tested against
them. They play no part in placing anything, and deleting them would not change
a single assignment.

### How a record is actually placed

1. Format-aware extraction pulls natural-language content (keys/syntax dropped).
2. The text is embedded by `potion-multilingual-128M` with SIF pooling, then
   corrected with the frozen mean/PCA/L2 constants. Since 1.2 the encoder is
   stored as its first 128 columns at one byte per weight with a per-row scale —
   63 MB instead of 489 MB — and the atlas is fitted in that quantised
   coordinate system, on the same reference corpus as before, so the encoder a
   scan applies is the encoder the map was built with. The report names it in
   `atlas.identity.encoder_weight_hash`; if a scan ever applies an atlas
   through weights it was not fitted on, the fitted hash is kept alongside as
   `encoder_built_with` so the report says so.
3. Cosine similarity is computed against all fine (L2) centroids. Soft
   assignment keeps the top-5 with a temperature tuned so a typical document
   holds weight on ~2–3 regions; the hard nearest cell still drives the
   histogram.
4. Below the off-atlas cutoff the record is placed nowhere. Coarse subject area
   is the parent L1 cell — a strict coarsening of the fine map, not a second
   model.

Word overlap is not consulted at any point. Four real placements:

| text | region | contains how many of the 5 label words |
| --- | --- | --- |
| "Yesterday I watched a three hour epic about a submarine crew…" | 0 `film, movie, films, filmi, best` | **none** |
| "Bu akşam sinemaya gidip yeni çıkan bilim kurgu yapımını izledik…" | 0 `film, movie, films, filmi, best` | **none** |
| "SELECT customer_id, SUM(total) FROM orders GROUP BY…" | 167 `select, where, count, show, order` | 2 |
| "If a train leaves the station at 60 km/h and another at 90 km/h…" | 134 `hours, minutes, hour, miles, total` | **none** |

The first two are the point. An English sentence about a submarine film and a
Turkish sentence about a science-fiction film land in the *same* region, sharing
no vocabulary with the label or with each other. That is the embedding doing the
work.

### Where the labels come from

After clustering, the **first 150 members in corpus order** — not a random 150 —
are word-counted, non-letters are stripped from inside each word, words of three
characters or fewer are dropped, an English and Turkish stoplist is applied, and
the five most frequent survivors become the label.

The stoplist has 38 entries but **only 16 of them do anything**: the other 22
(`the`, `and`, `bir`, `ve`, `bu`, …) are three characters or fewer and were
already removed by the length filter one step earlier.

### Why they read like random words

Because that method is weak, and measurably so. Three defects compound:

**No inverse-document weighting.** Frequency is counted within a region, not
against the other regions, so a word that is common *everywhere* still floats to
the top. Only those 16 effective stopwords hold it back. `their` appears as a
label word in **33 of 258 regions**, `they` in 21, `about` in 18. Across the whole
atlas, **21.6% of the 1,290 label slots are filled by a word that appears in at
least 8 regions** — words that by construction cannot distinguish anything. On
the shipped 215-region artifact the same measurement is 14.1% of 2,580 slots:
better, and still one slot in seven spent on a word that separates nothing.

**No lemmatisation, and Turkish is agglutinative.** Inflections of one stem are
counted as separate words and eat multiple slots. **21% of regions spend two or
more of their five slots on the same stem:**

```
  0  film, movie, films, filmi, best          → 3 slots, one concept
  7  cümle, cümlenin, cümleyi, adım, doğru    → 3 slots, one concept
 15  veri, verilen, oluşturun, verileri, verin → 3 slots, one concept
```

**The 150 sampled members are the first 150, not a random 150.** In corpus order
that is often one source file, so a large region can be named after whichever
dataset happened to be read first.

Between the generic words and the duplicated stems, roughly 40% of the label
text carries no information. The regions are real; their captions are poor.

### What this does and does not affect

| affected | not affected |
| --- | --- |
| how readable a coverage report is | which region a record lands in |
| whether you can guess a region's topic from its name | off-atlas rate |
| how easy the atlas is to review by hand | region entropy, coverage counts, fingerprint comparability |

Every number the atlas produces is computed from centroids and assignments.
Relabelling would change none of them.

### The planned fix

Score words by frequency inside the region against frequency across all regions,
lemmatise before counting, and sample members randomly rather than taking a
prefix. Production would name regions with an LLM, as Essential-Web did. All of
this requires a rebuild, because member texts are not stored in the artifact —
only centroids are.

## Why the coarse level is a hierarchy prefix, not a second model

v1 drops the supervised taxonomy probe. L1 is k-means over the same vectors as
L2, fitted first; L2 is k-means *within* each L1 membership. Lite reports are
therefore exact unions of fine cells — they cannot contradict the full map.

Topic and language breadth still come from **stratified sampling** of the
reference corpus (math held separate from academic prose, instruction/chat as
its own mass, legal/finance capped in, Turkish and other languages over-weighted
relative to the web), not from a classifier trained on dataset provenance.

## Why language is not a clustering axis

Multilingual embeddings separate partly by language, so a flat k-means over a
multilingual corpus can spend much of its region budget distinguishing Turkish
from Arabic from Chinese rather than distinguishing topics. At 256 regions that
would consume the entire map.

An earlier design solved this by neutralising language geometrically: computing a
mean embedding per detected language, subtracting it, and projecting out the
components that predict language identity. **That was rejected**, for three
reasons recorded here so the decision is not casually revisited.

1. It conditions the geometry on a label that is least reliable exactly where
   this tool needs it most. Language identification is weakest on short text and
   on closely related languages, which is precisely the Turkish, Azerbaijani,
   Turkmen and Ottoman cases. A misidentified record has the wrong centroid
   subtracted and lands somewhere meaningless.
2. A language centroid does not encode only language. Turkish web text is not
   translated English web text; it has a different topical distribution.
   Subtracting its mean removes part of what Turkish corpora are *about*.
3. It is not inspectable. When a user asks why a record landed in a region, the
   honest answer would involve a hidden vector subtraction they cannot examine.

The adopted approach conditions on topic through **supervision** and alters
nothing: fine clustering is fitted within each level-0 category, so once you have
conditioned on topic there is much less room left for language to dominate. Any
language splitting that survives inside a category is visible and explicable
rather than erased.

Coverage is therefore reported as **category by language** and **region by
language**, and language remains its own fingerprint facet, measured by
identification rather than by clustering.

### The Ottoman case

Ottoman Turkish written in Arabic script gets its language and script from the
language facet. Its content — legal, administrative, poetic — classifies into the
corresponding level-0 category. So Ottoman legal text and modern Turkish legal
text occupy the **same category with different language tags**, which is what
makes a marginal-contribution comparison meaningful: this corpus adds language
coverage without adding topical coverage, or the reverse.

Under an unfactorised atlas the two would be separated by script alone and the
topical relationship would be invisible.

## Records too short to place

A record below 80 characters is **excluded from placement**, not assigned. Its
embedding is dominated by noise, for the same reason language identification is
gated on length. Including such records would inflate the off-atlas rate with
records that were never placeable in the first place.

The number excluded is reported alongside the coverage figures. On a typical
short-form instruction corpus this can be most of the records, and that is worth
knowing rather than hiding: it means coverage describes the long tail of your
data, not all of it.

## Off-atlas data

A record is **off-atlas** when its cosine similarity to the nearest centroid
falls below a threshold calibrated at build time, currently 0.392. Those records
are excluded from the region histogram and the category counts, so every share
the report prints is a share of the **placed** records, and the placed count is
printed beside it.

Until 0.1.4, an off-atlas rate above 10% discarded the whole coverage report and
printed a sentence saying the numbers had been withheld. That was wrong twice
over. The histogram never contained off-atlas records to begin with — they are
filtered out before counting — so withholding it threw away a measurement that
was correct for every record it covered. And the off-atlas set is the most useful
thing the atlas produces on a corpus that does not fit it: it is a list of the
records unlike anything in the reference corpus.

So the report describes them instead:

```
    Off-atlas    11.0%  44 records (the atlas covers most of this corpus)
      Why: mostly short records: the off-atlas half has a median of 120
      characters against 363 for the placed half. Similarity to a region rises
      with length, so short records read as off-atlas whatever they are about
      Distance    off-atlas median 0.27 similarity, cutoff 0.39, placed median 0.59
      Nearest regions despite missing the cutoff, spread over 17 regions
         13       7  cümle, entryway, bench, kimya, önemli
          9       5  cevap, frac, şimdi, adım, equation
      Off-atlas rate by language
        unknown                  39% (18 of 46 records)
        tr                        8% (19 of 235 records)
      Furthest from the atlas
        0.11  N/A no yes N/A N/A no yes N/A N/A no yes N/A N/A no yes N/A
```

The fit is graded rather than passed or failed, because the underlying quantity
is continuous and a corpus at 10.1% is not meaningfully different from one at
9.9%:

| rate | fit | what it means |
| --- | --- | --- |
| ≤ 10% | good | the atlas covers this corpus |
| 10–35% | partial | the atlas covers most of this corpus |
| > 35% | poor | the atlas covers a minority of this corpus |

Ten percent is not arbitrary. The cutoff was set at the 2nd percentile of the
atlas's own reference records, so a corpus drawn from the same distribution as
the atlas sits near 2%. Ten percent is five times that.

### Read the off-atlas rate as length first

This is measured, not assumed. Similarity to the nearest centroid rises steeply
with record length. The same English paragraph scores **0.363 truncated to 20
characters and 0.787 at 2000**, landing in the same region throughout. Across a
real corpus the correlation between log length and similarity is about **0.49**,
and the off-atlas rate falls from **33% for records under 80 characters to 0%
above 150**.

A high off-atlas rate is therefore a statement about record length first,
language second, and topic only third. The `Why:` line attributes it in that
order rather than letting you assume the third. The causes it distinguishes:

| diagnosis | how it is decided |
| --- | --- |
| not written like prose | off-atlas whitespace share below half the placed share, or non-letter share more than 0.15 above it |
| mostly short records | off-atlas median length below 60% of the placed median |
| one kind of thing, not scattered | mean pairwise cosine inside the off-atlas set exceeds the placed set by 0.05, and the surface test did not fire |
| concentrated in one dataset or language | one group holds ≥ 60% of the off-atlas records |
| near misses | ≥ 50% of them sit within 0.05 of the cutoff, so it is a threshold effect |
| scattered | none of the above |

#### Why coherence alone is not enough

Coherence — how much the off-atlas records resemble **each other** — sounds like
it should identify a missing subject area. It does not, and the reason is
measured. Against this atlas:

| off-atlas set | coherence | what it is |
| --- | --- | --- |
| minified JavaScript | 0.969 | template |
| HTML boilerplate | 0.961 | template |
| DNA strings | 0.947 | template |
| Ottoman endowment-deed vocabulary | 0.886 | **a genuinely missing topic** |
| hex log lines | 0.875 | template |
| base64 blobs | 0.871 | template |
| real English prose | 0.277 | the baseline |

Every machine format scores far above prose, and the one real missing topic sits
in the middle of them. **High coherence means the records are alike and nothing
more.**

What separates them is how the text is written. Whitespace share runs 0.158 for
prose and 0.132 for the missing-topic case, against 0.000 for base64 and DNA,
0.037 for HTML and 0.041 for minified JavaScript. Non-letter share runs 0.048 for
prose and 0.000 for the missing topic, against 0.191 for base64, 0.395 for
minified JavaScript and 0.556 for hex logs. Either test alone leaves a gap;
together they caught all six machine formats and neither prose case.

So the report prints both numbers and says "not written like prose" when the
surface test fires, and reserves the coherence reading for the case where the
surface looks like prose. Even then it stops at what was measured and points at
the nearest-region words, which are what actually name the subject.

#### Off-atlas is not the garbage detector

This is worth stating plainly, because the new output invites the opposite
reading. Machine formats usually **place**, confidently and wrongly, rather than
going off-atlas. On a corpus of 400 records where 100 were base64 blobs and
minified JavaScript, the off-atlas count was **zero** — the blobs landed in
regions labelled `return, denklemin, array, tdrow, function` and `data, should,
technology, provide, their`.

Those 100 records were caught, but by `T1-LANG-001` (language composition and
detection confidence), which flagged exactly 100 of 400. The encoding and
degeneracy checks are the instrument for junk. The atlas is a coordinate system,
and a coordinate system will happily give nonsense a coordinate.

The "not written like prose" diagnosis therefore fires only when machine-format
records *also* happen to fall below the cutoff, which is a narrower case than it
sounds. When it fires it is right; it is not a substitute for the checks.

### What off-atlas does not mean

It is not a quality score, and it does not run in the direction you might guess.
Measured against this atlas: a base64 blob scores **0.441 and places on-atlas**,
in a region of Turkish history. A string of nothing but the letter A scores
**0.538**. A real Turkish sentence about training data scores **0.315 and goes
off-atlas**. The cutoff separates *typical* from *atypical*, not *good* from
*bad*, and short or non-English text is atypical whatever it says.

The rate is reported **per language as well as globally**. For a language the
embedding model represents poorly, topical assignment is unreliable no matter how
good the clustering is, and a global average would hide that.

## Reading the quality numbers

Two figures belong next to any coverage number, and both travel in the
`coverage` facet of every fingerprint:

| number | meaning |
| --- | --- |
| level-0 held-out accuracy | how well the taxonomy probe generalises. Low accuracy means category counts look precise and are not. |
| region purity by taxonomy | mean share of each region occupied by its most common category. Low purity means regions are mixing topics. |

### What 0.864 accuracy does not mean

It measures how well the probe reproduces the **provenance labels** it was
trained on. It does not measure whether those labels are correct.

In v0 the level-0 label of every reference record is inherited from the dataset
it came from. Where a dataset is topically narrow that works. Where it is not, a
high accuracy means the probe faithfully learned a wrong taxonomy. Three
consequences are visible in the shipped artifact and are stated here rather than
left to be discovered:

**`general_chat` holds 106 of 258 regions.** UltraChat, Alpaca, Dolly and four
Turkish instruction sets were all labelled `general_chat`, but instruction
datasets span every topic there is. The probe learned "general_chat" to mean
"came from an instruction dataset" rather than any subject. Its regions
therefore include film, colour theory, poetry, blockchain, football and code —
things that have proper categories elsewhere in the taxonomy.

**Two categories are mislabelled outright.** `summarization` (regions 106–115)
is `tr-wikihow-summ`, whose records are how-to instructions, not summaries —
its label words are `tıkla, dokun, ekranın` (click, tap, screen).
`religion_philosophy` (219–225) is Arabic Wikipedia, mapped there at build time;
two of its regions are about languages and computers.

Read category counts as approximate, and read `general_chat` as "unclassified".
Region assignment and the off-atlas rate are unaffected — those come from
embedding geometry, not from labels.

The fix is per-record annotation rather than per-dataset inheritance: annotate a
sample with a strong model, distil a small annotator, and label each record on
its own content, as Essential-Web did. That requires a rebuild.

## Tiers

Tiers are **resolution levels of one hierarchy**, not separate atlases.
Lite (L1) is a strict prefix of full (L2): every fine cell has one immutable
parent. Fingerprints against lite and full stay comparable; upgrading
re-aggregates rather than invalidating.

The package ships one bundle, `atlas-v1-lite.npz`, carrying both levels.

Every figure below is read from the shipped artifact's own metadata; the fuller
build record is in `tools/atlas-data/atlas-lite-v3-release-notes.json`, filed
under the name the bundle was built as.

| property | value |
| --- | --- |
| L1 regions (lite) | 48 |
| L2 fine cells | 215 |
| reference records (after both dedup passes) | 2,125,556 |
| distinct sources | 102 |
| embedding | potion-multilingual-128M, first 128 columns, int8 per-row scale, SIF pool |
| normalization | per-language mean + top-2 PCA removed + L2 |
| soft-assign | top-5, T=0.08 (2.57 regions with weight > 0.15) |
| topic purity (macro / micro) | 0.540 / 0.544 |
| source purity (macro / micro; lower is better) | 0.298 / 0.295 |
| source cluster AMI | 0.252 |
| directly calibrated cells (≥200 members) | 215 of 215 |
| non-English share | 31.1% |
| artifact size | 1.08 MB |
| off-atlas cutoff | 0.277 cosine |

Every L2 cell clears the 200-member calibration floor directly, so none of them
falls back to its L1 parent's residuals. The fallback path still exists, and the
direct support and reliability flag travel with the artifact either way.

The topic/source diagnostic does not compare raw NMI values directly: a
language-specific source such as German Wikipedia makes source identity and
language identical. Source purity is well below topic purity, which is the
direction that matters — cells group by subject rather than by where the text
came from. L1 exemplar review shows recognisable regions for clinical
medicine, legal agreements, finance, SQL, mathematics, machine learning,
biology, sports, and code. Some intentionally distinct registers remain visible
(assistant dialogue, licences, and task-formatted instructions); format syntax
itself is stripped before embedding.

Superseded bundles are no longer installed. They live in `tools/atlas-data/`
for rebuilds and for reading old fingerprints, and a wheel carries only
`atlas-v1-lite.npz`, so a scan cannot silently report coordinates from a map
other than the pinned one.

## Hand intervention

Human judgement is used where it has leverage and nowhere else.

| level | who decides |
| --- | --- |
| level 0, ~30 categories | designed by hand |
| level 1, 256 regions | clustered, then reviewable by hand |
| deeper levels | unsupervised; 16,384 regions are not reviewable |

Any hand edit must be declarative and versioned, so the atlas stays a
reproducible function of the reference corpus, the embedding model and that file.
Edits applied directly to the artifact would make it impossible to rebuild.

Level-0 category ids are **append-only and never renumbered**, because they are
part of the fingerprint schema.

## Rebuilding

```bash
python tools/build_atlas.py \
  --scale 4.0 \
  --budget 180 \
  --out src/dropoutt/data/atlas/atlas-v1-lite.npz
```

`--scale` multiplies every per-source sample target; `--budget` sets a per-source
wall-clock limit so one slow shard cannot stall the build. Sources that have
moved, gone gated or changed split names are skipped and recorded in the
manifest rather than failing the build. A JSON timing log is written next to the
artifact (`build-timing.json`) with collect / tokenize / embed / cluster wall
times. `build-diagnostics.json` records every L1 label, exemplar, source
concentration, topic mix, format mix, language mix, and calibration support.
The build fails if the useful compressed artifact falls outside 3–5 MB; it does
not add padding to meet the lower bound.
