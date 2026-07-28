# The atlas

## What it is

A **coordinate system**, like latitude and longitude. It contains no notion of
quality and it is not a collection of good datasets. Its only job is to give
every dataset the same bins, so that two fingerprints computed on different
machines can be compared.

```bash
dropoutt atlas
```

## Why a frozen one

Every existing tool refits a UMAP or t-SNE projection separately for each
dataset. That produces pictures that cannot be compared with each other. Freeze
the projection once and a user learns the geography of the map a single time,
then reads any dataset as a heatmap over familiar ground.

The cost is low: once embeddings exist, assignment is one matrix multiply.

## How it is built

`tools/build_atlas.py`, fully reproducible, writing a manifest of exactly which
sources were used and which were unavailable.

1. **Sample a stratified reference corpus.** Roughly thirty public datasets
   spanning the taxonomy, with Turkish and regional content deliberately
   over-weighted. This is the important word: if the corpus mirrored the real
   distribution of the web it would be about 90% English, and the Turkish regions
   would be coarse and useless for exactly the people this was built for.
2. **Embed with a static model.** `potion-multilingual-128M`, 256 dimensions,
   101 languages. Static embeddings are a token-id lookup into a matrix followed
   by a mean pool, so there is no torch and no GPU. Roughly 4,500 records per
   second on a laptop CPU.
3. **Fit level 0: a supervised taxonomy.** About thirty hand-designed categories,
   assigned by a logistic probe over the embeddings with labels bootstrapped from
   dataset provenance.
4. **Fit level 1: k-means within each category**, allocated proportionally to
   category mass.
5. **Name each region from the most frequent words among its members.**
   Deterministic and offline. See [Region labels](#region-labels-the-five-words)
   for what these words are and, more importantly, what they are not.
6. **Project to 2D with PCA.** Deterministic.
7. **Record the quality numbers inside the artifact** so they can be printed
   next to any coverage figure.

Total build cost is a few minutes of CPU and well under $20 of compute even at
full scale.

## Putting your data on it

Coverage is computed on every `dropoutt scan` that has the atlas extra
installed. It appears in the terminal, in `report.html`, and in the `coverage`
facet of `fingerprint.json`.

```bash
dropoutt scan ./my-corpus
```

```
  Atlas coverage (atlas-lite-v0)
    Regions      24 of 258 occupied
    Spread       40% of even coverage  (specialised)
    Off-atlas    0.0%
    Top categories
      general_chat            46%
      code_explanation        31%
      news_current_events     12%
      code_generation          9%
      sql_data                 1%
    Top regions
      160      49  import, return, file, license, version
      161      47  import, return, class, none, name
       62      32  return, şekilde, code, codeprep, kullanarak
```

| line | how to read it |
| --- | --- |
| Regions occupied | how much of the map this corpus touches at all |
| Spread | region entropy against the entropy of a corpus spread evenly. Low is not bad: a specialised corpus *should* be concentrated, and a pretraining mixture should not be. |
| Placed | how many sampled records landed on the atlas. Every share below is over these. |
| Off-atlas | the rest: records too far from every centroid to place. Described, never grounds for withholding the rest. |

`--no-atlas` skips it. If it never appears, run `dropoutt doctor` — coverage
needs the `atlas` extra.

## Comparing two corpora

One corpus on the map is a description. Two is a decision, and that is the
question the atlas exists to answer: **what does this dataset cover that the one
I already have does not?**

```bash
dropoutt scan ./candidate  --out ./fp/candidate
dropoutt scan ./have       --out ./fp/have
dropoutt diff ./fp/candidate/fingerprint.json ./fp/have/fingerprint.json
```

Real output, Python code instructions against Turkish general instructions:

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

Each region prints with five words next to it:

```
  0  film, movie, films, filmi, best
167  select, where, count, show, order
```

**These words are a caption, not a rule.** No record is ever tested against
them. They play no part in placing anything, and deleting them would not change
a single assignment.

### How a record is actually placed

1. The record's text is embedded into a 256-dimensional vector by
   `potion-multilingual-128M`.
2. Cosine similarity is computed against all 258 region centroids.
3. The record joins the nearest one — unless the best similarity falls below the
   off-atlas cutoff of 0.392, in which case it is placed nowhere.

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
least 8 regions** — words that by construction cannot distinguish anything.

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

## Why level 0 is supervised

Two reasons, and both matter.

**Clustering would never produce a Turkish administrative-legal category.** It is
small in a global corpus and large in this market. A hand-designed taxonomy can
contain it; k-means cannot be argued into it.

**The coarse level survives a rebuild.** A refitted k-means changes every bin at
every level, so `atlas-v2` would break comparability with every fingerprint ever
computed. A fixed taxonomy at level 0 means the coarse history still lines up
after a rebuild.

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
| mostly short records | off-atlas median length below 60% of the placed median |
| one coherent group the atlas does not cover | mean pairwise cosine inside the off-atlas set exceeds the placed set by 0.05 |
| concentrated in one dataset or language | one group holds ≥ 60% of the off-atlas records |
| near misses | ≥ 50% of them sit within 0.05 of the cutoff, so it is a threshold effect |
| scattered | none of the above: usually filler, boilerplate or markup rather than a missing topic |

The coherence test separates the two cases worth acting on. Records that resemble
**each other** more than the placed records do are a real subject area missing
from the reference corpus. Records that resemble nothing, including each other,
are junk.

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

`dropoutt atlas` prints two figures that belong next to any coverage number:

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

**One source carries the defect the tool detects.** Some TurkishMMLU records in
`education_pedagogy` are ASCII-folded (`gore`, `asagıdakilerden`), exactly what
`T1-LANG-004` flags.

Read category counts as approximate, and read `general_chat` as "unclassified".
Region assignment and the off-atlas rate are unaffected — those come from
embedding geometry, not from labels.

The fix is per-record annotation rather than per-dataset inheritance: annotate a
sample with a strong model, distil a small annotator, and label each record on
its own content, as Essential-Web did. That requires a rebuild.

## Tiers

Tiers are **resolution levels of one hierarchy**, not separate atlases.
Independent atlases would destroy the property the atlas exists for, because
fingerprints computed against different atlases cannot be compared and the shared
index would fragment. Coarse mass is exactly the sum of the fine masses beneath
it, so a free-tier fingerprint and a paid-tier fingerprint remain comparable and
upgrading re-aggregates existing history rather than invalidating it.

This release ships level 0 plus level 1 in the package:

| property | value |
| --- | --- |
| regions | 258 |
| reference records | 152,622 |
| level-0 categories | 20 (of 31 defined; under-populated ones are dropped and named at build time) |
| level-0 held-out accuracy | 0.864 |
| region purity by taxonomy | 0.785 |
| artifact size | 271 KB |
| off-atlas cutoff | 0.392 cosine |

Measured on 3,000 real Turkish instruction records: **6.3% off-atlas**, 93 of 258
regions occupied, region entropy 3.91 of a possible 5.55.

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
python tools/build_atlas.py --scale 1.0 --out src/dropoutt/data/atlas/atlas-lite-v0.npz
```

`--scale` multiplies every per-source sample target; `--budget` sets a per-source
wall-clock limit so one slow shard cannot stall the build. Sources that have
moved, gone gated or changed split names are skipped and recorded in the
manifest rather than failing the build, which matters when the corpus is
assembled from thirty independently maintained public datasets.
