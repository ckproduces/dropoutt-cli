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

`tools/build_atlas_v2.py` builds the map from one read-only corpus cache (the
file is named for the generation it was written for and builds atlas-v3). `tools/run_atlas_v3_build.sh` holds the exact invocation for
atlas-v3 and copies the finished artifact into the package with its checksum;
the build's gates are recorded in
`src/dropoutt/data/atlas/atlas-v3-release-notes.json`. Client and build share
one pipeline library under `dropoutt.atlas` — extraction, chunking, embedding,
normalization — so coordinates stay comparable.

```
manifest → extract text → dedup → detect language → embed
         → fit normalization → L1 k-means → L2 k-means per L1
         → calibrate and name → artifact
```

1. **Read a frozen manifest.** The atlas-v3 corpus is 244 manifest sources
   over nine axes — web, encyclopedic, books, legal and government,
   scientific, training, code, forum, educational — each with a byte target,
   plus a non-English floor. 166,324,909 input records; 163,452,464 retained
   after filtering, 212.8 GB, 60.6% non-English by bytes. The corpus hash
   `4203fc3a0858de369f83763613f1911b` is stamped into the artifact. The cache
   is never modified, so a rebuild reads the same bytes.
2. **Format-aware extraction.** JSON/CSV/HTML/markdown/code are reduced to
   natural-language content before embedding and truncated to 2,000
   characters. The builder keeps records of 80 characters or more; placement
   at run time needs 40 (see [Records too short to
   place](#records-too-short-to-place)). `detected_format` is metadata, not
   vector content — otherwise static embeddings collapse into a fake
   "structured data" cluster.
3. **Dedup.** Exact duplicates are dropped by a 64-bit hash of the lower-cased
   text, held in a disk-backed set across all 166 million rows.
4. **Detect the language of every record.** Until v3 a record's language was
   whatever its source declared, inherited by every row in the shard. That is
   wrong often enough to matter: an earlier v3 candidate partitioned
   Ukrainian text into cells named Serbian and Bulgarian because the shard
   said so, and the per-language centering then subtracted the wrong mean.
   The byte n-gram detector now runs on every record beside the encoder
   (`language_labels_source` is `detected-per-record:dropoutt.langid`), and a
   record keeps its declared language only when the detector will not commit.
5. **Embed with `potion-multilingual-128M`, through the encoder input policy.**
   The tokenizer runs once and one sparse-dense multiply pools each document
   with SIF weights, fitted in a first pass over a stratified sample of the
   manifest. Before tokenizing, runs of ALL-CAPS words are sentence-cased and
   table rules are dropped; inside the pool, one- and two-character cased
   pieces, symbol-only and digit-only tokens, and every token of a template
   phrase count for a fifth or a tenth of their SIF weight. See [What the
   encoder reads](#what-the-encoder-reads). Vectors land in one float16 store
   on disk that both clustering and the off-atlas calibration read from.
6. **Freeze normalization.** One mean per language, accumulated over every row
   of that language — 59 languages cleared the 6,000-row floor — with the
   global mean for the rest; then the top-2 principal directions removed, then
   L2. The global mean and the PCA are fitted on a 2,000,000-row draw balanced
   over (language, axis) strata rather than a proportional one, because a
   proportional draw is 53% English web and its principal directions were
   English-web-shaped. A linear probe on 300,000 held-out rows is recorded in
   the artifact: balanced accuracy for language falls from 0.528 on raw
   vectors to 0.365 after normalization, for axis from 0.697 to 0.580. Most
   of the language signal goes; most of the subject signal stays.
7. **Fit a two-level spherical k-means, weighted by content.** 256 L1 regions
   on a 2,000,000-row draw taken at even steps of *text length* rather than of
   row count, then every record assigned to its nearest L1 centroid. A record
   weighs the characters the encoder reads, up to 2,000, so the corpus's
   one-line prompts — a fifth of its rows — shape about as much of the map as
   the text they contain, not a fifth of it. Budgets, the L2 fits and the
   centroid means use the same weights; `region_size` and every density stay
   in records (`fit_weights: length` in the artifact). The 4,096-cell L2
   budget is split across regions as content to the power 0.75 (recorded as
   `sqrt_population_budget`): proportional
   allocation starves the small regions worth telling apart — code,
   mathematics, law — and a square root swings cell populations by 11×, so the
   exponent sits between. Each region is fitted with its allotted k (1 to 64
   allowed; the shipped regions hold between 6 and 32 cells), sibling cells at
   or above 0.95 cosine are merged (none were in this build), and every
   centroid is recomputed from all of its members.
8. **Calibrate and name.** Each cell carries 50 distance quantiles, 8 radial
   prototype vectors, 64 contrastive terms, and its 32 strongest source-level
   co-occurrence neighbours. Names come from
   `tools/atlas-data/l1_labels_atlas-v3.json` and
   `tools/atlas-data/region_labels_atlas-v3.json`, written by hand and keyed
   to the corpus hash; see [Hand intervention](#hand-intervention). The
   off-atlas cutoff is stamped afterwards by `tools/calibrate_off_atlas_v3.py`
   — see [How the cutoff is calibrated](#how-the-cutoff-is-calibrated).

Every result carries `atlas_version` + `pipeline_hash`. Encoder weights stay on
disk — the published 489 MB float32 table is quantised on first use to one byte
per weight with a per-row scale, 142 MB in
`~/.cache/dropoutt/embedder/potion-multilingual-128M/` including the 18 MB
tokenizer, and the original is deleted — and the artifact stores their content
hash, not the weights.

What ships is 13.8 MB compressed: centroids, reference sizes, normalization
constants (global mean, 59 language means, two PCA directions), distance
quantiles, prototype vectors, co-occurrence neighbours, the IDF table, term
lists and names. The reference-record excerpts the builder writes for review
(`exemplar_texts`) are not in the wheel. That is also why the report quotes
your own records and never the map's.

The whole v3 build ran 17,397.5 s (4.8 h); the release notes record only the
total.

### What the encoder reads

The encoder is a static table: a record's vector is the weighted mean of its
token rows, and nothing in that mean knows what a token means in context. So
anything that fills a record with distinctive tokens without saying what the
record is about decides where it lands. An audit of the first atlas-v3 (13
September 2026) found four such things that had drawn cells of their own, each
confirmed by removing the feature from a cell's members and placing them again:

| feature | example | what removing it did |
| --- | --- | --- |
| ALL-CAPS text | `INFORMATIONEN ÜBER DATENSCHUTZERKLÄRUNG` tokenises into 13 capital-letter fragments and no word | sentence-casing moved every member of a "shouted web copy" cell to a subject cell |
| table rules | pipe-delimited infoboxes, `---` separators | removing pipes moved 85% of a "tables about anything" cell |
| one- and two-letter pieces | initials and the first fragment of rare names (`▁T`, `▁P`) | districts that shared only a first letter; removing each cell's heaviest pieces moved about half its members |
| instruction templates | "generate a more complex version of this sentence", in fifty translations | ten districts split by the template's language, not by what the sentences said |

The encoder input policy (`dropoutt.atlas.textnorm`, version 1) is the frozen
answer. Before tokenizing, runs of three or more ALL-CAPS words are
sentence-cased — so `NASA` in a sentence stays `NASA` — and table rules become
spaces. Inside the pool, cased pieces of one or two characters, symbol-only
tokens and digit-only tokens keep a fifth of their SIF weight; scripts without
letter case are untouched, because a one-character token in Chinese or Japanese
is a word. And every token inside a *template phrase* keeps a tenth: a
four-token phrase is a template when it occurs in at least 10% of one source's
sampled documents. Measured on the corpus, the rewriting template sits at
37–83% of its source; the most repeated phrases of ordinary prose stay under
15%, and a subject phrase such as "is a village in" at 4.6%. The builder finds
these phrases in the same pass that fits the IDF table and ships them in the
artifact (`encoder_phrase_hashes`).

The policy is part of the coordinate system. The artifact declares it
(`encoder_input`), and a scan binds the same policy to the encoder through
`Atlas.bind_embedder`, so no record is read differently from the records the
cells were drawn from. Maps built before the policy declare none and keep
reading raw text.

Measured on a 5-million-record pilot built through the same code, the policy
cut template cells from 68 to 9 and, with content-weighted fitting, to 2;
records clustering by language fell (cell–language NMI 0.093 → 0.080, the
normalised language probe 0.30 → 0.20). It costs some separation of *sources*
— the 52-source held-out panel's source AMI fell from 0.284 to 0.268 — because
formatting that used to tell sources apart no longer places records. That is the
trade the map is for.

### A build timed stage by stage

The stage-by-stage timing below was measured on the 786,180-record build of
the v2 pipeline — 48 working source/configuration pairs, 736,966 records after
both dedup passes — that preceded atlas-v2. It was taken with the earlier
`tools/build_atlas.py`, whose MinHash, semantic-dedup and crosswalk stages the
v3 build does not run, and it is kept as the one stage-by-stage timing this
document has. atlas-v2 was later refitted on 69,071,324 records from 59
sources, and atlas-v3 on 163,452,464 from 244.

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
dropoutt atlas ./my-corpus --sampling 500      # a quick look
dropoutt atlas ./my-corpus --sampling 0        # every record
```

One map ships, so there is nothing to choose: the command places on atlas-v3
in a terminal and in a CI job alike. `--model atlas-v3` and `atlas = "atlas-v3"`
under `[scan]` in `dropoutt.toml` both still parse, so a run can say which map
it meant; any other name is a usage error rather than a silent fallback. A map
that moves cell ids ships under a new product name, so two runs that name the
same map are on the same coordinate system. `--sampling` defaults to 500,000
records.

Splitting it out was not tidying. Placement runs every sampled record through a
neural encoder — the one part of a scan whose cost had nothing to do with which
checks were enabled — and it needs a 142 MB encoder that a scan otherwise has
no use for. It also answers a different question. A scan asks what would break
a training run and gives you a list to act on; the map asks where the corpus
sits, and the answer is right or wrong only against a goal the tool has not
been told. Those two things sharing an exit code and a report was the mistake.

`fingerprint.json` still carries a `coverage` facet either way, so two
fingerprints have the same shape and can be compared. A scan fills it with
`not computed by scan (run dropoutt atlas)`.

The bundled `tests/fixtures/messy` on atlas-v3, trimmed:

<!-- transcript:start -->
```
  ◧◨ Where this corpus sits on the map
     atlas-v3

  332 records   ·   5 datasets   ·   tr 100%

    Effective coverage 13 of 4,096 (13 subregions hold any records) (specialised)
    250 placed of 321 sampled records; 2 were too short to place · 71 off the map (22.1%)

    Each subject area you reached, then the subregions inside it. Density is your share of a 
subregion against the reference corpus's share of the same one: 1.0× matches the map.

    Names and entries starting with V  2/15 reach · 54.4% share · 136 records
         49×  Taxonomy entries and medical t…           18×  Multiple-choice sentence-compl…  

    Sentence-rewriting prompts over institutional history  1/16 reach · 18.4% share · 46 records
         23×  Sentence-rewriting prompts abo…      

    Websites, cookies and web hosting  1/16 reach · 16.0% share · 40 records
         21×  Articles about blogging platfo…      

    Mathematics papers, proofs and theorems  1/18 reach · 6.4% share · 16 records
        8.8×  Articles about measurement uni…      

    Commission regulations on export refunds and prices  3/20 reach · 1.6% share · 4 records
        2.0×  EU regulations on customs impo…          1.4×  EU court case filings and lega…  
        1.5×  Pre-2004 EEA Joint Committee d…                                                 

    Essay writing, research papers and author guidelines  1/15 reach · 1.6% share · 4 records
        2.9×  Forum advice about writing res…      

    Chatty personal posts and forum confessions  1/16 reach · 0.4% share · 1 record
        1.5×  Forum confessions about health…      

    Materials, metals and industrial surfaces  1/17 reach · 0.4% share · 1 record
        1.5×  Product descriptions for home…      

    2 further areas reached; their subregions are named in the report files

    246 of the map's 256 subject areas never reached

    40% of your data sits in a single place on the map
      Records there are 1.00 alike, which is one thing written out many times rather than one 
subject covered many ways. Near-duplicate detection will not catch it: they share almost no wording.
      “Türkiye'nin 114. en kalabalık şehri hangisidir? Bu sorunun cevabı 114 numaralı şehirdir. 
Detaylı açıklama: veri veri veri veri veri veri veri veri veri veri ver”

    Names and entries starting with V — 48× denser here than on the map
      The map holds 0.4% of its reference text in that subject (15 of its 4,096 places); 54% of your
placed records land there. That is what a specialist corpus looks like, and it is only a problem if 
you meant to build a general one.

    Sentence-rewriting prompts over institutional history — 16× denser here than on the map
      The map holds 0.4% of its reference text in that subject (16 of its 4,096 places); 18% of your
placed records land there.

    Websites, cookies and web hosting — 14× denser here than on the map
      The map holds 0.4% of its reference text in that subject (16 of its 4,096 places); 16% of your
placed records land there.

    Of the 13 places you reach, 7 hold 3% of your data between them
      Real presence in 6 places, a toehold in the rest. An occupancy count reads a place holding one
record the same as one holding a third of the corpus, which is how a narrow corpus comes to look 
broad.
```
<!-- transcript:end -->

Trimmed: the run goes on to print what you have most and least of, where your
mix differs most from the map's in either direction, and the off-map
diagnosis.
The fixture is deliberately broken — the place holding 40% of the data is one
template, 1.00 alike — which is why the map reads as it does.

| line | how to read it |
| --- | --- |
| Effective coverage | two numbers, because occupancy alone is unreadable. The count in brackets says how many subregions hold *any* records, which reads a subregion holding one record the same as one holding a third of the corpus. Effective coverage sums `min(1, density)` over subregions: parity is a full score, thinner coverage a fraction, and over-representation does not count past one. The gap between them is the size of the tail. |
| Placed / off the map / too short | every share below is over the placed records, and all three counts are printed so you can see what the shares are not about. Placement needs at least 40 characters; below that an embedding is noise. |
| Density | your share of a subregion against *the map's* share of the same one, where the map's share is its reference records in that cell over all 163,452,464 (`region_size`). 1.0× is parity. This is the number a histogram of your own data cannot give you: a histogram says what is present, and it takes a fixed coordinate system to say what is absent or thin. A cell holding one sampled record is shrunk toward parity rather than printed as a raw quotient, because one record in a rarely-used cell is a coin flip, not a 40× density. |
| Reach | `min(1, density)` summed over that area's subregions, against how many it has. `2/15` means you cover two subregions' worth of fifteen, however unevenly your records are spread across them. |
| What the map says | sentences that clear both a size gate and a significance gate. Nothing appears for being true; it appears for being large *and* true. |
| the quoted record | your own record sitting closest to a cell's centre — the only description of a neighbourhood that is true by construction. The cell's hand-written name describes the reference corpus, not yours. Suppressed by `--no-evidence`. |
| Off the map | records too far from every centroid to place. Described, never grounds for withholding the rest. |

Nothing here is a verdict. A specialised corpus *should* be concentrated and a
pretraining mixture should not, and the tool has not been told which you are
building. The section on where your mix differs most from the map's says where
the difference is largest; whether to move it depends on what you are building.

When a scan covers more than one dataset, a further section reports the cosine
between each pair's region histograms. Two datasets can share no wording and
still occupy the same ground, which is what "we added a third source and gained
no new coverage" looks like from the outside; `T1-OVERLAP-001` compares text and
cannot see it.

### What the atlas still cannot tell you

The map stores `region_size` and `l1_size` for the reference mass, so a gap is
reported as under-representation against the reference corpus, not only as
absence. Read that baseline as a property of *this*
reference corpus — nine axes with byte targets, a non-English floor, per-source
caps, 60.6% non-English by bytes on atlas-v3 — not as a natural population.
And nothing here says whether a gap matters: the map has not been told what you
are building.

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

## Cell names

Every one of atlas-v3's 4,096 cells and 256 subject areas carries a
hand-written name (`region_labels` and `l1_labels` in the artifact; sources
`curated:region_labels_atlas-v3.json` and `curated:l1_labels_atlas-v3.json`).
The maps before it captioned cells from word frequencies; what was wrong with
that is measured under [Where the names come from](#where-the-names-come-from).

```
  Personal feelings, grief and confessional writing
    Casual venting posts about bad days and dating frustration
    Infidelity confessions and celebrity breakup stories
    Marriage, family life and explicit romance stories
```

**A name is a caption, not a rule.** No record is ever tested against it. Names
play no part in placing anything, and renaming a cell would not change a single
assignment.

### How a record is actually placed

1. Format-aware extraction pulls natural-language content (keys/syntax dropped).
2. The text is embedded by `potion-multilingual-128M` with SIF pooling, read
   through the map's own encoder input policy (see [What the encoder
   reads](#what-the-encoder-reads)). The
   encoder is stored quantised, one byte per weight with a per-row scale
   (142 MB on disk with its tokenizer), and the map uses the first 128 of its
   256 columns. The map was fitted in that same
   quantised coordinate system, so the encoder a run applies is the encoder the
   map was built with. The report names it in
   `atlas.identity.encoder_weight_hash`; if a run ever applies an atlas through
   weights it was not fitted on, the fitted hash is kept alongside as
   `encoder_built_with` so the report says so.
3. The frozen constants are applied: the mean of the record's detected language
   is subtracted (59 languages; the global mean for any other or unknown
   language), the two stripped principal
   directions are removed, and the vector is L2-normalised. See
   [Language is a nuisance parameter](#language-is-a-nuisance-parameter-not-a-clustering-axis).
4. Cosine similarity is computed against all fine-cell centroids, and the
   nearest cell drives the histogram.
5. Below the off-atlas cutoff — 0.3538 on atlas-v3 — the record is placed
   nowhere. The subject area is the cell's parent L1: a strict coarsening of the
   fine map, not a second model.

Word overlap is not consulted at any point. Four real placements, measured on
the 258-region build that preceded `atlas-v1-lite` — the region ids and the
five-word captions are that build's, and the point survives because the
placement rule has not changed:

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

### Where the names come from

`tools/atlas_naming_worklist.py` writes one worklist per subject area: for
every cell, thirty members of the build reservoir spread from the cell's centre to
its edge, its source, language and axis mix, its 64 contrastive terms, and its coherence under an
encoder that is not the atlas's own (see below). The names were written by hand
from those under `tools/atlas-data/NAMING_GUIDE.md`, one per cell and one per
subject area, and stored in `tools/atlas-data/region_labels_atlas-v3.json` and
`l1_labels_atlas-v3.json` keyed to the corpus hash. A rebuild on a different
corpus does not inherit them: it gets automatic contrastive-term captions and
says so in `region_labels_source`, rather than carrying names for cells that no
longer exist.

Never only the records nearest the centroid. The first atlas-v3 names were
written from the few nearest records, and in a cell with no shared subject those
records are unrelated to each other, so the name became a list of topics the cell
did not have: "Linux hardening logs, Brazilian court appeals and
sentence-rewriting prompts". The thirty are drawn by rank instead: a cell's
members are ordered by closeness to the centroid, cut into thirty bands of equal
count, and one is drawn at random from each band. Equal bands keep the sample in
proportion to the cell — two thirds of the thirty are two thirds of the cell —
while a purely random draw could, by luck, come out mostly core or mostly edge.
Bands are by rank rather than by distance, because a few stray records stretch
the distance range and equal-width distance bands would give them as many slots
as the dense core.

**Every name carries a kind** (`region_kinds`, `l1_kinds` in the artifact, and
`kind` beside each region in a fingerprint's `top_regions`):

| kind | the members share | the name reads |
| --- | --- | --- |
| `subject` | one subject, for at least two thirds of them | the subject, at the level the members support |
| `form` | a format, genre or template, but not a subject | the form, and that the subjects vary |
| `mixed` | neither | "Mixed …", then what little they share |

A k-means map places every record somewhere, so some cells will always be
catch-alls of short fragments and leftovers. A `mixed` cell is still a
coordinate — two corpora can be compared in it — but it is not a subject, and
the report does not pretend it is. `tools/atlas_label_rules.py` refuses a name
that lists three or more items for a form or mixed cell, names a first letter,
or calls a subject cell mixed; the stamping tools run it on every name.

**Names and cells are checked by readers that did not write them.** B7 in
`experiments/atlas-v3-benchmark` embeds twenty random members of every cell with
`paraphrase-multilingual-MiniLM-L12-v2` and scores each cell by the mean
pairwise cosine of its members, language by language centred: unrelated records
score 0.00, two cells of one subject area 0.17, a median cell about 0.25. The
atlas's own encoder cannot do this audit — a cell it drew is coherent to it by
construction, which is how cells held together by a first letter passed every
earlier name check. B7 then writes 160 cells, twenty from each coherence octile,
for blind readers who classify the cell and grade its name from sixteen members
drawn the same way, none of them shown to the namer where the cell has enough.

Hand names replaced frequency captions because the captions were measured to
be poor. On the 258-region build that preceded `atlas-v1-lite`, 21.6% of the
1,290 label slots were filled by a word that appeared in at least 8 regions
(14.1% of 2,580 slots on the shipped 215-region artifact), and 21% of regions
spent two or more of their five slots on inflections of one stem, because
nothing was lemmatised and Turkish is agglutinative. Roughly 40% of that label
text carried no information. The regions were real; their captions were not.

### What this does and does not affect

| affected | not affected |
| --- | --- |
| how readable a coverage report is | which cell a record lands in |
| whether you can tell a cell's subject from its name | off-atlas rate |
| how easy the atlas is to review by hand | region entropy, coverage counts, fingerprint comparability |

Every number the atlas produces is computed from centroids and assignments.
Renaming would change none of them.

## Why the coarse level is a hierarchy prefix, not a second model

No shipped product carries a supervised taxonomy probe. L1 is k-means over the
same vectors as L2, fitted first; L2 is k-means *within* each L1 membership.
The subject area of a fine cell is its parent, so a subject-area row in the
report is an exact union of fine cells and cannot contradict the fine map.

Topic and language breadth come from the **corpus plan** — nine axes with byte
targets, per-source caps, and a non-English floor that atlas-v3 clears at 60.6%
by bytes — not from a classifier trained on dataset provenance.

## Language is a nuisance parameter, not a clustering axis

Multilingual embeddings separate partly by language, so a flat k-means over a
multilingual corpus spends much of its region budget distinguishing Turkish
from Arabic from Chinese rather than distinguishing topics. The map before
this one had coarse regions named "Turkish television and radio" and
"Spanish-language server documentation" — registers of a language, not
subjects — and the scan already reports language separately.

So atlas-v3 applies **per-language mean centering** as a nuisance-parameter
correction, and this page says so plainly because
[design.md](design.md) rule 7 records the case against altering the embedding
space for language. What `Atlas.project` does: the mean vector of the record's
detected language is subtracted — 59 languages on atlas-v3, each mean
accumulated over every row of that language in the reference corpus — then the
two principal directions and the L2 step as before. A record whose language is
unknown, or which the build had fewer than 6,000 rows of, gets the global mean
instead. Nothing is projected out that predicts language identity; only the
mean moves, and the means ship in the artifact as `norm_lang_means`.

The three objections in rule 7 are still real. Where each one lands:

1. *It conditions on a label that is least reliable on short text and on
   closely related languages.* A record the detector will not commit to is
   `unknown` and gets the global mean, so a low-confidence record is not
   corrected wrongly — it is not corrected at all. The residual risk is a
   confident misidentification, and that was measured once in the wrong
   direction: an earlier v3 candidate took language from the source shard
   rather than detecting it per record, subtracted Serbian and Bulgarian means
   from Ukrainian text, and left the language in the geometry. Per-record
   detection is the fix that shipped.
2. *A language mean also encodes what that language's corpus is about.* True,
   and paid for. The probe on 300,000 held-out rows shows the normalization as a
   whole taking language balanced accuracy from 0.528 to 0.365 and axis
   balanced accuracy from 0.697 to 0.580: most of the language signal goes,
   and some subject signal goes with it.
3. *It is not inspectable.* The 59 means ship in the artifact, the detected
   language of every record is in the scan, and the fallback is a rule, so why
   a record landed where it did can be reconstructed from the files. It is
   hidden only in the sense that the report does not print the vector.

The off-atlas rate is still reported **per language as well as globally**, and
language remains its own fingerprint facet, measured by identification rather than by
clustering.

### The Ottoman case

Ottoman Turkish written in Arabic script gets its language and script from the
language facet. The atlas has no Ottoman mean — the 59 languages are those with
6,000 or more reference rows — so the detector's call decides which mean is
subtracted, or the global one if it says `unknown`. After that its cell is
decided by its content, and its language tag by the language facet, so the two
stay separable in the report: a corpus can be seen to add language coverage
without adding topical coverage, or the reverse. Whether Ottoman legal text and
modern Turkish legal text actually share cells on atlas-v3 has not been
measured.

## Records too short to place

A record below 40 characters is **excluded from placement**, not assigned. Its
embedding is dominated by noise, for the same reason language identification is
gated on length. Including such records would inflate the off-atlas rate with
records that were never placeable in the first place.

The number excluded is reported alongside the coverage figures. On a typical
short-form instruction corpus this can be most of the records, and that is worth
knowing rather than hiding: it means coverage describes the long tail of your
data, not all of it.

## Off-atlas data

A record is **off-atlas** when its cosine similarity to the nearest centroid
falls below a threshold calibrated at build time and stamped into the artifact
as `off_atlas_threshold`; atlas-v3 carries **0.354** (stored as 0.3538). Those records are
excluded from the region histogram and the category counts, so every share the
report prints is a share of the **placed** records, and the placed count is
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

### How the cutoff is calibrated

The sentence above, made exact:

> `off_atlas_threshold` is the 2nd percentile of nearest-cell cosine over the
> build's language-and-axis-balanced calibration draw.

The draw is the builder's `balanced_calibration_draw`: the same 2,000,000
rows, from the same seed, that fit the global mean and the two stripped
principal directions, so the cutoff and the normalization are calibrated on one
sample. Every (language, axis) stratum gets the same quota, and a stratum too
small to fill its quota hands the remainder back to the rest. Balance is the
point. A proportional draw of the reference corpus is 53% English web, and a
percentile of it is an English-web cutoff that rejects ordinary records of every
smaller stratum at more than 2%. Measured on atlas-v3:

| draw | p1 | p2 | p5 | p50 |
| --- | --- | --- | --- | --- |
| balanced calibration draw, 1,998,898 rows over 750 strata | 0.334 | **0.354** | 0.388 | 0.587 |
| the same rows reweighted to the corpus's own proportions | | 0.379 | | |
| held-out in-family prose (fineweb en, wikipedia en, fineweb-2 tr; 26,000 records, benchmark b4) | 0.403 | 0.425 | 0.466 | 0.653 |

The cutoff is one number for every axis, and the axes do not sit at the same
place under it. Second percentile by axis in the same draw: code 0.312, training
0.334, educational 0.353, forum 0.370, books 0.375, encyclopedic 0.381, web
0.391, legal and government 0.413, scientific 0.421. English alone is 0.422, and
the English-web stratum 0.431, which is where the held-out in-family figure
comes from. So a corpus of ordinary English web prose sits near 0.2% off-atlas,
a corpus balanced like the reference draw at 2%, and a corpus of nothing but
code near 9%, all under one cutoff and none of it because the records are unlike
the atlas. The rate has to be read against what the corpus is made of, which is
why the report says what the off-atlas records are.

The similarities come from the build's own embedding store, whose rows are
exactly what the runtime encoder produces for the same record (checked: cosine
1.0 against re-encoding the reservoir text), projected through the artifact's
per-language centering with each row's detected language. Rows the store never
wrote — all-zero blocks left behind by an interrupted ingest, 115,200 rows in 17 runs of the
163 million — are dropped rather than scored, since a zero row scores zero and
enough of them would drag the percentile there.

`tools/calibrate_off_atlas_v3.py` computes the number and stamps it; the
artifact records the draw, the percentiles and the per-axis breakdown under
`off_atlas_calibration`, and `tests/test_atlas_pipeline.py` fails if a bundled
v3 ever ships without the key. A map that leaves the key out gets the loader's
0.35 fallback, and that is not a calibrated number: on the map before this
one, whose own 2nd percentile was 0.309, it put 12–18% of ordinary held-out
prose off-atlas.

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

The cutoff cannot be asked to do more, and the reason is in the numbers rather
than in the choice of number. On this encoder the machine formats sit inside the
band where ordinary prose scores. Median similarity on atlas-v3 (benchmark b4,
1,400 synthetic records): DNA strings 0.43, random letters 0.46, random unicode
0.47, base64 0.49, minified JavaScript 0.52, hex log lines 0.59,
comma-separated numbers 0.75 — against a 2nd percentile of 0.425 and a median of
0.65 for held-out in-family prose. No cutoff separates the two. At the
in-family 2nd percentile, 0.425, the cutoff rejects 2% of prose and 9.6% of the
machine formats (45% of the DNA, none of the hex, JavaScript or numbers); at
0.50 it rejects 9% of prose to catch 49% of them. Raising it buys machine
formats with prose, roughly one for one, and the calibrated cutoff is set for
prose. What identifies a machine-format off-atlas set is the surface-share
diagnosis above — whitespace share and non-letter share — and what catches
machine formats wherever they place is the encoding and degeneracy checks.

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

The artifact carries no supervised taxonomy probe, so the two figures that
used to travel in the `coverage` facet — level-0 held-out accuracy and region
purity by taxonomy — do not exist for it. They were v0 and v1
concepts: a probe trained to reproduce the provenance label of each reference
record, and a purity score against those labels. What was wrong with them is
kept here because it is why v3 has no taxonomy at all.

### What 0.864 accuracy did not mean

Measured on the 258-region v0 build. It measured how well the probe reproduced
the **provenance labels** it was trained on, not whether those labels were
correct. The level-0 label of every reference record was inherited from the
dataset it came from, so `general_chat` held 106 of 258 regions — UltraChat,
Alpaca, Dolly and four Turkish instruction sets, whose regions covered film,
colour theory, poetry, blockchain, football and code — and two categories were
mislabelled outright: `summarization` (regions 106–115) was `tr-wikihow-summ`,
how-to instructions rather than summaries, and `religion_philosophy` (219–225)
was Arabic Wikipedia, two of whose regions were about languages and computers.
A high accuracy meant the probe had faithfully learned a wrong taxonomy.

The fix was not a better probe but no probe: v3 has no categories to inherit or
to learn. Subject areas are k-means over the same vectors as the cells, and
every one is named by hand from its own members.

What travels with an atlas-v3 map instead:

| number | where | meaning |
| --- | --- | --- |
| language probe | `language_probe` | balanced accuracy of a linear probe for language on 300,000 held-out rows: 0.528 on raw vectors, 0.365 after normalization. The same probe for axis: 0.697 to 0.580. Read as: most of the language signal removed, most of the subject signal kept. |
| off-atlas calibration | `off_atlas_calibration` | the draw, the percentiles and the per-axis breakdown behind the 0.3538 cutoff; see [How the cutoff is calibrated](#how-the-cutoff-is-calibrated). |
| reference mass | `region_size`, `l1_size` | how many of the 163,452,464 reference records sit in each cell and subject area — the denominator of every density the report prints. |
| identity | `encoder_weight_hash`, `corpus_hash`, `pipeline_hash` | whether two maps, or a map and a run, are in the same coordinate system. |

## The product

The package bundles one artifact. Every figure below is read from its own
metadata.

| property | atlas-v3 |
| --- | --- |
| fine cells (L2) | 4,096 |
| subject areas (L1) | 256 |
| dimensions | 128 |
| reference records | 163,452,464 |
| sources | 244 |
| non-English share | 60.6% by bytes |
| language means | 59 |
| L2 allocation | population budget of 4,096, k 1–64 per L1 |
| default sample | 500,000 |
| off-atlas cutoff | 0.3538, stamped |
| size in the wheel | 13.8 MB |
| corpus hash | `99687a41…` |

The map reports at the fine-cell level (`user_resolution` is `l2`); the
subject area is the cell's parent and only groups and names rows.

The maps before it — `atlas-v2` (296 cells over 128 areas, 128-d),
`atlas-v2-lite` (65 cells over 32 areas, 64-d) and `atlas-v1-lite` (215 cells
over 48 areas) — were fitted on smaller corpora with cells of their own, and
none of them is bundled or loadable by name. A fingerprint placed on one of
them is not comparable with one placed on atlas-v3, and `diff` refuses the
pair. Measurements in this document that were taken on one of those maps, or
on the 258-region build before them, say so where they appear.

## Hand intervention

Human judgement is used where it has leverage and nowhere else, and every hand
edit is a versioned file the build reads, never an edit to the artifact. Edits
applied directly to the artifact would make it impossible to rebuild.

| what | who decides |
| --- | --- |
| the corpus plan: nine axes, byte targets, per-source caps, the non-English floor | designed by hand, in `tools/atlas_sources.py` |
| 256 subject areas and 4,096 cells | clustered |
| the name of every subject area and every cell | written by hand, from the build's contrastive terms and review excerpts |
| the off-atlas cutoff | computed by `tools/calibrate_off_atlas_v3.py` and stamped |

The names live in `tools/atlas-data/l1_labels_atlas-v3.json` and
`tools/atlas-data/region_labels_atlas-v3.json`, keyed to the corpus hash, so a
build on a different corpus cannot pick them up by accident.

Cell ids are never renumbered inside a product, because they are part of the
fingerprint schema. A rebuild that moves them ships under a new product name,
which is why the map that replaces atlas-v3 will be called atlas-v4 rather
than shipped as a new atlas-v3.

## Rebuilding

```bash
tools/run_atlas_v3_build.sh
```

That script runs, detached and with a live log,

```bash
ATLAS_BUILD_HASH_SLOTS=$((1 << 28)) \
python tools/build_atlas_v2.py --product atlas-v3 \
  --cache "$storage/corpus-cache" --work "$storage/work" --out-dir "$storage/release"
```

and, when the build exits 0, copies `atlas-v3.npz` and
`atlas-v3-release-notes.json` into `src/dropoutt/data/atlas/` and writes
`atlas-v3-SHA256SUMS` beside them. The storage root defaults to
`/Volumes/ck512/dropoutt-atlas-v2` and is overridden with
`DROPOUTT_ATLAS_STORAGE`; `ATLAS_NORM_DIR` points the normalized memmap at a
fast disk for the L2 phase, which reads each region's members back as one
contiguous slice.

The corpus cache is fetched separately and never modified by a build. The
manifest, its hash and the source ledger are recorded in the release notes, so
the same manifest yields the same corpus hash. Ingest checkpoints after every
shard and clustering at three points — normalization, the L1 fit, and every
eight L1 regions of the L2 loop — each keyed to the corpus hash, so an
interrupted build resumes where it stopped and a different corpus can never
resume from its files; `tools/resume_atlas_v3_build.sh` restarts one. A build
that did not consume every manifest shard fails rather than shipping a partial
map. It warns, and the release notes record, when the IDF token mass, the
non-English floor (50.5%) or an axis byte floor is missed: on the shipped
build, books, scientific and code missed their floors and the rest cleared
them.

Three steps follow the build. `tools/calibrate_off_atlas_v3.py` computes the
off-atlas cutoff from the build's own embedding store and stamps it into the
artifact; the hand-written names are read from `tools/atlas-data/` when
their corpus hash matches, otherwise the artifact carries automatic
contrastive-term captions and `region_labels_source` says so; and
`tools/strip_atlas_exemplars.py` removes the review excerpts
(`exemplar_texts`) from the copy that ships and restamps the checksum, keeping
the full artifact on the build volume for the next labelling pass.
`tests/test_atlas_pipeline.py` fails if a bundled map carries a text array.
