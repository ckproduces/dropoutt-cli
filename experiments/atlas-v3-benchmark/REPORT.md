# How good is the rebuilt atlas map? Benchmark report, 16 September 2026

*Tests run on 16 September 2026 against the atlas-v3 map shipped on 14 September (corpus `99687a41`, 163,189,202 documents), the map it replaced (built 12 September, corpus `4203fc3a`), and the older atlas-v2 and atlas-v2-lite. Every number comes from the files in `results/`; `python -m bench.summary` prints them all. The technical detail is in the appendix at the end.*

## The short version

The atlas is a fixed map of text: 4,096 small areas ("cells") grouped into 256 larger ones ("districts"), each with a hand-written name. `dropoutt atlas` places every record of a dataset on it and reports which areas the dataset fills, how densely, and what it never reaches. Two datasets placed on the same map can be compared.

The map was rebuilt on 14 September to fix one problem: too many cells were grab-bags, groups of records that shared a spelling quirk or a prompt template rather than a subject, and the names on those cells hid it. The rebuild changed what the encoder reads (ALL-CAPS runs folded, table rules stripped, short cased, symbol and digit tokens damped, template phrases damped) and weighted the fit by document length.

**Verdict: the rebuild did what it was for, at a cost that is now measured.**

What got better:

- **Fewer grab-bag cells.** Cells an outside encoder rates as incoherent halved, from 147 to 75 of 4,096. The 40-cell blind read found 7 grab-bags, and every one of the worst was already labelled "Mixed ... with no shared subject". No name is a soup any more; 85% of names read as accurate.
- **Better subject sorting on exam questions.** MMLU questions land in the right subject cell 40.7% of the time against 39.3% before, a real gain (the interval on the paired difference excludes zero). MMLU-Pro districts improved by 5 points.
- **Better across languages.** A sentence and its Turkish translation land in the same district 41.3% of the time against 40.2%; Turkish sentences fall off the map half as often (3.1% against 6.4%).
- **Junk is caught more often.** Columns of numbers are now kept off the map 94% of the time and base64, DNA and minified code about a quarter of the time. The previous map kept none of it off.
- **The map is used more fully.** A typical English web corpus now covers 68% of the map's effective area against 64%, and the coverage ladder rises at every step where the previous map dipped once.

What got worse:

- **Source code is placed worse.** On the previous map, code datasets sat 43 to 73% inside one code district. Now most of them put only 25 to 30% in a how-to district, two (C and StarCoder) have their largest share in a district the map itself calls mixed, and one Python dataset lands 81% in mixed cells. The encoder changes damp exactly the tokens code is made of, and the code axis reached only a tenth of its planned size in the corpus.
- **Kind of data is read slightly worse.** Naming a held-out record's kind (code, math, law, science, chat, ...) from its cell alone dropped from 65.7% to 63.0%, a real 2.7-point loss. Naming its exact source dataset did not change (30.3% against 30.6%, within noise).
- **Encoding is 40% slower.** 500,000 records now take 82 seconds to encode against 59, because the input policy does more work per record. A full default run is about 2 minutes 10 seconds against 1 minute 50.
- **One in eight records lands in a cell named "Mixed".** 448 cells (11% of the map, 11% of its reference mass) are honestly labelled mixed. 14% of held-out records and 9% of English web records land in them. Those records are located, not described. The previous map had the same kind of cells but named them as if they were subjects.

What did not change: how well hidden slices are found (EU law and biomedical QA at 0.25% of a 40,000-record corpus, estimated within 5%), the honesty of the density readout (the map's own reference sample reads as 0.98x, and 99% of cells sit between 0.5x and 2x), and the sample size a reliable comparison needs (20,000 records per side at cell level, 1,000 at district level).

## Why these tests

The atlas is not a classifier and it is not a quality score. Its whole job is to give every dataset the same coordinates so that a person can read where a dataset sits and what a second dataset adds. So the tests are organised around the questions a user of `dropoutt atlas` actually asks, not around abstract clustering metrics.

| the user's question | what would make the answer wrong | test |
| --- | --- | --- |
| Where does my corpus sit? | Records of one subject scattered over many cells, or records of different subjects in one cell | B1 (52 unfamiliar datasets), B2 (exam questions by subject) |
| Is "3x the map" true? | Densities that drift from 1.0x on data that matches the map | B8 (place the map's own reference sample) |
| Are the names right? | A name that does not describe what lands under it; a cell that has no one subject to name | B5, B7, B9 (names read by the map's encoder, by an outside encoder, and blind by a reader) |
| What does this dataset add to what I have? | A small new slice that goes unnoticed; a "new" figure that is really sampling noise | B4 (hidden slices, breadth ladder, the CLI's own comparison, two halves of one corpus) |
| Does language get in the way? | The same topic in two languages landing in different places; cells that are really language buckets | B3 (14 topics in 4 languages, 8,000 translation pairs, language mix inside cells) |
| What does "off the map" mean? | Ordinary text rejected; machine junk accepted | B4 and B8 (cutoff on held-out prose, on junk, on the map's own mix by kind and language) |
| How many records, how long? | A readout that needs more records than users will place; a run that takes too long | B4 (two halves of one corpus against sample size), B6 (timing) |

Three of the nine tests are new since the 12 September report and were added because of what the rebuild changed:

- **B8, the density readout.** The rebuild weighted the fit by document length. If that had shifted where the reference mass sits, the CLI's "1.0x matches the map" would have quietly stopped being true. It had to be checked directly.
- **B9, names read from outside and blind.** The 13 September naming audit showed that a cell built by an encoder is coherent *to that encoder* by construction, so B5 (which uses the atlas's own encoder) passed cells that shared only a first letter. B9 embeds names and members with a different encoder, and adds two blind reads: does the name of the cell that holds most of a held-out dataset describe that dataset, and does a cell hold one subject.
- **Records landing in mixed cells** (part of B9). The new map is the first whose names admit which cells are grab-bags. That makes a new user-facing number measurable: how much of a corpus the map can only locate, not describe.

Two things from the earlier suite were kept but strengthened. Every headline number now carries a 95% interval, and every difference between two maps is *paired*: the same records are scored on both maps and the difference is bootstrapped record by record, so the interval speaks about the maps rather than about which records happened to be sampled. And the "new between two halves of one corpus" figure, which the earlier report flagged as inflated, is now shown beside what pure sampling noise would produce, which turns out to explain nearly all of it.

## How the tests were run

### The rules

1. **Test on data the map never saw.** The 52-dataset panel, C4, MMLU, MMLU-Pro, the Turkish EXAMS set and the WMT17 translation pairs are not inputs to any v3 build; this was checked against the build manifest of 244 sources. Web and Wikipedia panels come from the same public collections as the build (but not necessarily the same pages) and are used only for language and coverage readings, never for accuracy.
2. **Every map gets the same records through the real command's steps.** Same text window (2,000 characters), same encoder with the map's own word weights and, for the new map, its input policy; same per-record language detection; same placement rule; same stamped cutoff. Nothing was tuned for the test.
3. **Every sorting score is shown next to luck and next to a ceiling.** Luck is what you get by always guessing the commonest answer. The ceiling is a trained classifier on the same vectors the map uses. The map's score between the two says how much of what the encoder knows the frozen grid keeps.
4. **Uncertainty is stated.** Proportions carry a Wilson 95% interval; accuracies carry a bootstrap 95% interval over records; map-to-map differences are paired bootstraps. A difference is called real only when its interval excludes zero.
5. **Sampling noise gets its own baseline.** Where a number can look bad purely because of how many records were placed, the same number is computed for a random draw of the same size.
6. **Blind reading.** Names and cells were shuffled and the map's identity hidden before a reader scored them. The reader was Claude (Fable 5.1) in the benchmark session, not a human; the sheets and answers are in `results/` so a human can redo them.
7. **Compare like with like.** Statistics that reward having more cells stay in the appendix and are compared only between the two v3 maps, which have the same 4,096 cells.

### The maps

| map | cells / districts | built from | names | off-map cutoff |
| --- | ---: | --- | --- | ---: |
| **atlas-v3 (shipped 14 Sep)** | 4,096 / 256 | 163.2 million documents, 244 sources, 59 language means; encoder input policy v1; length-weighted fit | hand-written at both levels, with a kind (subject / form / mixed) on every cell | 0.3352, calibrated |
| previous v3 (12 Sep) | 4,096 / 256 | 163.5 million documents, same sources, no input policy | hand-written at both levels | 0.3538, calibrated |
| atlas-v2 | 296 / 128 | 69 million documents, 59 sources | district names hand-written, cell captions automatic | 0.35, fallback |
| atlas-v2-lite | 65 / 32 | same corpus as v2, 64 dimensions | district names hand-written | 0.35, fallback |

### The data

| what | how much | seen in building? | used for |
| --- | ---: | --- | --- |
| 52 public datasets: code in 11 languages, competition and word-problem maths, EU and US law, finance, scientific papers, chat and instruction data in English and Turkish, question answering, text-to-SQL, general web, Turkish news | 1,000 records each, 51,489 in all | no | sorting by source and by kind (B1); blind names (B9) |
| MMLU, MMLU-Pro, Turkish EXAMS | 13,756 / 12,007 / 1,561 questions | no | sorting by subject (B2); blind names (B9) |
| 14 topics written in English, Turkish, Arabic and Chinese; 4 junk blobs | 56 + 4 texts | no | language (B3) |
| WMT17 news sentences with Turkish translations | 8,000 pairs | no | language (B3) |
| Web pages in 9 languages, Wikipedia in 13 languages | 36,000 / 39,000 | same collections | language mix (B3), ladder (B4) |
| C4 English web | 40,000 | no | base corpus for hidden slices, density readout, mixed-cell share |
| The map's own reservoir: 500,000 records sampled uniformly during the build, with the build's axis and language per record | 200,000 placed; 20 members per cell read | yes, by design | density calibration (B8); cell coherence and names (B7, B9) |
| Synthetic junk: base64, hex dumps, DNA, minified JavaScript, columns of numbers, random letters and characters | 1,400 | no | cutoff (B4) |

### Steps

1. Load the four maps and the encoder. Confirm each map's encoder weights and word table are the ones it was built with.
2. Read the test data; drop records under 80 characters, as the command does.
3. Detect each record's language the way the command does.
4. Place every record on every map; keep the cell, the district, the similarity score and the five nearest cells.
5. Score sorting with the majority-vote rule (next section), with luck, ceiling, interval, and the paired difference to the shipped map.
6. Score language: translations and same-topic texts landing together; language mix inside cells; the same again with per-language centering switched off.
7. Score coverage: hide a slice of one dataset in the web corpus at six sizes; build a corpus in nine steps of increasing breadth; run the command's own comparison on ten pairs; compare two halves of one corpus at seven sample sizes against a sampling-noise baseline; measure the cutoff on prose, on junk, and on the map's own mix.
8. Score the density readout on the map's own reservoir against a random draw from its stored reference shares.
9. Score names three ways: with the map's own encoder, with an outside encoder over 20 members per cell, and blind over 109 held-out slices on both v3 maps (218 items). Read 40 cells blind, five per coherence band.
10. Time a 500,000-record run on both v3 maps and run the real command on three small corpora, keeping the printouts.

### How a sorting score is computed

Take the 51,489 panel records. Place them. Split them in half at random.

- On the first half, look at each cell and note which dataset most of its records came from. That becomes the cell's label. This is a **majority vote**: no training, no model, just "records that land here usually come from X".
- On the second half, for each record, guess the label of the cell it landed in. Count how often the guess is right. That is the score.
- **Luck** is the score from always guessing the single commonest dataset. **Ceiling** is a logistic-regression classifier trained on the first half's vectors and scored on the second half. **Kept share** is (score minus luck) divided by (ceiling minus luck): the fraction of what could be known that the frozen grid keeps.
- The **paired difference** between two maps re-samples the second-half records 1,000 times and computes the difference in hit rate each time. If the middle 95% of those differences excludes zero, the maps really differ on this test.

## Results

### 1. Keeping unfamiliar datasets apart (B1)

**Question.** If I place a dataset the map has never seen, do its records land together, and does the place tell me what kind of data it is?

**Answer.** As well as before for the exact dataset, slightly worse for the kind of data. The shipped map keeps a larger share of what its encoder knows, but its encoder knows a little less about sources, because the input policy removes the surface cues (casing, symbols, digits) that identified them.

| map | off the map | source named right (52 choices) | 95% interval | luck | ceiling | kept share | kind named right (8 kinds) | 95% interval | luck | ceiling | kept share |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| **atlas-v3** | 0.8% | 30.3% | [29.7, 30.8] | 2.1% | 60.8% | 48% | 63.0% | [62.4, 63.6] | 25.4% | 79.4% | 70% |
| previous v3 | 0.7% | 30.6% | [30.1, 31.2] | 2.1% | 64.9% | 45% | 65.7% | [65.1, 66.2] | 25.4% | 82.3% | 71% |
| atlas-v2 | 11.8% | 23.9% | [23.3, 24.4] | 2.1% | 65.7% | 34% | 62.4% | [61.7, 62.9] | 25.4% | 82.2% | 65% |
| atlas-v2-lite | 16.9% | 17.0% | [16.5, 17.4] | 2.1% | 57.0% | 27% | 54.6% | [54.0, 55.3] | 25.4% | 76.6% | 57% |

Paired differences, shipped map minus previous v3, same records: source −0.3 points [−1.0, +0.3], not a real difference; kind −2.7 points [−3.2, −2.1], a real loss. Against atlas-v2 the shipped map gains 6.4 points on source and 0.6 on kind, both real.

**How to read this.** From nothing but the cell a record landed in, the shipped map names its source dataset right 30 times in 100 where luck gives 2 and a trained classifier gives 61. Note the ceiling: the classifier itself does worse on the new map's vectors (61% against 65%), because the input policy deliberately hides the formatting that made, say, a Java commit look like a Java commit. Of the headroom that remains, the grid keeps 48% against 45% before. So the grid got better at holding what it is given, and it is given less.

Where held-out datasets land on the shipped map, by the district that holds most of their records:

| dataset | share in main district | that district | districts touched |
| --- | ---: | --- | ---: |
| competition number theory | 85% | Mathematics, algorithms and programming problems | 42 |
| competition geometry | 83% | Mathematics, algorithms and programming problems | 27 |
| machine-learning papers | 79% | Machine learning, statistics and computational methods | 51 |
| Python text-to-code programs | 78% | Mixed encyclopedic fragments with no shared subject | 30 |
| PHP commits | 62% | Programming, web code and desktop software how-tos | 64 |
| US Supreme Court opinions | 53% | Court opinions, criminal appeals and litigation in the United States | 57 |
| EU regulations | 39% | EU agricultural market rules, levies and refunds | 42 |
| C4 English web | 2% | Local events, live entertainment and venue schedules | 229 |

The fourth row is the code problem in one line: a dataset of Python programs lands three-quarters in a district the map itself calls mixed. Section 9 measures it.

### 2. Sorting exam questions by subject (B2)

**Question.** Do questions about the same subject land in the same place, even though they look alike on the surface (a stem and four options)?

**Answer.** Better than before on the two English exam sets, unchanged on the Turkish one.

| map | exam set | questions | subjects | off the map | subject right (cell) | 95% interval | luck | ceiling | kept share | subject right (district) |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| **atlas-v3** | MMLU | 13,756 | 57 | 0.6% | 40.7% | [39.6, 41.8] | 11.0% | 62.9% | 57% | 38.4% |
| previous v3 | MMLU | | | 1.0% | 39.3% | [38.1, 40.5] | 11.0% | 63.6% | 54% | 36.1% |
| atlas-v2 | MMLU | | | 17.7% | 35.1% | [34.1, 36.2] | 11.0% | 63.7% | 46% | 32.5% |
| **atlas-v3** | MMLU-Pro | 12,007 | 14 | 0.7% | 51.8% | [50.4, 53.0] | 11.3% | 69.5% | 70% | 50.3% |
| previous v3 | MMLU-Pro | | | 0.4% | 50.6% | [49.3, 51.8] | 11.3% | 69.3% | 68% | 45.3% |
| **atlas-v3** | EXAMS (Turkish) | 1,561 | 8 | 2.1% | 55.1% | [51.6, 58.6] | 23.6% | 82.1% | 54% | 63.9% |
| previous v3 | EXAMS (Turkish) | | | 2.8% | 57.4% | [53.9, 60.9] | 23.6% | 81.7% | 58% | 63.3% |
| atlas-v2 | EXAMS (Turkish) | | | 25.9% | 61.7% | [58.5, 64.9] | 23.6% | 81.2% | 66% | 61.6% |

Paired differences, shipped minus previous: MMLU +1.4 [+0.2, +2.5] at cell level and +2.4 [+1.3, +3.4] at district level, both real; MMLU-Pro +1.1 [−0.1, +2.5] at cell level (not quite) and +4.9 [+3.6, +6.3] at district level (real); EXAMS −2.3 [−5.8, +1.2], not a real difference on 780 test questions.

**How to read this.** On MMLU the shipped map keeps 57% of the classifier's headroom against 54%. MMLU-Pro (a harder, 14-way version) is read at 52%, and its districts at 50%, which is where the largest gain sits. On the Turkish exam set atlas-v2 still scores highest at cell level, but it also throws 26% of the questions off the map, so it is scoring on the easy three-quarters.

Best- and worst-sorted MMLU subjects on the shipped map (share of the subject in its top district): moral scenarios 88% (in a district honestly named *Mixed personal stories and narrative fiction fragments*; these items are odd two-scenario prompts), abstract algebra 69%, college mathematics 64%, international law 55%, astronomy 52%; at the bottom, sociology 11%, global facts 11%, management 10%, high-school geography 9%, human aging 8%, miscellaneous 5%. The bottom subjects are the ones whose questions are about many things at once.

### 3. Subject, not language (B3)

**Question.** If I place the same topic written in different languages, does it land in the same place? And are cells secretly language buckets?

**Answer.** Slightly better than before, and far better than v2. The per-language centering the map applies before placing is what does it.

| map | same topic, other language, same district (84 pairs) | 95% interval | different topics, same language, same district | translation pair same district (8,000) | 95% interval | luck | same cell | luck | off the map, English / Turkish |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| **atlas-v3** | 54.8% | [44.1, 65.0] | 1.1% | 41.3% | [40.2, 42.4] | 2.1% | 26.2% | 0.7% | 0.9% / 3.1% |
| previous v3 | 48.8% | [38.4, 59.3] | 0.3% | 40.2% | [39.1, 41.3] | 2.0% | 24.8% | 0.5% | 1.9% / 6.4% |
| atlas-v2 | 29.8% | [21.0, 40.2] | 21.7% | 40.7% | [39.6, 41.8] | 3.6% | 35.2% | 2.0% | 28.6% / 39.2% |
| atlas-v3 without centering | 44.0% | [33.9, 54.7] | 1.4% | 39.6% | [38.5, 40.6] | 2.1% | 23.4% | 0.6% | 2.8% / 3.1% |

**How to read this.** A news sentence and its Turkish translation land in the same one of 256 districts 41 times in 100, where luck gives 2. In the same one of 4,096 cells 26 times in 100, where luck gives under 1. The two v3 maps overlap in their intervals on the translation test, so call that a small, probable gain rather than a proven one; the drop in Turkish records falling off the map (6.4% to 3.1%) is clear. atlas-v2 looks competitive on "same cell" only because it has 296 cells instead of 4,096, and it drops a third of the sentences off the map.

Inside cells, language is still visible but is not what the cells are about. On web pages in 9 languages, 13% of records sit in cells that are 90% one language (12% before); on Wikipedia in 13 languages, 6%. Switching centering off raises those to 22% and 10%. No language in either panel falls off the map more than 1.4% of the time.

### 4. Finding a hidden slice, and what "new" means (B4)

**Question.** If a small amount of something new is mixed into a big corpus, does the map notice, and does it measure the amount? When I compare two datasets, is "new" real or noise?

**Answer.** Concentrated slices are found at a quarter of one percent and measured within 5%. Broad slices need 1 to 2% and are underestimated by a quarter. Two halves of the same corpus read as 19% "new" at 5,000 records per side, which the new baseline shows is almost entirely sampling noise; 20,000 records per side, or reading at district level, makes it go away.

Smallest share of a 40,000-record English web corpus at which the injected slice was detected (z ≥ 3) with its share estimated within a factor of two, cell level:

| map | EU law | Python commits | maths word problems | Turkish instructions | text-to-SQL | biomedical QA | Turkish web |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **atlas-v3** | 0.25% | 0.50% | 1% | 2% | 1% | 0.25% | 2% |
| previous v3 | 0.25% | 0.25% | 1% | 2% | 1% | 0.25% | 2% |
| atlas-v2 | 0.5% | 0.25% | 2% | 5% | 2% | 0.5% | 5% |
| atlas-v2-lite | never | 1% | 5% | 10% | 2% | 0.5% | never |

How well the amount was measured on the shipped map (estimate divided by truth, across the six sizes): EU law 0.98 to 1.05; biomedical QA 0.99 to 1.04; Python commits 0.97 to 1.05; maths 0.86 to 0.94; text-to-SQL 0.85 to 0.90; Turkish instructions 0.73 to 0.81; Turkish web 0.67 to 0.75. The last two are broad: their home cells already hold 20% of ordinary web text, so part of the injected mass is indistinguishable from the base.

**How to read this.** 100 EU-law documents hidden among 40,000 web pages are found with a z-score of 23 (anything above 3 is a confident detection), and their share is estimated within 5%. Python commits now need 200 documents where 100 sufficed before; that is the code regression again.

**The breadth ladder.** A corpus of geometry problems alone reads as 1.1% of the map's effective area. Adding all competition maths takes it to 10.9%, code to 13.4%, chat to 33.9%, law to 33.9%, science to 35.2%, English web to 46.3%, web in 8 languages to 54.1%, Turkish instructions and 13 Wikipedias to 59.2%. It never falls. The previous map dipped once (30.5% to 29.7% when law was added); atlas-v2 dipped when science was added.

**The command's own comparison** on ten known pairs (similarity from 0 to 1, then the "new" share read left against right):

| pair | atlas-v3 | previous v3 | atlas-v2 |
| --- | --- | --- | --- |
| same corpus, two halves of 5,000 | 0.93, 19% new | 0.93, 19% new | 0.99, 0% new |
| Alpaca vs its Turkish translation | 0.74, 22% | 0.66, 20% | 0.76, 1% |
| Turkish instructions vs English instructions | 0.46, 40% | 0.36, 47% | 0.46, 2% |
| algebra vs intermediate algebra | 0.95, 16% | 0.94, 9% | 0.91, 5% |
| Python commits vs Python programs | 0.03, 65% | 0.18, 69% | 0.08, 18% |
| EU law vs human-rights court cases | 0.00, 90% | 0.00, 98% | 0.03, 52% |
| Turkish web vs English web | 0.21, 46% | 0.18, 47% | 0.46, 4% |
| Python commits vs chat | 0.12, 32% | 0.07, 48% | 0.08, 3% |
| biomedical QA vs EU law | 0.00, 97% | 0.00, 94% | 0.01, 6% |

The shipped map reads a dataset and its translation as more alike than before (0.74 against 0.66) and unrelated pairs as 0.00, which is right. Two rows need care. "Same corpus, two halves" should be 0% new and reads 19% on both v3 maps; atlas-v2 reads 0% only because 296 cells are easy to fill with 5,000 records. And "Python commits vs Python programs" reads 0.03 on the shipped map, lower than before, because the Python programs land in mixed cells (section 9) rather than beside the commits.

**Two halves of one corpus, against sample size** (shipped map, C4 English web; "floor" is what two independent random draws of that size from the same distribution would show):

| records per side | new, cell level | sampling floor | new, district level |
| ---: | ---: | ---: | ---: |
| 250 | 91% | 55% | 28% |
| 1,000 | 67% | 42% | 4.7% |
| 5,000 | 21% | 17% | 0.1% |
| 10,000 | 7.8% | 7.0% | 0.1% |
| 20,000 | 2.3% | 2.2% | 0.0% |

**How to read this.** With 4,096 cells, two random halves of the same corpus will always miss some of each other's cells until each half is large; the floor column says how much. The map adds only a few points above the floor, so this is a property of any fine grid, not instability in the map. The practical rule for users: to trust a cell-level "new" figure below 5%, place at least 20,000 records per side; at district level 1,000 is enough. Both v3 maps and all three corpora tested (English web, chat, Turkish web) give the same thresholds.

### 5. Off the map (B4, B8)

**Question.** When a record is "off the map", what does that mean, and does ordinary text ever get thrown off?

**Answer.** Ordinary text almost never falls off (0.08% of held-out English prose, 1.1% of the map's own mix). Machine junk is now rejected about a quarter of the time, up from never; number columns are rejected 94% of the time. Off-map is still not a garbage detector.

| map | cutoff | held-out prose off the map | junk off the map | number columns | DNA | minified code | base64 | hex dump |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **atlas-v3** | 0.3352 | 0.08% | 27% | 94% | 38% | 23% | 21% | 0% |
| previous v3 | 0.3538 | 0.15% | 0% | 0% | 0% | 0% | 0% | 0% |
| atlas-v2 | 0.35 | 5.4% | 7% | 0% | 0% | 44% | 0% | 0% |

On the map's own reservoir (200,000 records, the build's mix), 1.07% falls off. By kind of source: code 3.5% (2.2% before), instruction-training data 3.0%, books 0.8%, encyclopedic 0.8%, web 0.3%, scientific 0.1%. By language: English 0.3%, Turkish 1.6%, Chinese 1.5%, Indonesian 2.1%, Vietnamese 2.7%. The cutoff was set at the 2nd percentile of a language- and axis-balanced draw, so 1 to 3% on minority kinds is the design, not a fault.

**How to read this.** The cutoff separates "looks like some text on the map" from "looks like nothing here". The input policy damps digits and symbols, so a column of numbers now looks like almost nothing and falls below the line; a hex dump still reads as text because "0x" tokens are words to the encoder. Junk detection belongs to the scan, as the docs say; this test only checks that the cutoff does not fire on real text, and it does not.

### 6. Is 1.0x really "matches the map"? (B8)

**Question.** The command prints density: your share of an area divided by the map's own share. Is the map's own share right?

**Answer.** Yes. Placing a 200,000-record sample of the map's own reference corpus gives a median density of 0.98x; after the command's smoothing, 99.1% of cells sit between 0.5x and 2x. The previous map behaves the same (0.99x, 98.9%).

Without smoothing, 96.3% of cells sit between 0.5x and 2x where pure sampling noise would leave 99.9%, so there is a small real drift between where the build counted a record and where the runtime places it (the two v3 maps drift equally, 4.7 and 4.8 times the noise level). The command's smoothing, which pulls thinly observed cells towards 1.0x, absorbs it. The most over-dense cells on the map's own sample are legal (US court cases 3.9x, EU directives 3.5x), the most under-dense are also legal (court procedure 0.36x), which says the length weighting moved some reference mass between neighbouring legal cells.

The shipped map also reads the reservoir's language the same as the build did 99.4% of the time, so per-language centering at run time matches the build.

On held-out English web (C4, 40,000 records), the shipped map reaches 3,690 cells and 68.4% effective coverage (previous: 3,493 cells, 63.7%); 18% of reached cells read above 2x and 28% below 0.5x, which is what a broad but uneven crawl should look like. Its densest cells are *Crushers, grinding mills and mining machinery* (13.7x), *Web images, wallpapers and picture files* (10.9x) and *Homes for sale with floor plans* (8.1x), which anyone who has read C4 will recognise.

### 7. Does each cell hold one subject? (B7)

**Question.** Forget the names for a moment. Are the cells themselves groups of records about one thing?

**Answer.** Mostly, and much more than before. Cells an outside encoder rates as incoherent halved.

Twenty random members of every cell, drawn from the build's reservoir, were embedded with a different encoder (paraphrase-multilingual-MiniLM-L12-v2), centred per language, and scored by their mean pairwise similarity. Unrelated records score 0.00; on the first atlas-v3 the cells a blind reader called grab-bags sat mostly under 0.15.

| map | median coherence | cells under 0.10 | 95% interval | cells under 0.05 | cells under 0.15 | reference mass in cells under 0.10 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **atlas-v3** | 0.244 | 75 (1.83%) | [1.46, 2.29] | 9 | 7.5% | 2.6% |
| previous v3 | 0.251 | 147 (3.59%) | [3.06, 4.20] | 34 | 9.5% | 3.9% |

Difference, shipped minus previous: −1.76 points of cells under 0.10 [−2.52, −1.05], a real halving. The median moved −0.007 [−0.012, −0.002], a hair lower: the rebuild cleaned the tail, it did not tighten the typical cell.

**How to read this.** The release gate set on 13 September was "at most 1.5% of cells under 0.10". The shipped map fails it by 14 cells (75 against 61). The leftover incoherent cells are, on inspection, translated prompt templates from the instruction-training data and very short web and encyclopedia fragments; 43 of them share the single honest name *Mixed short web fragments with no shared subject*.

### 8. Are the names right? (B5, B9)

**Question.** When the report says my data sits in "Household budgeting, saving and personal financial planning", is that what is there?

**Answer.** Usually. Read blind, 85% of cell names are accurate and none is a soup. The name of the cell that holds most of a held-out dataset describes that dataset outright 32% of the time (25% before) and at least partly 83% of the time; the gain over the previous map is within noise.

Three readings, from most to least mechanical.

**With the map's own encoder** (B5). Embed each name and ask which of the 4,096 cell centres it is nearest. A name should be nearest its own cell. Median rank of the own cell: 3 (previous 4); top-1 35.5% (31.2%); top-10 69.6% (65.2%). District names: top-1 71.1%, top-5 89.1%. Placing the build's own exemplar texts back on the map returns them to their own cell 45% of the time, to their own district 67%, and to a cell among the five nearest 75%. Name hygiene: 6 names contain a language word (21 before), and 59 names repeat another name; 43 cells share the single honest name *Mixed short web fragments with no shared subject*.

**With an outside encoder** (B9a). The same test with MiniLM embeddings of the names and of 20 members per cell, so the map's encoder cannot grade its own work:

| map | own cell median rank (of 4,096) | top-1 | 95% interval | top-10 | top-100 | district names top-1 (of 256) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **atlas-v3** | 3 | 35.2% | [33.8, 36.7] | 73.8% | 93.4% | 75.0% |
| previous v3 | 3 | 32.2% | [30.7, 33.6] | 70.0% | 91.6% | 73.8% |

A random name would rank at 2,048. By stamped kind on the shipped map: subject cells rank their own name at median 2 (81% in the top 10); form cells at 38; mixed cells at 30. So the outside encoder agrees with the map about which cells are hard to name, which is itself evidence the kinds were stamped honestly.

**Blind, by a reader** (B9b, B9c). The reader saw 218 shuffled items, each a held-out dataset or MMLU subject and the name of the cell holding most of it, without knowing which map the name came from, and scored 0 (wrong), 1 (partly) or 2 (describes it).

| map | describes it | 95% interval | at least partly | wrong | on the 52 datasets | on the 57 MMLU subjects |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| **atlas-v3** | 32.1% | [24.1, 41.4] | 82.6% | 17.4% | 26.9% | 36.8% |
| previous v3 | 24.8% | [17.6, 33.6] | 84.4% | 15.6% | 26.9% | 22.8% |

Paired on the same 109 slices: the shipped map's name was better on 24, the previous map's on 18, tied on 67; mean difference +2.8 points [−4.1, +9.6], not a proven difference. The shipped map's wrong answers are mostly the code datasets and maths sets that fell into mixed cells; the previous map's are code datasets named as *Product listings spanning cosmetics, livestock and lab gear* and chat data named as *NLI premise-hypothesis pairs*.

The reader then read 40 cells (five from each eighth of the coherence range, ten members each, spread from the cell's centre to its edge) with their current names:

| what the reader saw | count | share | 95% interval |
| --- | ---: | ---: | --- |
| cell is about one subject | 30 | 75% | |
| cell is one form of text (product listings, forum boilerplate, podcast transcripts) | 3 | 7.5% | |
| cell is a grab-bag | 7 | 17.5% | [8.7, 32.0] |
| name accurate | 34 | 85% | [70.9, 92.9] |
| name vague | 5 | 12.5% | |
| name false-specific | 1 | 2.5% | |
| name a soup, or wrong | 0 | 0% | |

Of the 7 grab-bags, 4 carried an honest *Mixed ...* name, 1 a name that describes what links them (*Women and girls in short descriptive sentences*), 1 a vague name, 1 a false-specific one (*Obituaries recounting lives and survivors*, where 3 of 10 members were obituaries and 3 were translated prompt templates). The reader's content call matched the stamped kind on 33 of 40 cells; the 3 disagreements where the map said mixed and the reader saw a subject were all cases of the map being too modest (a coherent cookware cell named *Mixed kitchen and cooking notes with no shared subject*; a coherent theoretical-physics cell named *Mixed physics-paper fragments*).

Grab-bag rate by coherence band: under 0.15, 1 of 1; 0.15 to 0.20, 3 of 9; 0.20 to 0.25, 1 of 11; above 0.25, 2 of 19. The coherence score does find the grab-bags, but the band just above the 0.10 gate still holds a third of them.

**How to read this.** The gates from the 13 September audit were: at most 7% grab-bags among judged cells, no soup names, at least 70% accurate names. The names pass (0 soups, 85% accurate). The grab-bag gate fails (17.5%, interval 9 to 32%), but with a difference from before: the grab-bags now say so on the label. The reader is a language model in the session, not a person; the sheets in `results/b7_judge/atlas-v3/` and `results/names_blind.json` let a human repeat the read.

### 9. Records the map can only locate, not describe (B9d)

**Question.** How much of my corpus will come back labelled "Mixed ... with no shared subject"?

**Answer.** About one record in eight on varied data, one in eleven on English web, and most of a code dataset.

The shipped map stamps a kind on every cell: 3,502 subject, 146 form, 448 mixed. The 448 mixed cells hold 11.1% of the map's reference mass. Share of placed records that land in them:

| data | share in mixed cells |
| --- | ---: |
| 52 held-out datasets | 14.0% [13.7, 14.3] |
| MMLU questions | 15.5% |
| C4 English web | 9.4% |

Datasets most affected: Python text-to-code programs 81%, competition algebra 32%, arXiv maths papers 30%, open-web-math 27%, OpenMathInstruct 26%. The previous map cannot be scored the same way because it stamped no kinds and its names admitted almost nothing (15 cells by name words, 0.3% of records); B7 says it had twice as many incoherent cells, so its records were in grab-bags just as often, under confident names.

**How to read this.** A record in a mixed cell is still placed and still counts toward coverage and comparison; only the name is uninformative. This number is the honest cost of honest naming, and it points at two things worth doing: giving code its own handling in the encoder input policy, since the policy damps exactly the short, symbolic tokens code consists of, and merging the 43 cells that share one mixed name.

### 10. Speed (B6)

**Question.** How long does a default run take?

| map | 500,000 records | language detection | encoding | placement | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| **atlas-v3** | about 2 min 10 s | 37 s | 82 s (6,067 records/s) | 11 s | 5.3 GB |
| previous v3 | about 1 min 50 s | 37 s | 59 s (8,436 records/s) | 11 s | 4.6 GB |

Same machine, same 500,000 records (mean 1,500 characters), measured minutes apart. The input policy adds 23 seconds per 500,000 records. The real command on three small corpora (1,600 geometry problems; 12,000 records from four sources; 16,000 web pages in 8 languages) ran in 3, 4 and 6 seconds and wrote the HTML, Markdown and JSON reports each time; the printouts are under `runs/`.

## What the results say about the map, and what to do

1. **Ship it, with the code caveat stated.** On every test that measures the map's stated purpose (subject sorting, language independence, coverage detection, honest density, honest names) the rebuilt map equals or beats the one it replaced, and the two regressions (code placement, 2.7 points of kind accuracy) are consequences of the same design choice that produced the gains.
2. **Code needs its own treatment.** The input policy should not damp symbol and short-cased tokens on records the extractor already knows are code, and the code axis reached only 10% of its byte target in the corpus. Until then, users placing code corpora should expect a large share of "Mixed" cells and should read the district level.
3. **Comparison needs a sample-size note in the CLI.** The "new" figure at cell level is sampling noise below 20,000 records per side. Printing the sampling floor beside it, or aggregating to districts under that size, would stop users reading noise as novelty.
4. **The coherence gate should be re-examined, not lowered.** 75 cells fail it against a budget of 61; 43 of them share one honest name and could be merged into far fewer cells without losing anything a user can read. The band from 0.15 to 0.20 still holds a third of the grab-bags, so the gate at 0.10 is a floor, not a guarantee.
5. **Repeat the blind reads with a human.** Both reads were done by the model in the session. They are cheap to redo from the saved sheets, and the two headline shares (85% accurate names, 17.5% grab-bags) deserve a second reader before they go in the changelog.

## Glossary

- **Cell, district.** The 4,096 small areas and 256 large areas of the map. A record is placed in exactly one cell; the district is the cell's parent.
- **Off the map.** A record whose similarity to its nearest cell is below the cutoff. The cutoff is the 2nd percentile of similarity over a balanced sample of the build corpus, so about 2% of that mix falls off by design.
- **Density.** Your share of a cell divided by the map's reference share of the same cell. 1.0x means your corpus has as much there as the map does.
- **Effective coverage.** Sum over cells of min(1, density): how much of the map you fill to at least the map's own level.
- **Majority-vote score.** Label each cell by the commonest source among training records, then score test records by that label. The weakest possible reading of the map.
- **Luck, ceiling, kept share.** Luck: always guess the commonest answer. Ceiling: a trained classifier on the same vectors. Kept share: how far from luck to ceiling the map gets.
- **Paired difference.** Two maps scored on the same records; the difference is resampled 1,000 times. Its interval excluding zero means the maps differ, not the sample.
- **Coherence.** Mean pairwise similarity of 20 cell members under an encoder the map never used. 0.00 is unrelated text.
- **Grab-bag, form, subject.** A cell whose members share nothing; whose members share a type of text but not a topic; whose members share a topic.
- **Sampling floor.** What a statistic would show if both sides were random draws from the same distribution of the same size.

## Appendix: technical detail

### A. Statistics with more cells

These reward finer grids and are compared only between the two v3 maps.

| test | atlas-v3 | previous v3 |
| --- | ---: | ---: |
| B1 source: AMI cell / district | 0.273 / 0.299 | 0.291 / 0.327 |
| B1 source: NMI cell / district | 0.406 / 0.321 | 0.414 / 0.349 |
| B1 source: purity cell | 0.396 | 0.394 |
| B1 kind: AMI cell | 0.226 | 0.244 |
| B2 MMLU: AMI cell / district | 0.311 / 0.375 | 0.307 / 0.364 |
| B2 MMLU-Pro: AMI cell / district | 0.274 / 0.324 | 0.284 / 0.322 |
| B3 web, 9 languages: NMI(cell, language) / NMI(district, language) | 0.188 / 0.091 | 0.180 / 0.091 |
| B3 web: dominant-language lift, cell | 4.28 | 4.14 |
| B3 Wikipedia, 13 languages: NMI cell / district | 0.212 / 0.107 | 0.216 / 0.117 |
| B3 Wikipedia: lift, cell | 5.51 | 5.62 |
| B3 without centering, web: NMI cell / lift | 0.257 / 5.45 |  |
| B1 cells used by the 51,489 panel records | 3793 | 3599 |

B1 soft top-5 accuracy (source is among the mapped labels of the five nearest cells with weight ≥ 0.15): 48.3% against 51.1%. B1 code language (11 languages, 13,000 records): 32.3% against 34.4%, luck 23.1%, nearest-mean ceiling 55.0% / 58.3%.

### B. Injection z-scores, shipped map, cell level

| slice | home cells | base mass in home cells | z at 0.25% | 0.5% | 1% | 2% | 5% | 10% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EU law | 19 | 0.03% | 23.5 | 45.9 | 88.8 | 182 | 464 | 946 |
| Python commits | 69 | 2.26% | 2.7 | 5.4 | 10.6 | 20.8 | 51.5 | 107 |
| maths word problems | 244 | 9.22% | 1.0 | 2.1 | 4.3 | 8.6 | 22.4 | 46.0 |
| Turkish instructions | 897 | 20.05% | 0.5 | 1.1 | 2.1 | 4.0 | 10.2 | 20.5 |
| text-to-SQL | 431 | 13.37% | 0.8 | 1.6 | 3.1 | 6.5 | 16.8 | 34.7 |
| biomedical QA | 74 | 1.17% | 3.7 | 7.4 | 15.2 | 29.9 | 75.8 | 152 |
| Turkish web | 960 | 21.97% | 0.4 | 0.9 | 1.9 | 3.5 | 8.8 | 18.0 |

Method: home cells are the cells holding 80% of a disjoint 5,000-record reference sample of the slice; z is the Poisson excess of records in those cells in the mixed corpus over the base-only expectation; the estimate is (mixed share in home cells − base share) / (reference share − base share).

### C. Test-retest, all corpora, both v3 maps

New between halves at cell level / sampling floor / new at district level, shipped map: ultrachat 5,000: 18.3% / 14.3% / 0.0%; 20,000: 4.1% / 3.0% / 0.0%. Turkish web 5,000: 18.1% / 13.4% / 0.0%; 20,000: 4.0% / 3.1% / 0.0%. Previous map within 1 point on every row. Histogram cosine between halves at 20,000: 0.90 (C4), 0.98 (ultrachat), 0.97 (Turkish web). Floor: 20 multinomial simulations from the pooled two-half histogram.

### D. Off-map calibration detail

Shipped map: `off_atlas_calibration` records a 2nd-percentile rule over a 2,000,000-row language-and-axis-balanced draw (750 strata), p1 0.3147, p2 0.3352, p5 0.3708, p50 0.5859; the proportional-draw estimate of p2 is 0.3624. Held-out in-family prose (fineweb 20,000 + English Wikipedia 3,000 + Turkish web 3,000): p1 0.398, p2 0.425, p5 0.468, p50 0.662. Junk medians: base64 0.36, number columns 0.31, DNA 0.34, hex 0.42, minified JS 0.35, random letters 0.40, random characters 0.43; the prose p1 is 0.398, so most junk sits just under the prose band and a cutoff near 0.40 would reject most of it at the price of 1% of prose. The 14-topic probe file's four junk blobs score 0.35 to 0.44 and are all placed.

### E. Density calibration detail

Reservoir sample of 200,000: raw quotient within 0.8x to 1.25x for 56.4% of cells (multinomial expectation 85.7%); CLI shrunk estimate median 0.98, prior strength 10.0, unreached-cell density 0.02. Chi-square of placed counts against stored reference shares, relative to a multinomial draw: 4.76 (shipped), 4.66 (previous). Interpretation: a real but small mismatch between build-time counting and run-time placement, equal on both maps, absorbed by the Dirichlet smoothing.

### F. Blind reading protocol

Names over held-out slices: `bench/b9_names.py make_sheet` writes 218 items (109 slices x 2 maps), shuffled with a fixed seed, product hidden, key in `results/names_key.json`. The reader scored `results/names_scores.json`. Rubric: 2 = a reader would recognise the dataset from the name; 1 = right area or a real sub-theme of a broad dataset; 0 = wrong, or a "Mixed" name over a specific dataset. A "Mixed web" name over a web crawl was scored 1.

Cells: 40 cells, 5 per coherence octile, 10 members each drawn by `spread_sample` from the cell's centre to its edge and shuffled, name shown, coherence hidden. Content: subject / form / grab-bag. Name: accurate / vague / soup / false-specific / wrong. An honest "Mixed ... no shared subject" over a genuine grab-bag was graded accurate. Reader agreement with the stamped kind: 33 of 40.

### G. Reproduction

```bash
cd experiments/atlas-v3-benchmark
./run_core.sh                              # B1..B5, B7 on both reservoirs, B6
../../.venv/bin/python -m bench.b8_readout
../../.venv/bin/python -m bench.b9_names   # alignment, mixed-cell shares, blind sheets
# score the two sheets, then
../../.venv/bin/python -m bench.b9_names --score
../../.venv/bin/python -m bench.summary    # every table above
```

Inputs: the shipped `atlas-v3.npz` (sha256 `575e5b01…`), the previous build at `/Volumes/ck512/dropoutt-atlas-v2/release-v3-textnorm/previous-shipped-4203fc3a/`, the two build reservoirs under `/Volumes/ck512/dropoutt-atlas-v2/{work-v3-textnorm,work}/reservoir.jsonl`, the local atlas-corpus cache and the Hugging Face datasets cache. Wall time for the whole suite on a 14-core laptop: about 45 minutes, of which 25 are the two MiniLM passes.
