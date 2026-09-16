# Naming atlas cells

How the fine cells (L2) and districts (L1) of an atlas build are named. The rules
come from the 13 September 2026 audit of atlas-v3, which found names that listed
unrelated topics ("Linux hardening logs, Brazilian court appeals and
sentence-rewriting prompts"), names that described a first letter ("Names and
entries starting with T"), and names narrower than their cells ("La Liga
footballer biography stubs" for players from every league).

`tools/apply_hand_region_labels_v3.py` and `tools/apply_hand_l1_labels_v3.py`
enforce the mechanical parts through `tools/atlas_label_rules.py`. The judgement
parts are below.

## Inputs

`tools/atlas_naming_worklist.py` writes one worklist per district: for every cell,
30 members spread from its centre to its edge, its source, language and axis mix,
its distinctive terms, its size, and its coherence under an encoder that is not
the atlas's own.

The 30 are drawn by rank: the cell's members are ordered by closeness to the
centre, cut into 30 bands of equal count, and one member is drawn at random from
each band. Each is shown with its rank, `[0%]` at the centre to `[97%]` at the
edge. Because the bands are equal, the 30 stand for the cell in proportion, and
no draw can come out all centre or all edge by luck.

Never name a cell from the records nearest its centre alone: in a cell with no
shared subject those few records are unrelated to each other, and naming them
produces a list of topics the cell does not have.

## Step 1: decide the kind

Read all 30 members before looking at the terms. A subject that holds only for
the first few ranks is the core of the cell, not the cell: the kind is decided
over all 30.

| kind | when | how the name reads |
| --- | --- | --- |
| `subject` | at least two thirds of the members share one subject (it may be broad) | the subject, at the level the members support |
| `form` | members share a format, genre or template, but their subjects vary | the form, and that the subjects vary: "Pipe-delimited tables on assorted subjects" |
| `mixed` | neither a subject nor a form covers two thirds of the members | starts with "Mixed", then what little is shared: "Mixed short web fragments with no shared subject" |

Coherence is a hint, not the verdict: below 0.10 a cell is usually mixed; above
0.25 a cell usually has a subject, or a shared template. Language does not count
as shared: the same subject in five languages is one subject, and five subjects in
one language are mixed.

## Step 2: write the name

- **Broad and true beats narrow and wrong.** If the members are footballers from
  many leagues, the name is about footballers, not one league.
- **A list must be facets of one subject.** "Churches, castles and medieval
  monuments" is one subject. "Skincare products, recipes and baby bottle reviews"
  is three, and the cell is mixed or form.
- **No first letters, initials or abbreviations as the subject.** A cell held
  together by initial letters is mixed.
- **No language names unless the language is the subject** ("Languages,
  linguistics and language learning"). Name the place rather than the nationality
  adjective: "Italy, its painters and architects".
- **Siblings differ by subject.** Two cells under one district may not share a
  name. If they truly share a subject, distinguish them by a facet the members
  show, never by an invented one.
- At most about ten words. No "various", "misc" or "etc." in a subject name.

## Step 3: name the district

After its cells: read the cell names, kinds and a few members of each. A district
whose cells share a subject gets that subject. A district whose cells are mostly
mixed or unrelated is `mixed`, and its name says so. Do not stitch the three
largest children's names together.

## Answer format

Fill the JSON skeleton beside each worklist:

```json
{"l1": 12, "l1_name": "Human rights, democracy and governance", "l1_kind": "subject",
 "cells": {"201": {"name": "Election monitoring and electoral reform", "kind": "subject"}}}
```

## Verification

After stamping, blind readers who did not write the names judge 160 cells
sampled across the coherence range (`experiments/atlas-v3-benchmark/bench/b7_cells.py
--make-judge --naming-dir <worklists>`, then `--score-judge`). Readers see 16
members spread the same way but not shown to the namer, wherever the cell has
enough. The names ship only if no judged name is a
soup and at least 70% are judged accurate; the map ships only if at most 1.5% of
cells score below 0.10 coherence and at most 7% of judged cells are grab-bags.
