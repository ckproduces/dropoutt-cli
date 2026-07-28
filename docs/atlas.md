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
5. **Name regions from TF-IDF terms.** Deterministic and offline.
6. **Project to 2D with PCA.** Deterministic.
7. **Record the quality numbers inside the artifact** so they can be printed
   next to any coverage figure.

Total build cost is a few minutes of CPU and well under $20 of compute even at
full scale.

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

## Off-atlas data, and when coverage is withheld

If a corpus consists of text the reference corpus contains nothing like, every
record collapses into one or two distant bins and the coverage numbers become
meaningless.

So the atlas reports an **off-atlas rate**: the share of records whose cosine
similarity to their nearest centroid falls below a threshold calibrated at build
time. Above 10%, coverage numbers are **suppressed rather than displayed**, with
the reason stated.

The rate is reported **per language as well as globally**. For a language the
embedding model represents poorly, topical assignment is unreliable no matter how
good the clustering is, and a global average would hide that.

## Reading the quality numbers

`dropoutt atlas` prints two figures that belong next to any coverage number:

| number | meaning |
| --- | --- |
| level-0 held-out accuracy | how well the taxonomy probe generalises. Low accuracy means category counts look precise and are not. |
| region purity by taxonomy | mean share of each region occupied by its most common category. Low purity means regions are mixing topics. |

## Tiers

Tiers are **resolution levels of one hierarchy**, not separate atlases.
Independent atlases would destroy the property the atlas exists for, because
fingerprints computed against different atlases cannot be compared and the shared
index would fragment. Coarse mass is exactly the sum of the fine masses beneath
it, so a free-tier fingerprint and a paid-tier fingerprint remain comparable and
upgrading re-aggregates existing history rather than invalidating it.

This release ships level 0 plus level 1 (256 regions) in the package.

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
