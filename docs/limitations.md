# Limitations of this release

Stated plainly, because a scanner that overstates itself is worse than one that
does less.

## Nothing here is calibrated

Every finding is labelled `unverified`. That is not modesty, it is accurate: no
measured effect size links acting on any of these findings to a change in model
quality. They are structural observations about your data.

Two exceptions in kind, though still not in evidence grade:

- Structural defects (empty loss masks, records whose assistant span is entirely
  truncated away) are deterministic waste. Those records were contributing
  nothing regardless of what any experiment would show.
- Contamination removal gives you a *truer* evaluation. Your reported score will
  usually drop. That is the point, but it means "improvement" is the wrong word.

Everything else — deduplication, quality filtering, style, language mix — is
probabilistic and setting-dependent, and this release does not attempt to say by
how much.

## Not implemented yet

- **Tier 2**, meaning anything requiring embeddings per record: semantic
  deduplication, difficulty scoring, quality classifiers, cartography regions.
- **Tier 3 verdict runs**, the micro-ablation harness that would convert
  findings from opinion into measurement.
- **`marginal()` and `plan()`**, the operations that answer "what does adding
  this dataset contribute" and "select under a token budget". Nothing shipped
  shows the geometry two datasets stand in; it does not price the addition or
  select a mixture.
- **The hosted control plane**, history, and approvals.

## Known weaknesses

**Language identification is the largest source of false positives.** The
bundled `lid.176` model is unreliable below roughly 40 characters and confuses
Turkish with Azerbaijani, Turkmen and Crimean Tatar. Findings are gated on both
length and confidence to compensate, but the gate is a blunt instrument. GlotLID
would fix this properly and is a 1.7 GB download, so it is not bundled.

**Scale is untested above roughly a million records.** The design targets 8M
records on a laptop, and the streaming pass is written for that, but the
MinHash signature store is held in memory and the contamination accumulator
grows with the number of matching eval instances. Neither has been profiled at
that scale. Treat multi-million-record scans as unproven.

**The scan is single-process.** `tokenizers.encode_batch` parallelises
internally, but the pure-Python normalisation path does not. A process pool
would help and is not implemented.

**Corpus-relative checks use fixed thresholds.** Style tics fire above a fixed
15% rate rather than a distribution-aware one. Refusal boilerplate uses a fixed
pattern list, which is correct for identity leakage but blunt for genuinely
generic phrasing.

**Parquet row groups are read whole.** There is no column projection, so scanning
a wide Parquet dataset reads more than it needs.

**There is no built-in way to compare two corpora.** The fingerprint carries
everything needed — the full region histogram is written under `region_counts`
precisely so two scans are comparable — but the command that read it was cut
before 1.0 rather than frozen half-finished. Read the two fingerprints yourself
in the meantime.

**The map's reference distribution is the reference corpus's, not the
world's.** Density divides your share of a cell by the share of the
163,452,464 reference records that landed there (`region_size`), so "3× the
map" is measured against the corpus plan — nine axes with byte targets and a
non-English floor — and not against a natural population. The sentence "the
map spends 15 of its 4,096 places on that subject" counts cells, which is how
much resolution the clustering gave the area, and is a different number from
the area's share of the reference mass.

## What the atlas still lacks

The shipped default, `atlas-v3`, is built from 163,452,464 records over 244
sources, 60.6% non-English by bytes, and every one of its 4,096 cells and 256
subject areas is named by hand. It carries no taxonomy: subject areas are
k-means over the same vectors as the cells, so there is no held-out accuracy
to report and none travels in the fingerprint. What it does carry is a language
probe — balanced accuracy for language 0.528 on raw vectors and 0.365 after
normalization, on 300,000 held-out rows — and the calibration behind its
off-atlas cutoff of 0.3538; read both before trusting a coverage number. The
names describe the reference corpus, not yours, and the reference distribution
is the corpus plan's, not the world's. `atlas-v2` and `atlas-v2-lite` run on
the loader's 0.35 fallback cutoff rather than a stamped one, which puts 12–18%
of ordinary held-out prose off-atlas on them. `atlas-v1-lite` is bundled only
so fingerprints placed on it can be re-read.

## Things deliberately out of scope

- **Document extraction.** Converting PDFs or HTML dumps to text is a solved
  problem with mature tools. dropoutt checks the quality of their output.
- **Fixing your data.** Every finding names a fix; none of them are applied.
  A tool that both diagnoses and silently rewrites is a tool nobody can audit.
- **Anything that phones home.** There is no telemetry or hosted scan service.
  Network access is limited to fetching tokenizers, tokenizer configuration,
  and the atlas embedding model from the Hugging Face Hub. These are cached and
  all access is disabled by `--offline`, `DROPOUTT_OFFLINE=1`, or
  `HF_HUB_OFFLINE=1`.
