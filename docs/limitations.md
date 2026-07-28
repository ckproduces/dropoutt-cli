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
  this dataset contribute" and "select under a token budget".
- **Fingerprint diffing.** Fingerprints are written but `dropoutt diff` is not
  implemented.
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

## The atlas is a first version

The shipped `atlas-lite-v0` is built from a few hundred thousand records, its
level-0 taxonomy probe is trained on labels bootstrapped from dataset
provenance rather than from human annotation, and its regions are named from
TF-IDF terms rather than by a language model. Its own held-out accuracy and
region purity are recorded inside the artifact and printed by
`dropoutt atlas info`; read them before trusting a coverage number.

Atlas coverage is not yet wired into `dropoutt scan`. The artifact and its
loader exist; the per-record assignment step does not.

## Things deliberately out of scope

- **Document extraction.** Converting PDFs or HTML dumps to text is a solved
  problem with mature tools. dropoutt checks the quality of their output.
- **Fixing your data.** Every finding names a fix; none of them are applied.
  A tool that both diagnoses and silently rewrites is a tool nobody can audit.
- **Anything that phones home.** The scan makes network calls in exactly two
  places: resolving `--model` against the Hub, and fetching an embedding model
  for the atlas. Both are cached and both are disabled by `--offline`.
