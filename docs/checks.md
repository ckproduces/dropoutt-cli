# Check catalog

Run `dropoutt checks` for the live list, or `dropoutt checks T0-MASK-001` for one
check in detail.

Identifiers are `T{tier}-{GROUP}-{nnn}` and are **never renumbered**. Users mute
checks by id, and those mutes live in version control.

Every check declares what it needs. When a requirement is missing the check is
reported as skipped alongside the single flag that unlocks it, rather than
silently omitted.

## Tier 0 — structural, CPU only

These are bugs rather than quality judgements. The direction is never arguable:
a record whose trainable span is empty contributes nothing to any gradient.

| id | title | needs | blocks under |
| --- | --- | --- | --- |
| `T0-SCHEMA-001` | Files are not training data | — | sft, corpus, preference |
| `T0-SCHEMA-002` | One folder contains several record layouts | — | sft |
| `T0-SCHEMA-003` | Records failed to parse | — | sft, corpus |
| `T0-SCHEMA-004` | Message content was not a string | — | sft |
| `T0-ROLE-001` | Conversation role structure is valid | — | sft |
| `T0-ROLE-002` | Role names are not the canonical vocabulary | — | sft |
| `T0-TMPL-001` | Data is already formatted with a chat template | — | sft |
| `T0-TMPL-002` | Records fail to render with the target chat template | chat template | sft |
| `T0-MASK-001` | Records contribute zero trainable tokens | tokenizer, chat template | sft |
| `T0-MASK-002` | Stop token is outside the trainable span | tokenizer, chat template | sft |
| `T0-TRUNC-001` | Records exceed the sequence length | tokenizer, seq len | sft |
| `T0-PACK-001` | Packing efficiency under concat-and-chunk | tokenizer, seq len | — |
| `T0-ENC-001` | Text encoding is damaged | — | sft, corpus |
| `T0-ENC-002` | Turkish dotted and dotless I damage | — | — |
| `T0-DUP-001` | Exact and whitespace-identical duplicates | — | — |
| `T0-DEGEN-001` | Degenerate responses | — | — |

### The four that matter most

**`T0-SCHEMA-001` — not training data.** Agent session logs, telemetry and
rollout traces are structurally close enough to chat data that importers ingest
them happily. Detecting them is more useful than forcing them into a layout and
reporting confident nonsense about the result.

**`T0-ROLE-002` — role vocabulary.** The quietest way to lose an entire dataset.
A ShareGPT record uses `from: "gpt"`. A trainer that masks on `role ==
"assistant"` finds no assistant span, produces an all-ignored label vector, and
drops the record. The counters that would have told you are frequently computed
and then discarded.

**`T0-MASK-001` — zero trainable tokens.** The consequence of the above, measured
directly. These records cost tokens in the packed block, contribute nothing, and
appear nowhere in the training logs.

**`T0-TRUNC-001` — truncation.** Two different bad outcomes hide behind one
number. A record truncated from the end loses part of its answer. A record whose
*entire* assistant span falls beyond the limit teaches nothing at all, and that
subset is reported separately and escalates to blocking.

## Tier 1 — statistical

| id | title | needs | blocks under |
| --- | --- | --- | --- |
| `T1-NDUP-001` | Near-duplicate records | — | — |
| `T1-OVERLAP-001` | Datasets overlap with each other | 2+ datasets | — |
| `T1-CONTAM-001` | Training data overlaps evaluation benchmarks | benchmark index | sft, corpus, preference |
| `T1-LANG-001` | Language composition and detection confidence | langid | — |
| `T1-LANG-002` | Records deviate from their dataset's main language | langid | — |
| `T1-LANG-003` | Script does not match the detected language | langid | — |
| `T1-LANG-004` | Turkish text has lost its diacritics | — | — |
| `T1-PII-001` | Personal data and credentials in training text | — | sft, corpus, preference |
| `T1-IDENT-001` | Assistant identity leakage and refusal boilerplate | — | sft |
| `T1-STYLE-001` | Formulaic response openings | — | — |
| `T1-LIC-001` | Datasets have no recorded licence | — | — |

### Notes on specific checks

**`T1-NDUP-001` reports, it does not prescribe.** Two named presets ship, and the
report always states which was used and its parameters, because a deduplication
result is only arguable if the parameters are stated.

| preset | shingle | hashes | bands | threshold |
| --- | --- | --- | --- | --- |
| `fineweb` | word 5-gram | 112 | 14 × 8 | 0.75 |
| `hf-neardedup` | word 5-gram | 256 | 32 × 8 | 0.70 |

**`T1-OVERLAP-001` is directional.** Read it as "this share of *row* also appears
in *column*". A small dataset contained in a large one shows 100% one way and 1%
the other, and the asymmetry is the finding.

**`T1-CONTAM-001` applies the Tülu 3 rule as published.** An evaluation instance
is contaminated when more than 50% of its tokens are covered by 8-gram matches
against a single training instance; a training set is contaminated when more
than 2% of any evaluation's instances match. Removing contamination usually makes
your reported score go *down*.

**`T1-LANG-*` are gated on length and confidence.** Nothing under 40 characters
produces a language finding. Bag-of-n-gram identification is unreliable on short
text and on closely related languages, and that is where false positives come
from. When the reduced-accuracy fallback backend is in use, findings say so.

**`T1-LANG-004` is specific to this market.** Language identification will
confidently call `degil mi` Turkish, and it is, but it is damaged Turkish. No
general-purpose tool checks for it.

**`T1-PII-001` uses checksums where they exist.** A bare eleven-digit regex
matches every order id and timestamp in a corpus, so the Turkish national ID
pattern validates its checksum, IBANs validate mod-97 and card numbers validate
Luhn. Matched values are never written to any output; only a masked form.

**`T1-STYLE-001` fires on frequency, not presence.** One "Certainly!" is a
sentence. "Certainly!" opening forty percent of responses is a style your model
will inherit. The default threshold is 15%.

## Confidence

Every finding in this release is `unverified`. No measured effect size links
acting on any of them to a change in model quality. See
[limitations.md](limitations.md) and [design.md](design.md).
