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
| `T0-SCHEMA-005` | Record content sits in keys the layout never reads | — | sft |
| `T0-FORMAT-001` | Plain-text files are holding structured records | — | — |
| `T0-GEN-001` | Generator scaffolding sits outside the records | — | sft |
| `T0-REASON-001` | Only some responses carry a reasoning trace | — | — |
| `T0-TRUNC-002` | Responses stop at a generation length cap | — | sft |
| `T0-QUAL-001` | Documents whose lines mostly do not end in punctuation | corpus profile | — |
| `T0-QUAL-002` | Documents built mostly from very short lines | corpus profile | — |
| `T0-QUAL-003` | Documents repeating their own lines | corpus profile | — |
| `T0-ROLE-001` | Conversation role structure is valid | — | sft |
| `T0-ROLE-002` | Role names are not the canonical vocabulary | — | sft |
| `T0-TMPL-001` | Data is already formatted with a chat template | — | sft |
| `T0-TMPL-002` | Records fail to render with the target chat template | chat template | sft |
| `T0-MASK-001` | Records contribute zero trainable tokens | tokenizer, chat template | sft |
| `T0-MASK-002` | Stop token is outside the trainable span | tokenizer, chat template | sft |
| `T0-TRUNC-001` | Records exceed the sequence length | tokenizer, seq len | sft |
| `T0-PACK-001` | Packing efficiency under concat-and-chunk | tokenizer, seq len | — |
| `T0-ENC-001` | Text encoding is damaged | — | sft, corpus |
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

### Checks for data that came out of a generator

Synthetic SFT data fails differently from collected data. The generator is a
language model, so its mistakes are fluent: the file parses, the records are
well-formed, the text reads well, and something is still wrong. Every threshold
below was calibrated against a real 6,097-record Turkish generation run, and two
of these checks are deliberately silent on it — which is the evidence that they
are not simply firing on everything.

**`T0-FORMAT-001` — records inside .txt files.** Generation pipelines write JSON
to whatever path they were handed. Read at face value, a folder of such files
becomes one corpus document per file: the record count is wrong by orders of
magnitude, the profile is inferred from the wrong shape, and every conversational
check is skipped without appearing in the skipped list, because as far as the
scanner is concerned there was no conversational data. On the corpus above this
was the difference between 251 documents and 6,097 records, and between five
findings and fifteen. See `dropoutt.sniff` for how the decision is made and what
stops it firing on prose that quotes JSON.

**`T0-SCHEMA-005` — content in unread keys.** A record can be well-formed and
still lose its answer. Twenty-six records in that corpus carried a complete
assistant turn in a top-level `assistant` key sitting *beside* `messages` rather
than inside it. Every key the trainer reads was valid, so nothing complained; the
conversation simply ended on a user turn and trained on nothing. Bookkeeping
columns — `id`, `source`, `language` and the rest — are ignored.

**`T0-REASON-001` — inconsistent reasoning traces.** 11.6% of assistant turns in
that corpus opened with a `<think>` block and 88.4% did not. Both shapes are
valid records, so nothing noticed. Trained as-is the model learns to emit
reasoning about one time in eight, unpredictably, and inference-time parsers that
strip the block find nothing to strip in most responses. A dataset that is 100%
or 0% reasoning is a decision; one that is 12% reasoning is an accident.

**`T0-GEN-001` — scaffolding between records.** Text sitting outside the records
is whatever the generating model emitted around its output: a control tag it was
told to honour and echoed instead (`<thinking_mode>off</thinking_mode>`), a
preamble, reasoning that escaped the record. Harmless where it sits, but evidence
that the generator was not doing what the prompt asked — which usually means the
records are affected too.

**`T0-TRUNC-002` — generation length cap.** When a run hits `max_tokens` the
response stops mid-sentence and the record still looks fine. The signature is not
that responses are long, it is that too many are the *same* length. Two tests
must both pass: the top length bucket holds at least 5% of responses, and it
holds at least three times as many as any bucket within 250 characters beneath
it. The second test is what stops it firing on a corpus whose responses simply
take a small number of discrete lengths.

### Corpus-profile quality filters

The three line-shape filters FineWeb publishes, at FineWeb's own thresholds:
under 12% of lines ending in punctuation, over 67% of lines shorter than 30
characters, over 1% of characters in repeated lines.

They are **reported, never applied**, and every one is `unverified`. FineWeb
selected them by measuring downstream effect on their corpus; a threshold that
helped there is a hypothesis here. Each finding states the share of documents
that would be dropped and leaves the decision where it belongs.

All three describe line shape, which says nothing about a chat record, so they
run under the `corpus` profile only.

## Tier 1 — statistical

| id | title | needs | blocks under |
| --- | --- | --- | --- |
| `T1-NDUP-001` | Near-duplicate records | — | — |
| `T1-DUP-002` | The same prompt is answered two different ways | — | — |
| `T1-OVERLAP-001` | Datasets overlap with each other | 2+ datasets | — |
| `T1-ATLAS-001` | The corpus sits in very few topical regions | atlas | — |
| `T1-ATLAS-002` | A crowded region holds near-identical records | atlas | — |
| `T1-CONTAM-001` | Training data overlaps evaluation benchmarks | benchmark index | sft, corpus, preference |
| `T1-LANG-001` | Language composition and detection confidence | langid | — |
| `T1-LANG-002` | Records deviate from their dataset's main language | langid | — |
| `T1-LANG-003` | Script does not match the detected language | langid | — |
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
| `strict` (default) | word 5-gram | 104 | 8 × 13 | 0.85 |
| `fineweb` | word 5-gram | 112 | 14 × 8 | 0.75 |
| `hf-neardedup` | word 5-gram | 256 | 32 × 8 | 0.70 |

The default is `strict`. At 0.75 the check fired on records that share a
template or a category column but say different things, and a delete-shaped
finding next to those is wrong. Records that are exact copies of each other
are counted by `T0-DUP-001` and excluded here, so the two numbers no longer
double-charge the same rows.

**`T1-DUP-002` is the opposite of deduplication.** Duplicate detection finds
records that are identical and calls them redundant. The more damaging case is
the same prompt appearing several times with *different* answers — not
redundancy but contradiction, invisible to every duplicate check because the
records are genuinely distinct. The gradient points two ways at once. The usual
causes are merging two sets that overlap on prompts, or repeating a generation
run under a different system prompt. Prompts are compared after whitespace
normalisation only, so near-misses are not counted.

**`T1-OVERLAP-001` is directional.** Read it as "this share of *row* also appears
in *column*". A small dataset contained in a large one shows 100% one way and 1%
the other, and the asymmetry is the finding.

**`T1-ATLAS-*` see shape that no per-record check can.** A record is never
individually wrong for sitting in a crowded region, or for failing to sit in an
empty one. `T1-ATLAS-001` fires only on genuine narrowness — an effective region
count at or below 10, or a single region holding a quarter of the corpus — not on
the gap between occupied and effective regions, which every long-tailed
distribution has. `T1-ATLAS-002` is the case near-duplicate detection cannot
reach: a thousand records generated from one template with the nouns swapped
share almost no shingles, but pile into one region and sit very close together.
Both require a crowded region *and* high within-region cohesion, so a large topic
with varied writing does not trip them.

**`T0-DEGEN-001` does not call extraction degenerate.** It once flagged a
response as copying the prompt whenever one contained the other, which on a real
corpus was wrong 38 times out of 38: every hit was a multiple-choice answer or
extractive QA, where being a substring of the prompt is the task. Containment now
has to account for at least 90% of the other side.

**`T1-CONTAM-001` applies the Tülu 3 rule as published.** An evaluation instance
is contaminated when more than 50% of its tokens are covered by 8-gram matches
against a single training instance; a training set is contaminated when more
than 2% of any evaluation's instances match. Removing contamination usually makes
your reported score go *down*.

**`T1-LANG-*` are gated on length and confidence.** Nothing under 40 characters
produces a language finding. Bag-of-n-gram identification is unreliable on short
text and on closely related languages, and that is where false positives come
from. When the reduced-accuracy fallback backend is in use, findings say so.

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
