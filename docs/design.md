# Design rules, and why they exist

Every rule here exists because of a specific published result or a specific way
this kind of tool fails. None of them are style preferences.

## 1. Removing more data is not automatically better

The FineWeb team deduplicated Common Crawl across all 96 snapshots at once,
which is the obvious thing to do. The result was 4 trillion tokens and a model
that scored *below* the RefinedWeb baseline. The oldest snapshots lost more than
90% of their content.

They then ran the diagnostic that settles it. They took the 2013-48 snapshot,
trained one model on the data that survived deduplication and another on the
data that had been thrown away, and **the thrown-away data produced the better
model**. Their fix was to deduplicate within each snapshot and never across, and
their stated conclusion is that the benefit comes from removing very large
duplicate clusters and that going further hurts.

A plausible explanation: text that reappears across the web over years tends to
be text people found worth republishing. Machine-generated spam is often unique,
because it is produced once and never mirrored. Under that condition, "delete
everything that appears more than once" deletes the good material and keeps the
unique junk.

**The rule.** A check may always report a measurement. A check may only
recommend deleting data when there is a measured effect size for that action in
a comparable setting. Otherwise the finding is labelled `unverified`, and the
tool reports cluster sizes rather than offering a delete button.

Every finding in this release is `unverified`. That is stated in the terminal
output, in the HTML report, and in every findings record.

## 2. Nothing blocks a run whose purpose was never declared

Blocking asserts that something is wrong *for a goal*. Without a declared target
there is no goal, so there is no basis for the assertion.

Findings therefore carry `would_block_under: ["sft"]` and the exit code stays 0
until the user passes `--target` or sets one in `dropoutt.toml`.

A checker that fails someone's build on an assumption it made by itself gets
uninstalled once and never reinstalled.

## 3. Degrade rather than error, and always say so

If schema induction cannot find a known layout, every record is treated as raw
text and the report says that is what happened. If a tokenizer reports offsets
into a normalized string we never see, the loss mask cannot be derived reliably,
so the mask-dependent checks are skipped with that reason rather than producing
wrong numbers. If the reduced-accuracy language backend is in use, every language
finding says so instead of presenting its guesses as equivalent.

Silent degradation is worse than an error, because the user acts on the output
either way.

## 4. Never report our own methodology as the user's bug

The clearest case is the stop-token check. When a chat template carries the
Hugging Face `{% generation %}` tag, we know exactly which characters the
template considers generated, and if the end-of-turn token falls outside that
region the model genuinely will not learn to stop.

When the template has no such tag — Mistral's does not, and neither does the
shipped Qwen3 template — spans are recovered by re-rendering with the assistant
content replaced by a sentinel and taking the difference. That method recovers
the assistant *content* and by construction excludes whatever the template
appends after it. Reporting that as a defect would be reporting our own
technique as their problem, so in that case the check emits an informational
note that names the convention and says explicitly that it is not a data defect.

## 5. Cross-dataset overlap is directional

If a 10,000-record dataset sits entirely inside a 1,000,000-record dataset, the
containment is 100% in one direction and 1% in the other. A symmetric Jaccard
reports something small and conceals the only fact worth acting on, which is
that the small dataset can be dropped.

The matrix is therefore computed and rendered as "this share of *row* also
appears in *column*", and a test asserts it is not symmetric.

## 6. The scan output must be safe to share

The report exists to be pasted into a pull request. So:

- PII matches are masked before they reach any output file. The check emits
  `ah***@***.com`, never the address, and deliberately omits the surrounding
  excerpt that would put the value back.
- A test plants a secret in a fixture and fails if it appears in the generated
  report.
- Control characters are mapped into the Unicode Control Pictures block rather
  than stripped, because a tool that reports control characters and then renders
  them invisibly has told the user nothing.
- Bidirectional overrides are neutralised, since one unbalanced RTL override in
  one record would otherwise reverse the layout of every finding after it.
- JSON destined for a `<script>` element has its angle brackets escaped as
  unicode sequences, so a record containing `</script>` cannot close the element.

## 7. Language is handled by supervision, not by altering the embedding space

An earlier version of this design proposed neutralising language geometrically:
computing a mean embedding per detected language, subtracting it, and projecting
out the components that predict language identity. That was rejected as a
foundation for three reasons.

It conditions the geometry on a label that is least reliable exactly where this
tool needs it most — short text, and closely related languages such as Turkish,
Azerbaijani and Turkmen. A misidentified record has the wrong centroid
subtracted and lands somewhere meaningless.

A language centroid does not encode only language. Turkish web text is not
translated English web text; it has a different topical distribution. Subtracting
its mean removes part of what Turkish corpora are *about*.

And it is not inspectable. When a user asks why a record landed in a region, the
honest answer would involve a hidden vector subtraction they cannot examine.

The adopted approach conditions on topic through supervision and alters nothing:
a supervised level-0 taxonomy, with fine clustering fitted within each category
rather than globally. See [atlas.md](atlas.md).

## 8. Identifiers are stable, and hashes cover everything that changes results

Check ids are never renumbered, because users mute checks by id and those mutes
live in version control.

The fingerprint id covers the pipeline version, the config, the content hash,
the tokenizer hash, the atlas hash and the language backend. Omitting any of the
last three would let a silent upgrade produce different findings under an
identical id, which would break the reproducibility claim outright.

## Sources

- FineWeb: [blog](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html),
  [paper](https://arxiv.org/abs/2406.17557)
- Tülu 3 decontamination rule: [paper](https://arxiv.org/abs/2411.15124)
- DataComp-LM: [paper](https://arxiv.org/abs/2406.11794)
- Essential-Web v1.0 taxonomy approach: [paper](https://arxiv.org/abs/2506.14111)
- Multilingual corpus audit: [Quality at a Glance](https://aclanthology.org/2022.tacl-1.4.pdf)
- Dataset licensing audit: [Data Provenance Initiative](https://arxiv.org/abs/2310.16787)
