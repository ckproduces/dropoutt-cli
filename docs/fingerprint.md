# Fingerprint schema

A fingerprint is a fixed-schema description of a dataset that can be compared
with any other fingerprint produced by the same pipeline version. It contains
no record excerpts or recoverable records. It does contain the scan root,
dataset names, aggregate measurements, and stable hashes, so apply the same
metadata policy used for build manifests before pasting it into a pull request
outside the VPC.

```bash
dropoutt scan ./data          # writes .dropoutt/fingerprint.json
```

## The identifier

```
fingerprint_id = "fp_" + hash(
    pipeline_version, config_hash, content_hash,
    tokenizer_hash, atlas_hash, langid_backend
)
```

The last three matter more than they look. Silently upgrading the language
model, the embedding model or the atlas would produce different findings under
an identical id, which would break the reproducibility claim outright.

## Facets and evidence grades

The fingerprint is a **description, not a scorecard**. Different facets have very
different amounts of evidence behind them, and for several there is no
universally correct direction. Presenting them as one combined quality score
would be dishonest, so each facet carries its grade.

| facet | direction | grade | why |
| --- | --- | --- | --- |
| `contamination` | lower is always better | **strong** | Tülu 3 gives an operational rule; audits find 1–45% leakage across popular benchmarks |
| structural validity (in `quality`) | fewer errors is always better | **deterministic** | a record with an empty loss mask contributes nothing to any gradient; this is a bug, not a judgement |
| `redundancy` | conditional | **conditional** | FineWeb found the benefit comes from removing very large clusters, and that deduplicating harder made their corpus worse |
| `quality` | a trade-off | **conditional** | FineWeb-Edu's filter raised MMLU from 33% to 37% and slightly degraded HellaSwag |
| `coverage` | depends entirely on your goal | **goal-dependent** | a coding-agent dataset *should* be narrow; low diversity is correct for it |
| `language` | depends entirely on your goal | **goal-dependent** | label accuracy matters; the target mix is your decision |
| `shape` | descriptive | **descriptive** | token counts and turn counts have no good or bad direction. Truncation loss is the exception and is deterministic |
| `compliance` | risk, not quality | **risk** | no measurable effect on evaluation scores; it affects legal exposure |

Only the first two justify a hard gate. That is why the blocking set is limited
to structural validity and contamination.

## Structure

```json
{
  "fingerprint_id": "fp_...",
  "schema_version": "fp-v0.1",
  "pipeline_version": "0.1.3",
  "root": "/path/to/data",
  "profile": "sft",
  "facets": {
    "shape": {
      "name": "shape",
      "values": {
        "records": 8123456,
        "datasets": 83,
        "total_chars": 28100000000,
        "token_estimates": {
          "Qwen3":     {"total_tokens": 1020000000, "tokens_per_word": 1.9},
          "Llama-3.1": {"total_tokens": 1340000000, "tokens_per_word": 2.5}
        },
        "cheapest_tokenizer": "Qwen3",
        "packing_algorithm": "concat-and-chunk",
        "residual_tokens": 2841
      },
      "note": "Sizes and token budget. No good or bad direction.",
      "evidence_grade": "descriptive"
    },
    "redundancy":    { "...": "..." },
    "coverage":      { "...": "..." },
    "quality":       { "...": "..." },
    "language":      { "...": "..." },
    "compliance":    { "...": "..." },
    "contamination": { "...": "..." }
  },
  "provenance": {
    "content_hash": "...", "config_hash": "...", "tokenizer_hash": "...",
    "atlas_hash": "", "langid_backend": "fasttext-lid.176",
    "model_id": "Qwen/Qwen3-8B", "seq_len": 4096
  },
  "datasets": [
    {"name": "turkish-qa", "files": 3, "bytes": 91234567, "records": 42000,
     "layout": "chatml", "licence": "apache-2.0", "declared_language": ["tr"]}
  ],
  "capabilities": {
    "tokenizer": true, "chat_template": true, "langid": true,
    "atlas": false, "contamination_index": true
  },
  "degradations": []
}
```

## Packing efficiency names its algorithm

`shape.packing_algorithm` is always present next to the number, because packing
efficiency differs by ten to twenty points between concat-and-chunk, first-fit
decreasing and best-fit. A figure with no algorithm attached is not comparable to
anything, so this release reports concat-and-chunk explicitly, which is what
most SFT packing pipelines actually do.

## Degradations

Anything that fell back rather than failing is listed here and in the report:
schema induction that could not find a layout, a tokenizer whose offsets index a
normalized string, a language backend running in reduced-accuracy mode. Silent
degradation is worse than an error, because the user acts on the output either
way.

## What is not in this release

`coverage` is populated when the bundled atlas and its embedding backend are
available. `difficulty` is not computed at all; it needs Tier 2. See
[limitations.md](limitations.md).
