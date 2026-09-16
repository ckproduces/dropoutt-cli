# atlas-v3 token count

How many tokens the atlas-v3 training corpus (163,452,464 records, 212.8 GB of text on the
build volume) contains under 15 popular tokenizers, estimated from a stratified sample and
cross-checked against the build's own uniform reservoir. The write-up is [REPORT.md](REPORT.md);
raw numbers are in `results/`.

## Stages

| stage | what | script |
| --- | --- | --- |
| 0 | exact population: per-source, per-axis and per-language record counts and byte totals from the build's work dir | `results/population.json` (built inline, see REPORT appendix) |
| 1 | stratified within-source reservoir sample of every source file on the volume | `tokens/sample.py` |
| 2 | token counts per sampled record under each tokenizer | `tokens/tokenize.py` |
| 3 | ratio-estimator extrapolation with standard errors, plus the reservoir-heads cross-check | `tokens/estimate.py` |
| 4 | the report | `tokens/make_report.py` |

## Rerun

```bash
cd experiments/atlas-v3-tokens
../../.venv/bin/python -m tokens.sample 2        # ~50 min: the volume reads at ~40 MB/s
../../.venv/bin/python -m tokens.tokenize        # reservoir heads, then the sampled sources
../../.venv/bin/python -m tokens.estimate
../../.venv/bin/python -m tokens.make_report
```

Sampled texts and per-record counts live under the session scratch directory
(`TOKENS_SCRATCH` overrides it); only the aggregated results are kept here. Tokenizer files
are read from `~/.cache/huggingface/hub`; the ones not already cached were fetched from their
official repos on 12 September 2026 (Xenova/gpt-4, deepseek-ai/DeepSeek-V3,
mistralai/Mistral-Small-3.1-24B-Instruct-2503, google/gemma-3-12b-it, microsoft/phi-4,
allenai/OLMo-2-1124-7B-Instruct).
