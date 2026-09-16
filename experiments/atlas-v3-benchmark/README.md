# atlas-v3 benchmark

Tests of the shipped atlas map (atlas-v3, corpus `99687a41`, built 14 September 2026
from 163 million documents with the encoder input policy and a length-weighted fit)
against the map it replaced (the 12 September build, corpus `4203fc3a`, kept on the
build volume) and against atlas-v2 and atlas-v2-lite.

Two write-ups of the same numbers:

- [REPORT.md](REPORT.md): the current report (16 September 2026). Plain language
  first, with a technical appendix. Explains why each test was chosen, how it was
  run, how it was scored, and what it says about the map.
- [REPORT-2026-09-12.md](REPORT-2026-09-12.md) and
  [REPORT-technical-2026-09-12.md](REPORT-technical-2026-09-12.md) describe the
  12 September run against the previous build; they are kept for the record.

Raw numbers are in `results/*.json`; `python -m bench.summary` prints every table the
report quotes from them. Real command runs are under `runs/` with their terminal
printouts. `results/before-textnorm/` holds the B7 numbers of the previous build.

## What the tests ask

Grouped by the question a user of `dropoutt atlas` actually asks.

| user question | # | test | script |
| --- | --- | --- | --- |
| Where does my corpus sit? | 1 | 52 unfamiliar datasets: does the map keep them apart and sort them by kind? | `bench/b1_sources.py` |
| | 2 | Exam questions (MMLU, MMLU-Pro, Turkish EXAMS): does it sort them by subject? | `bench/b2_topics.py` |
| | 8 | Is a density of 1.0x really "matches the map"? Place the map's own reference sample and check. | `bench/b8_readout.py` |
| Are the names right? | 5 | Names checked with the map's own encoder; exemplar texts return home? | `bench/b5_labels.py` |
| | 7 | Does each cell hold one subject, read by an encoder that is not the atlas's? | `bench/b7_cells.py` |
| | 9 | Names checked with that outside encoder; blind reading of names over held-out data; blind reading of cells. Records that land in cells the map itself calls mixed. | `bench/b9_names.py` |
| What does this add to what I have? | 4 | Find a small slice hidden in a web corpus; breadth ladder; the CLI's own `compare()` on known pairs; how many records a reliable comparison needs; the off-map cutoff. | `bench/b4_coverage.py` |
| Does language get in the way? | 3 | Same topic in four languages; 8,000 sentence pairs and their translations; language mix inside cells. | `bench/b3_language.py` |
| How long does it take? | 6 | 500,000 records timed; three real command runs. | `bench/b6_perf_cli.py` |

## Rules the tests follow

- Test on data the map was never built from. The 52-dataset panel, C4, MMLU, MMLU-Pro,
  EXAMS and WMT17 are not inputs to any v3 build (checked against the build manifest,
  244 sources). Web and Wikipedia panels come from the same public datasets as the
  build but are used only for language and coverage readings.
- Every map gets the same records through the exact steps `dropoutt atlas` uses:
  same text window, same encoder with the map's own word weights and input policy,
  same per-record language detection, same placement rule, same stamped cutoff.
- Every sorting score is shown next to luck (always guess the commonest answer) and
  next to a ceiling (a trained classifier on the same vectors), and as the share of
  that headroom the frozen grid keeps.
- Every headline number carries a 95% interval (bootstrap over records, or Wilson for
  a proportion). Differences between maps are paired on the same records, so the
  interval speaks about the maps, not about the test set.
- Sampling noise gets its own baseline: the "new between halves" figure is shown
  beside what pure multinomial sampling would produce, and the density readout beside
  a multinomial draw from the map's reference shares.
- Names and cells are read blind: the map is hidden and items are shuffled before a
  reader scores them. The reader is stated in the report.
- Statistics that reward having more cells (NMI, purity) stay in the appendix and are
  compared only between maps with the same number of cells.

## Rerun

```bash
cd experiments/atlas-v3-benchmark
./run_core.sh                                   # B1..B5, B7 (both v3 reservoirs), B6; logs/ has the printouts
../../.venv/bin/python -m bench.b8_readout
../../.venv/bin/python -m bench.b9_names        # alignment + writes the two blind sheets
# score results/names_blind.json -> results/names_scores.json and
# results/b7_judge/atlas-v3/mini_sheet.txt -> mini_answers.json, then
../../.venv/bin/python -m bench.b9_names --score
../../.venv/bin/python -m bench.summary         # every table in the report
```

Placements are cached under `BENCH_SCRATCH` (default: this session's scratch
directory), keyed by the artifact's size and mtime, and re-apply the map's current
off-map cutoff when loaded. The previous v3 build and both reservoirs are read from
`/Volumes/ck512/dropoutt-atlas-v2` (see `bench/common.py`). Data comes from
`~/.cache/dropoutt/atlas-corpus` and `~/.cache/huggingface/datasets`; the machine-junk
texts are generated in `bench/data.py`. B7 and B9 need
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in the Hugging Face cache.
