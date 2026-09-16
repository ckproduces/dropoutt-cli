#!/bin/zsh
# B1..B5, B7 (both v3 reservoirs), then B6 last. Logs beside results.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export BENCH_SCRATCH=/private/tmp/claude-501/-Users-crokan-Documents-dropoutt-cli/b585583f-4892-462f-a850-64929286d9e7/scratchpad/bench-cache
for b in b1_sources b2_topics b3_language b4_coverage b5_labels; do
  echo "=== $b $(date -u +%FT%TZ)"; $PY -m bench.$b > logs/$b.log 2>&1 || echo "FAILED $b"
done
echo "=== b7 atlas-v3 $(date -u +%FT%TZ)"
$PY -m bench.b7_cells atlas-v3 --reservoir /Volumes/ck512/dropoutt-atlas-v2/work-v3-textnorm/reservoir.jsonl > logs/b7_v3.log 2>&1 || echo "FAILED b7 v3"
echo "=== b7 atlas-v3-prev $(date -u +%FT%TZ)"
$PY -m bench.b7_cells atlas-v3-prev --reservoir /Volumes/ck512/dropoutt-atlas-v2/work/reservoir.jsonl > logs/b7_prev.log 2>&1 || echo "FAILED b7 prev"
echo "=== b6 $(date -u +%FT%TZ)"
$PY -m bench.b6_perf_cli > logs/b6_perf_cli.log 2>&1 || echo "FAILED b6"
echo "=== done $(date -u +%FT%TZ)"
