#!/usr/bin/env bash
# Recover the last Atlas checkpoint, then resume build and package promotion.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
cache_dir="$storage_root/corpus-cache"
work_dir="$storage_root/work"
release_dir="$storage_root/release"
python_bin="$repo_root/.venv/bin/python"
[[ -x "$python_bin" ]] || {
    printf 'missing project Python: %s\n' "$python_bin" >&2
    exit 1
}
cd "$repo_root"

printf 'reconstructing dedup state from the last completed-source checkpoint\n'
PYTHONUNBUFFERED=1 "$python_bin" tools/rebuild_atlas_seen.py \
    --cache "$cache_dir" --work "$work_dir"
printf 'dedup recovery complete; resuming Atlas build\n'
PYTHONUNBUFFERED=1 /usr/bin/caffeinate -dimsu "$python_bin" tools/build_atlas_v2.py \
    --cache "$cache_dir" --work "$work_dir" --out-dir "$release_dir"
printf 'build complete; validating and promoting package artifacts\n'
"$python_bin" tools/promote_atlas_v2.py \
    --release "$release_dir" \
    --package-dir "$repo_root/src/dropoutt/data/atlas"
printf 'Atlas resume, build, and package promotion complete\n'
