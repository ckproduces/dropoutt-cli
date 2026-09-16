#!/usr/bin/env bash
# Run Atlas v2 and Atlas v2 lite locally from the same frozen corpus.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
cache_dir="$storage_root/corpus-cache"
work_dir="$storage_root/work"
release_dir="$storage_root/release"
log_dir="$storage_root/logs"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$log_dir/build-$run_stamp.log"

[[ -f "$cache_dir/manifest.json" ]] || {
    printf 'missing corpus manifest: %s\n' "$cache_dir/manifest.json" >&2
    exit 1
}
mkdir -p "$work_dir" "$release_dir" "$log_dir"
cd "$repo_root"

printf 'build log: %s\n' "$log_path"
printf 'progress: python3 %s/tools/atlas_build_progress.py --watch 20\n' "$repo_root"

caffeinate -dimsu uv run python tools/build_atlas_v2.py \
    --cache "$cache_dir" \
    --work "$work_dir" \
    --out-dir "$release_dir" \
    "$@" 2>&1 | tee -a "$log_path"

uv run python tools/promote_atlas_v2.py \
    --release "$release_dir" \
    --package-dir "$repo_root/src/dropoutt/data/atlas" 2>&1 | tee -a "$log_path"
