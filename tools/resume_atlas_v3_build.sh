#!/usr/bin/env bash
# Resume an interrupted atlas-v3 build from its last source-boundary checkpoint.
#
# The builder has no signal handler, so a stop lands between the checkpoint
# and however many rows of the next source were appended before the kill.
# Those rows sit past checkpoint.n and are simply overwritten, but their
# dedup hashes may already be in seen.u64 -- which would make the resume drop
# them as "already seen". rebuild_atlas_seen.py reconstructs the table from the
# consumed sources alone, so the next source is ingested from its first row.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
cache_dir="$storage_root/corpus-cache"
work_dir="$storage_root/work"
release_dir="$storage_root/release"
log_dir="$storage_root/logs"
package_dir="$repo_root/src/dropoutt/data/atlas"
python_bin="$repo_root/.venv/bin/python"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$log_dir/build-v3-resume-$run_stamp.log"

# Must match the table the interrupted build used, or the rebuilt set has the
# wrong shape and the builder rehashes it on load. The original run pinned 1<<28.
export ATLAS_BUILD_HASH_SLOTS=$((1 << 28))

[[ -x "$python_bin" ]] || { printf 'missing project Python: %s\n' "$python_bin" >&2; exit 1; }
[[ -f "$cache_dir/manifest.json" ]] || { printf 'corpus volume not mounted: %s\n' "$cache_dir" >&2; exit 1; }
[[ -f "$work_dir/checkpoint.json" ]] || { printf 'no checkpoint to resume from in %s\n' "$work_dir" >&2; exit 1; }
mkdir -p "$log_dir"
cd "$repo_root"

printf 'resume log: %s\n' "$log_path"
printf 'watch:      %s tools/atlas_build_progress.py --watch 20\n' "$python_bin"
{
    printf 'reconstructing dedup state from the last completed-source checkpoint\n'
    PYTHONUNBUFFERED=1 "$python_bin" tools/rebuild_atlas_seen.py --cache "$cache_dir" --work "$work_dir"
    printf 'dedup recovery complete; resuming atlas-v3 build\n'
    PYTHONUNBUFFERED=1 /usr/bin/caffeinate -dimsu "$python_bin" tools/build_atlas_v2.py \
        --product atlas-v3 \
        --cache "$cache_dir" --work "$work_dir" --out-dir "$release_dir" "$@"
    printf 'build exited 0; packaging atlas-v3\n'
    cp "$release_dir/atlas-v3.npz" "$package_dir/atlas-v3.npz"
    cp "$release_dir/atlas-v3-release-notes.json" "$package_dir/atlas-v3-release-notes.json"
    (cd "$package_dir" && shasum -a 256 atlas-v3.npz > atlas-v3-SHA256SUMS)
    printf 'packaged: %s\n' "$package_dir/atlas-v3.npz"
} >>"$log_path" 2>&1
