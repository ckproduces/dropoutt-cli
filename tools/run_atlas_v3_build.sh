#!/usr/bin/env bash
# Build atlas-v3 from the frozen corpus cache, detached, with a live log.
# Promotion is deliberate and separate: promote_atlas_v2.py knows the v2 pair
# only, so the finished artifact is copied into the package with its checksum
# once the build has exited cleanly.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
cache_dir="$storage_root/corpus-cache"
# A build with a different encoder input policy needs a fresh work directory;
# the builder refuses to resume one started under another policy.
work_dir=${ATLAS_WORK_DIR:-$storage_root/work}
release_dir=${ATLAS_RELEASE_DIR:-$storage_root/release}
log_dir="$storage_root/logs"
package_dir="$repo_root/src/dropoutt/data/atlas"
python_bin="$repo_root/.venv/bin/python"
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
log_path="$log_dir/build-v3-$run_stamp.log"

[[ -x "$python_bin" ]] || { printf 'missing project Python: %s\n' "$python_bin" >&2; exit 1; }
[[ -f "$cache_dir/manifest.json" ]] || { printf 'missing corpus manifest\n' >&2; exit 1; }
mkdir -p "$work_dir" "$release_dir" "$log_dir"
cd "$repo_root"

printf 'build log: %s\n' "$log_path"
printf 'watch:     %s tools/atlas_build_progress.py --watch 20\n' "$python_bin"

# 166M rows into the default 1<<27-slot dedup table would force a mid-build
# rehash to 1<<28 anyway; start there so the ingest never pauses to do it.
export ATLAS_BUILD_HASH_SLOTS=$((1 << 28))
{
    PYTHONUNBUFFERED=1 /usr/bin/caffeinate -dimsu "$python_bin" tools/build_atlas_v2.py \
        --product atlas-v3 \
        --cache "$cache_dir" --work "$work_dir" --out-dir "$release_dir" "$@"
    if [[ -n "${ATLAS_SKIP_PACKAGE:-}" ]]; then
        printf 'build exited 0; ATLAS_SKIP_PACKAGE set, artifact left in %s\n' "$release_dir"
    else
        printf 'build exited 0; packaging atlas-v3\n'
        cp "$release_dir/atlas-v3.npz" "$package_dir/atlas-v3.npz"
        cp "$release_dir/atlas-v3-release-notes.json" "$package_dir/atlas-v3-release-notes.json"
        (cd "$package_dir" && shasum -a 256 atlas-v3.npz > atlas-v3-SHA256SUMS)
        printf 'packaged: %s\n' "$package_dir/atlas-v3.npz"
    fi
} >>"$log_path" 2>&1
