#!/usr/bin/env bash
# Start the Atlas v3 200 GiB fetch supervisor. Survives child crashes.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
log_dir="$storage_root/logs"
mkdir -p "$log_dir"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
wrapper_log="$log_dir/fetch-v3-supervisor-$stamp.log"

cd "$repo_root"
printf 'supervisor log: %s\n' "$wrapper_log"
printf 'watch: uv run python tools/atlas_fetch_progress.py --watch 20\n'

exec /usr/bin/caffeinate -dimsu uv run python tools/supervise_atlas_v3_fetch.py \
    --cache "$storage_root/corpus-cache" \
    --logs "$log_dir" \
    --storage "$storage_root" \
    "$@" >>"$wrapper_log" 2>&1
