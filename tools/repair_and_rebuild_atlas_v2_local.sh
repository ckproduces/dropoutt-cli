#!/usr/bin/env bash
# Repair the corrupted FineWeb memmap range, then rebuild and promote both atlases.
set -Eeuo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${DROPOUTT_ATLAS_STORAGE:-/Volumes/ck512/dropoutt-atlas-v2}
python_bin="$repo_root/.venv/bin/python"
[[ -x "$python_bin" ]] || {
    printf 'missing project Python: %s\n' "$python_bin" >&2
    exit 1
}
cd "$repo_root"

PYTHONUNBUFFERED=1 /usr/bin/caffeinate -dimsu "$python_bin" \
    tools/repair_atlas_fineweb.py \
    --cache "$storage_root/corpus-cache" \
    --work "$storage_root/work"

exec "$repo_root/tools/run_atlas_v2_local.sh"
