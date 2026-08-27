#!/usr/bin/env bash
#SBATCH --job-name=atlas-v2
#SBATCH --account=c00005
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:1
#SBATCH --mem=480G
#SBATCH --time=10-00:00:00

# The only submission entry point for the production Atlas v2 build.  It runs
# from the staged dropoutt checkout after the operator has deleted the retired
# atlas directory and created /ari/users/cokan/dropoutt.
set -Eeuo pipefail
umask 027

INSTALL_ROOT=/ari/users/cokan/dropoutt
PERSISTENT_LOG_DIR="$INSTALL_ROOT/logs"
JOB_ID="${SLURM_JOB_ID:?run under Slurm}"
SCRATCH_ROOT="/YEREL/dropoutt-atlas-v2-$JOB_ID"
REPO="$SCRATCH_ROOT/repo"
CACHE="$SCRATCH_ROOT/corpus-cache"
WORK="$SCRATCH_ROOT/work"
RELEASE="$SCRATCH_ROOT/release"
LEDGER="$RELEASE/source-ledger.json"
LOCK="$RELEASE/atlas-v2-build.lock"
LOG="$PERSISTENT_LOG_DIR/atlas-v2-$JOB_ID.log"
REPO_HASH="${ATLAS_REPOSITORY_HASH:?set ATLAS_REPOSITORY_HASH to the staged source hash}"
SOURCE_ARCHIVE="${ATLAS_SOURCE_ARCHIVE:?set ATLAS_SOURCE_ARCHIVE to the staged source archive}"
TARGET_ROWS=10627780

mkdir -p "$PERSISTENT_LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

copy_diagnostics() {
    local status=$?
    set +e
    mkdir -p "$PERSISTENT_LOG_DIR/atlas-v2-$JOB_ID"
    cp -f "$LOG" "$PERSISTENT_LOG_DIR/atlas-v2-$JOB_ID/" 2>/dev/null || true
    cp -f "$LEDGER" "$PERSISTENT_LOG_DIR/atlas-v2-$JOB_ID/" 2>/dev/null || true
    cp -f "$WORK/ingest-summary.json" "$PERSISTENT_LOG_DIR/atlas-v2-$JOB_ID/" 2>/dev/null || true
    return "$status"
}
cleanup() {
    local status=$?
    copy_diagnostics || true
    # Corpus text, downloaded shards, token caches, embeddings, graph files,
    # and the scratch virtualenv never survive the job.  The source ledger and
    # diagnostics were copied before this removal.
    if [[ -n "${SCRATCH_ROOT:-}" && -d "$SCRATCH_ROOT" ]]; then
        rm -rf -- "$SCRATCH_ROOT"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 128' HUP INT TERM

require_space() {
    local available
    available=$(df -Pk /YEREL | awk 'NR==2 {print $4}')
    [[ "$available" =~ ^[0-9]+$ ]] || { echo "cannot determine /YEREL space"; exit 1; }
    (( available >= 40 * 1024 * 1024 )) || { echo "need 40 GiB free on /YEREL"; exit 1; }
}

require_space
mkdir -p "$REPO" "$RELEASE"
[[ "$SOURCE_ARCHIVE" == "$INSTALL_ROOT/"* && -f "$SOURCE_ARCHIVE" ]]
printf '%s  %s\n' "$REPO_HASH" "$SOURCE_ARCHIVE" | sha256sum -c -
tar -xzf "$SOURCE_ARCHIVE" -C "$REPO"
printf '%s\n' "$REPO_HASH" > "$RELEASE/repository-snapshot-sha256.txt"
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 - <<'PY'
import urllib.request
urllib.request.urlopen("https://huggingface.co", timeout=20).close()
PY

cd "$REPO"
UV_INSTALL_DIR="$SCRATCH_ROOT/uv-bin"
export UV_INSTALL_DIR
curl -LsSf https://astral.sh/uv/0.10.10/install.sh | sh
UV="$UV_INSTALL_DIR/uv"
"$UV" python install 3.12
"$UV" venv --python 3.12 "$SCRATCH_ROOT/.venv"
source "$SCRATCH_ROOT/.venv/bin/activate"
"$UV" pip compile --generate-hashes tools/atlas-v2-build.in -o "$LOCK"
"$UV" pip install --require-hashes -r "$LOCK"

# Stream sources into embeddings, deleting each shard after use.  The ledger
# records attempts and byte counts, but no source text survives the job.
python tools/build_atlas_v2.py \
    --cache "$CACHE" \
    --work "$WORK" \
    --out-dir "$RELEASE" \
    --source-ledger "$LEDGER" \
    --scale 3.0 \
    --target-rows "$TARGET_ROWS"

python -m pytest -q tests/test_atlas_v2_builder.py
sha256sum "$RELEASE/atlas-v2.npz" "$RELEASE/atlas-v2-lite.npz" \
    "$RELEASE/source-ledger.json" "$RELEASE/atlas-v2-release-notes.json" > "$RELEASE/SHA256SUMS"

# Atomically publish only compact release assets.  No corpus or build work is
# copied to persistent storage.  A failed test never reaches this block.
STAGE="$INSTALL_ROOT/.atlas-v2-stage-$JOB_ID"
FINAL="$INSTALL_ROOT/src/dropoutt/data/atlas"
mkdir -p "$STAGE"
cp "$RELEASE/atlas-v2.npz" "$RELEASE/atlas-v2-lite.npz" "$LEDGER" \
   "$RELEASE/atlas-v2-release-notes.json" "$RELEASE/SHA256SUMS" \
   "$RELEASE/repository-snapshot-sha256.txt" "$RELEASE/atlas-v2-build.lock" "$STAGE/"
mkdir -p "$FINAL"
for asset in "$STAGE"/*; do
    mv -f "$asset" "$FINAL/$(basename "$asset")"
done
rmdir "$STAGE"

echo "Atlas v2 artifacts completed: $FINAL"
