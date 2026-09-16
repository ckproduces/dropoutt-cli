#!/usr/bin/env python3
"""Reconstruct Atlas dedup state from the last source-boundary checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fetch_corpus  # noqa: E402
from atlas_sources import DEFAULT_CACHE, DEFAULT_WORK  # noqa: E402
from build_atlas_v2 import (  # noqa: E402
    ATLAS_V2,
    BLOCK,
    HASH_SLOTS,
    BuildProgress,
    Uint64Set,
    _prepare_batch,
    manifest_sources,
)


def occupied_slots(path: Path, slots: int) -> int:
    table = np.memmap(path, dtype=np.uint64, mode="r", shape=(slots,))
    count = 0
    for start in range(0, slots, 8_000_000):
        count += int(np.count_nonzero(table[start:start + 8_000_000]))
    del table
    return count


def trim_to_checkpoint(work: Path, checkpoint: dict) -> None:
    capacity = int(checkpoint["cap"])
    targets = {
        "emb.f16": capacity * ATLAS_V2.dim * np.dtype(np.float16).itemsize,
        "axis.u8": capacity * np.dtype(np.uint8).itemsize,
        "lang.u8": capacity * np.dtype(np.uint8).itemsize,
        "src.u16": capacity * np.dtype(np.uint16).itemsize,
    }
    # Builds that record per-record lengths carry a fifth column; older work
    # directories have none, and there is nothing to trim.
    if (work / "len.u16").is_file():
        targets["len.u16"] = capacity * np.dtype(np.uint16).itemsize
    for name, size in targets.items():
        path = work / name
        if not path.is_file() or path.stat().st_size < size:
            raise RuntimeError(f"{path} is smaller than checkpoint capacity")
    for name, size in targets.items():
        path = work / name
        with path.open("r+b") as handle:
            handle.truncate(size)


def rebuild(cache: Path, work: Path) -> Path:
    checkpoint_path = work / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    consumed = set(map(str, checkpoint.get("consumed", [])))
    if not consumed:
        raise RuntimeError("checkpoint has no completed sources")
    sources, manifest = manifest_sources(cache)
    source_by_slug = {source.slug: source for source in sources}
    unknown = consumed.difference(source_by_slug)
    if unknown:
        raise RuntimeError(f"checkpoint contains unknown sources: {sorted(unknown)}")
    rows_by_slug = {
        str(item["slug"]): int(item.get("rows", 0))
        for item in manifest["sources"]
    }
    ordered = [source for source in sources if source.slug in consumed]
    total_rows = sum(rows_by_slug[source.slug] for source in ordered)
    rebuilt_path = work / "seen-rebuilt.u64"
    initializing_path = rebuilt_path.with_suffix(".u64.new")
    if initializing_path.exists():
        raise FileExistsError(f"incomplete table initialization must be archived: {initializing_path}")

    progress = BuildProgress(work / "build-progress.json", int(manifest["totals"]["rows"]))
    if rebuilt_path.is_file():
        recovered = occupied_slots(rebuilt_path, HASH_SLOTS)
        print(f"resuming existing reconstruction with {recovered:,} hashes", flush=True)
    else:
        recovered = 0
    seen = Uint64Set(rebuilt_path, slots=HASH_SLOTS)
    scanned = 0
    started = time.time()
    for source_index, source in enumerate(ordered, 1):
        batch: list[str] = []
        axes: list[str] = []
        for text in fetch_corpus.iter_cached_texts(cache, source.slug, ATLAS_V2.max_chars):
            scanned += 1
            batch.append(text)
            axes.append(source.axis)
            if len(batch) < BLOCK:
                continue
            kept, _ = _prepare_batch(batch, axes, seen)
            recovered += len(kept)
            batch, axes = [], []
            if scanned % 250_000 < BLOCK:
                stage_pct = 100.0 * scanned / max(total_rows, 1)
                progress.update(
                    "dedup-recovery",
                    15.0 + 45.0 * total_rows / max(int(manifest["totals"]["rows"]), 1),
                    stage_pct=stage_pct,
                    detail=f"{source.slug} ({source_index}/{len(ordered)} sources)",
                    retained_rows=recovered,
                )
        if batch:
            kept, _ = _prepare_batch(batch, axes, seen)
            recovered += len(kept)
        seen.flush()
        print(
            f"recovered {source_index}/{len(ordered)} {source.slug}: "
            f"{recovered:,} hashes",
            flush=True,
        )

    seen.flush()
    del seen
    gc.collect()
    if scanned != total_rows:
        raise RuntimeError(f"recovery scanned {scanned:,} rows; expected {total_rows:,}")
    occupied = occupied_slots(rebuilt_path, HASH_SLOTS)
    if occupied != recovered:
        raise RuntimeError(f"rebuilt table has {occupied:,} hashes; expected {recovered:,}")
    if occupied < int(checkpoint["n"]):
        raise RuntimeError(
            f"rebuilt hash count {occupied:,} is below checkpoint rows {checkpoint['n']:,}"
        )

    original = work / "seen.u64"
    if not original.is_file():
        raise FileNotFoundError(original)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = work / f"seen-partial-{stamp}.u64"
    original.replace(backup)
    try:
        rebuilt_path.replace(original)
        trim_to_checkpoint(work, checkpoint)
    except Exception:
        if not original.exists() and backup.exists():
            backup.replace(original)
        raise

    safe_manifest_rows = sum(rows_by_slug[slug] for slug in consumed)
    safe_stage_pct = 100.0 * safe_manifest_rows / max(int(manifest["totals"]["rows"]), 1)
    progress.update(
        "resume-ready",
        15.0 + 0.45 * safe_stage_pct,
        stage_pct=100.0,
        detail=f"{len(consumed)} sources; rebuilt {occupied:,} dedup hashes",
        retained_rows=int(checkpoint["n"]),
    )
    summary = {
        "checkpoint_rows": int(checkpoint["n"]),
        "checkpoint_capacity": int(checkpoint["cap"]),
        "completed_sources": len(consumed),
        "manifest_rows_scanned": scanned,
        "dedup_hashes": occupied,
        "partial_seen_backup": str(backup),
        "elapsed_s": round(time.time() - started, 1),
    }
    (work / "recovery-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return backup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    rebuild(args.cache, args.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
