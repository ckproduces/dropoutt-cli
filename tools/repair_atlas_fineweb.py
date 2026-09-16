#!/usr/bin/env python3
"""Repair the FineWeb vector range lost by an unflushed memmap resize."""

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
    DEFAULT_MODEL,
    HASH_SLOTS,
    BuildProgress,
    DiskCorpus,
    Uint64Set,
    _ingest_prepared,
    _prepare_batch,
    _source_language,
    load_embedder,
    manifest_sources,
)


class NullReservoir:
    def offer(self, *_args) -> None:
        pass


def first_source_row(ids: np.memmap, source_id: int, n: int) -> int:
    for start in range(0, n, 2_000_000):
        block = np.asarray(ids[start:min(start + 2_000_000, n)])
        found = np.flatnonzero(block == source_id)
        if len(found):
            return start + int(found[0])
    raise RuntimeError(f"source id {source_id} has no rows")


def rebuild_prior_hashes(
    cache: Path,
    sources,
    stop_index: int,
    seen: Uint64Set,
    progress: BuildProgress,
) -> int:
    recovered = 0
    for source_index, source in enumerate(sources[:stop_index], 1):
        batch: list[str] = []
        axes: list[str] = []
        for text in fetch_corpus.iter_cached_texts(cache, source.slug, ATLAS_V2.max_chars):
            batch.append(text)
            axes.append(source.axis)
            if len(batch) < BLOCK:
                continue
            kept, _ = _prepare_batch(batch, axes, seen)
            recovered += len(kept)
            batch, axes = [], []
        if batch:
            kept, _ = _prepare_batch(batch, axes, seen)
            recovered += len(kept)
        seen.flush()
        progress.update(
            "embedding-repair-dedup",
            0.0,
            stage_pct=100.0 * source_index / stop_index,
            detail=f"reconstructing prior hashes ({source_index}/{stop_index} sources)",
            retained_rows=recovered,
        )
        print(
            f"repair hashes {source_index}/{stop_index} {source.slug}: "
            f"{recovered:,} new",
            flush=True,
        )
    return recovered


def repair(cache: Path, work: Path) -> None:
    checkpoint = json.loads((work / "checkpoint.json").read_text(encoding="utf-8"))
    recovery = json.loads((work / "recovery-summary.json").read_text(encoding="utf-8"))
    sources, manifest = manifest_sources(cache)
    fineweb_index = next(
        i for i, source in enumerate(sources)
        if source.slug == "HuggingFaceFW__fineweb__sample-100BT"
    )
    if fineweb_index == 0 or fineweb_index + 1 >= len(sources):
        raise RuntimeError("FineWeb repair boundaries are unavailable")

    start_row = int(recovery["checkpoint_rows"])
    final_n = int(checkpoint["n"])
    final_cap = int(checkpoint["cap"])
    corpus = DiskCorpus(work, ATLAS_V2.dim)
    corpus.load_checkpoint()
    end_row = first_source_row(corpus.src_ids, fineweb_index + 1, final_n)
    expected = end_row - start_row
    if expected <= 0:
        raise RuntimeError(f"invalid repair range {start_row:,}:{end_row:,}")

    state_path = work / "fineweb-repair-state.json"
    seen_path = work / "seen-fineweb-repair.u64"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"phase": "dedup", "written": 0}
    )
    progress = BuildProgress(work / "build-progress.json", int(manifest["totals"]["rows"]))
    seen = Uint64Set(seen_path, slots=HASH_SLOTS)
    if state["phase"] == "dedup":
        rebuild_prior_hashes(cache, sources, fineweb_index, seen, progress)
        state = {"phase": "encoding", "written": 0}
        state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")

    written = int(state.get("written", 0))
    if not 0 <= written <= expected:
        raise RuntimeError(f"invalid saved repair offset {written:,}")
    corpus.n = start_row + written
    corpus.cap = final_cap
    embedder = load_embedder(DEFAULT_MODEL, out_dim=ATLAS_V2.dim)
    if embedder is None or embedder.dim < ATLAS_V2.dim:
        raise RuntimeError("Atlas encoder is unavailable")
    null = NullReservoir()
    source = sources[fineweb_index]
    language = _source_language(source)
    preserved = 0
    if 0 < end_row - corpus.n <= BLOCK:
        tail_index = np.arange(corpus.n, end_row, dtype=np.int64)
        tail = corpus.rows_f32(tail_index)
        tail_is_valid = (
            np.all(np.linalg.norm(tail, axis=1) > 0)
            and np.all(np.asarray(corpus.lang_ids[tail_index]) == corpus._lang_ix[language])
            and np.all(np.asarray(corpus.src_ids[tail_index]) == corpus._src_ix[source.slug])
        )
        if tail_is_valid:
            preserved = end_row - corpus.n
            written += preserved
            corpus.n = end_row
    batch_text: list[str] = []
    batch_axis: list[str] = []
    scanned = 0
    started = time.time()
    texts = (
        fetch_corpus.iter_cached_texts(cache, source.slug, ATLAS_V2.max_chars)
        if written < expected
        else ()
    )
    for text in texts:
        scanned += 1
        batch_text.append(text)
        batch_axis.append(source.axis)
        if len(batch_text) < BLOCK:
            continue
        kept, _ = _prepare_batch(batch_text, batch_axis, seen)
        if kept:
            remaining = expected - written
            if len(kept) > remaining:
                kept = kept[:remaining]
            added = _ingest_prepared(
                corpus,
                embedder,
                kept,
                [source.axis] * len(kept),
                [language] * len(kept),
                [source.slug] * len(kept),
                null,
                null,
                None,
            )
            written += added
        batch_text, batch_axis = [], []
        if scanned % 250_000 < BLOCK or written >= expected:
            state_path.write_text(
                json.dumps({"phase": "encoding", "written": written}) + "\n",
                encoding="utf-8",
            )
            corpus.flush()
            seen.flush()
            progress.update(
                "embedding-repair",
                100.0 * written / expected,
                stage_pct=100.0 * written / expected,
                detail=f"FineWeb vectors {written:,}/{expected:,}",
                retained_rows=start_row + written,
            )
        if written >= expected:
            break
    if batch_text and written < expected:
        kept, _ = _prepare_batch(batch_text, batch_axis, seen)
        if kept:
            kept = kept[:expected - written]
            written += _ingest_prepared(
                corpus,
                embedder,
                kept,
                [source.axis] * len(kept),
                [language] * len(kept),
                [source.slug] * len(kept),
                null,
                null,
                None,
            )
    state_path.write_text(
        json.dumps({"phase": "encoding", "written": written}) + "\n",
        encoding="utf-8",
    )
    corpus.flush()
    seen.flush()
    if written != expected or corpus.n != end_row:
        raise RuntimeError(
            f"repair wrote {written:,}/{expected:,} vectors; corpus row {corpus.n:,}"
        )
    del seen
    gc.collect()

    # Validate the repaired range before allowing clustering to restart.
    sample_index = np.linspace(start_row, end_row - 1, 20_000, dtype=np.int64)
    sample = corpus.rows_f32(sample_index)
    norms = np.linalg.norm(sample, axis=1)
    if np.count_nonzero(norms == 0):
        raise RuntimeError("repaired FineWeb range still contains zero vectors")
    if not np.all(np.asarray(corpus.lang_ids[sample_index]) == corpus._lang_ix[language]):
        raise RuntimeError("repaired FineWeb language column is inconsistent")
    if not np.all(np.asarray(corpus.src_ids[sample_index]) == corpus._src_ix[source.slug]):
        raise RuntimeError("repaired FineWeb source column is inconsistent")

    summary = {
        "start_row": start_row,
        "end_row": end_row,
        "vectors_repaired": written - preserved,
        "vectors_preserved": preserved,
        "source_rows_scanned": scanned,
        "elapsed_s": round(time.time() - started, 1),
    }
    (work / "fineweb-repair-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    state_path.unlink(missing_ok=True)
    seen_path.unlink(missing_ok=True)
    progress.update(
        "repair-complete",
        100.0,
        stage_pct=100.0,
        detail=f"repaired {written:,} FineWeb vectors",
        retained_rows=final_n,
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    repair(args.cache, args.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
