#!/usr/bin/env python3
"""Sample reference records per L1 cell for atlas-v2 naming."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from atlas_sources import DEFAULT_WORK  # noqa: E402
from build_atlas_v2 import (  # noqa: E402
    RESERVOIR_FULL,
    RESERVOIR_LITE,
    Reservoir,
    stratified_indices,
)

from dropoutt.atlas import load_bundled  # noqa: E402
from dropoutt.atlas.profiles import ATLAS_V2  # noqa: E402

MIN_SAMPLES = 50
SAMPLE_CAP = 80


def _terms(texts: list[str], limit: int = 20) -> list[str]:
    per: Counter[str] = Counter()
    df: Counter[str] = Counter()
    for text in texts:
        words = {w.lower() for w in text.split() if len(w) > 3 and w.isalpha()}
        df.update(words)
        per.update(words)
    total = max(sum(df.values()), 1)
    ranked = sorted(
        ((count * math.log(1 + total / (1 + df[word])), word) for word, count in per.items()),
        reverse=True,
    )
    return [word for _, word in ranked[:limit]]


def _load_checkpoint(work: Path) -> dict:
    return json.loads((work / "checkpoint.json").read_text())


def _assign_l1(atlas_name: str, items: list[tuple], emb: np.memmap, ckpt: dict) -> dict[int, list[dict]]:
    atlas = load_bundled(atlas_name)
    assert atlas.l1_centroids is not None
    l1_centroids = np.asarray(atlas.l1_centroids, dtype=np.float32)

    by_l1: dict[int, list[dict]] = defaultdict(list)
    rows = np.asarray([int(item[0]) for item in items], dtype=np.int64)
    langs = [item[3] for item in items]

    for start in range(0, len(rows), 4096):
        stop = min(start + 4096, len(rows))
        block_rows = rows[start:stop]
        raw = np.asarray(emb[block_rows], dtype=np.float32)
        projected = atlas.project(raw, langs[start:stop])
        parents = projected @ l1_centroids.T
        l1_ids = parents.argmax(axis=1)
        scores = parents[np.arange(len(l1_ids)), l1_ids]
        for offset, (row, l1, score) in enumerate(zip(block_rows, l1_ids, scores, strict=True)):
            item = items[start + offset]
            by_l1[int(l1)].append(
                {
                    "row": int(row),
                    "text": item[1][:600],
                    "axis": item[2],
                    "lang": item[3],
                    "source": item[4],
                    "score": float(score),
                }
            )

    # Keep top-scoring records per L1 (most representative).
    for records in by_l1.values():
        records.sort(key=lambda r: -r["score"])
    return by_l1


def _summarize_l1(l1: int, records: list[dict]) -> dict:
    texts = [r["text"] for r in records[:SAMPLE_CAP]]
    return {
        "l1": l1,
        "reservoir_hits": len(records),
        "axes": Counter(r["axis"] for r in records).most_common(8),
        "langs": Counter(r["lang"] for r in records).most_common(8),
        "sources": Counter(r["source"].split("__")[0] for r in records).most_common(8),
        "terms": _terms(texts),
        "samples": texts[:12],
    }


def inspect_product(atlas_name: str, items: list[tuple], emb: np.memmap, ckpt: dict) -> list[dict]:
    atlas = load_bundled(atlas_name)
    by_l1 = _assign_l1(atlas_name, items, emb, ckpt)
    summaries = []
    short = []
    for l1 in range(atlas.n_l1):
        records = by_l1.get(l1, [])
        summary = _summarize_l1(l1, records)
        summaries.append(summary)
        if len(records) < MIN_SAMPLES:
            short.append((l1, len(records)))
    return summaries, short


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    work = args.work
    ckpt = _load_checkpoint(work)
    cap = int(ckpt["cap"])
    emb = np.memmap(
        work / "emb.f16", dtype=np.float16, mode="r", shape=(cap, ATLAS_V2.dim)
    )

    full_res = Reservoir.load(work / "reservoir.jsonl", RESERVOIR_FULL, 42)
    lite_res = Reservoir.load(work / "lite-reservoir.jsonl", RESERVOIR_LITE, 43)
    lite_axes = [item[2] for item in lite_res.items]
    lite_langs = [item[3] for item in lite_res.items]
    lite_pick = stratified_indices(lite_axes, lite_langs, min(130_000, len(lite_res.items)))
    lite_items = [lite_res.items[i] for i in lite_pick]

    out_dir = work / "l1-inspection"
    out_dir.mkdir(exist_ok=True)

    for atlas_name, items in [("atlas-v2", full_res.items), ("atlas-v2-lite", lite_items)]:
        summaries, short = inspect_product(atlas_name, items, emb, ckpt)
        path = out_dir / f"{atlas_name}-l1-samples.json"
        path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
        print(f"{atlas_name}: wrote {path}")
        print(f"  L1 cells with <{MIN_SAMPLES} reservoir hits: {len(short)}")
        if short[:5]:
            print(f"  examples: {short[:5]}")


if __name__ == "__main__":
    main()
