"""B7. Does each cell hold one subject?

Added after the 13 Sep 2026 naming audit. B5 checks names with the atlas's own
encoder, and a cell the encoder built is coherent to that encoder by
construction: cells that shared only a first letter, ALL-CAPS spelling or a
translated prompt template passed every B5 check. This test reads cells with an
encoder that is not the atlas's, and then with a reader.

  a. Coherence. Twenty random members of every cell -- drawn from the build's
     own reservoir, placed with the map's own encoder input policy -- are
     embedded with paraphrase-multilingual-MiniLM-L12-v2, centred per language,
     and scored by mean pairwise cosine. Unrelated records score 0.00; on the
     first atlas-v3, cells a blind reader called grab-bags sat mostly under 0.15.
  b. Blind reading. ``--make-judge`` writes 160 cells, 20 per coherence octile,
     each with 16 members spread from the centre of the cell to its edge (the
     worklist's draw, :func:`atlas_naming_worklist.spread_sample`) and its
     current name, in four batches for readers who classify the cell (subject /
     form / grab-bag) and grade the name (accurate / vague / soup /
     false-specific / wrong). With ``--naming-dir`` the readers get members the
     namer was not shown, wherever the cell has enough. ``--score-judge`` joins
     their answers.

Gates (a map that fails any of these is not released with its names):

  * at most 1.5% of cells below coherence 0.10;
  * at most 7% of judged cells are grab-bags;
  * no judged name is a soup, and at least 70% are accurate.

    python -m bench.b7_cells atlas-v3 --reservoir /Volumes/.../work/reservoir.jsonl
    python -m bench.b7_cells atlas-v3 --reservoir ... --make-judge --naming-dir naming/
    python -m bench.b7_cells atlas-v3 --score-judge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C

sys.path.insert(0, str(C.REPO / "tools"))
import minilm_numpy
from atlas_naming_worklist import spread_sample

MEMBERS = 20
JUDGE_MEMBERS = 16
JUDGE_PER_OCTILE = 20
GATE_BELOW_010 = 0.015
GATE_GRAB_BAG = 0.07
GATE_ACCURATE = 0.70
SEED = 7


def _digest(path: str) -> str:
    return hashlib.blake2b(path.encode(), digest_size=6).hexdigest()


def _reservoir(path: str):
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r[1] for r in rows], [r[3] for r in rows]


def _members(name: str, reservoir: str):
    texts, langs = _reservoir(reservoir)
    placed = C.place(name, texts, langs, tag=f"b7-{_digest(reservoir)}")
    cells = np.asarray(placed.nearest)
    n = C.atlas(name).n_regions
    rng = np.random.default_rng(SEED)
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(n + 1))
    picked = {c: rng.permutation(order[bounds[c]:bounds[c + 1]]) for c in range(n)}
    return texts, langs, cells, picked, np.asarray(placed.score)


def coherence(name: str, reservoir: str) -> dict:
    texts, langs, cells, picked, _ = _members(name, reservoir)
    n = len(picked)
    rows = [int(i) for c in range(n) for i in picked[c][:MEMBERS]]
    owner = np.array([c for c in range(n) for _ in picked[c][:MEMBERS]])
    cache = C.SCRATCH / f"b7-minilm-{name}-{_digest(reservoir + C.fingerprint(name))}.npz"
    if cache.exists():
        vectors = np.load(cache)["v"]
    else:
        vectors = minilm_numpy.encode_parallel([texts[i] for i in rows])
        np.savez(cache, v=vectors)
    vectors = minilm_numpy.language_centred(vectors, [langs[i] for i in rows])
    coh = minilm_numpy.cell_coherence(vectors, owner, n)
    np.save(C.RESULTS / f"b7_coherence_{name}.npy", coh)
    sizes = np.bincount(cells, minlength=n)
    scored = ~np.isnan(coh)
    return {
        "cells": n,
        "cells_scored": int(scored.sum()),
        "median": float(np.nanmedian(coh)),
        "share_below_0.10": float(np.nanmean(coh < 0.10)),
        "share_below_0.15": float(np.nanmean(coh < 0.15)),
        "reservoir_share_in_cells_below_0.10": float(
            sizes[scored & (coh < 0.10)].sum() / max(sizes[scored].sum(), 1)
        ),
        "gate_below_0.10": float(np.nanmean(coh < 0.10)) <= GATE_BELOW_010,
    }


def make_judge(name: str, reservoir: str, naming_dir: str | None = None) -> None:
    texts, _, _, picked, similarity = _members(name, reservoir)
    shown: dict[str, list[int]] = {}
    if naming_dir:
        shown = json.loads((Path(naming_dir) / "shown.json").read_text(encoding="utf-8"))
    else:
        print("no --naming-dir: readers may see members the namer also saw")
    spread_rng = np.random.default_rng(SEED)
    coh = np.load(C.RESULTS / f"b7_coherence_{name}.npy")
    labels = C.atlas(name).region_labels
    edges = np.nanpercentile(coh, np.linspace(0, 100, 9))
    rng = random.Random(SEED)
    chosen = []
    for k in range(8):
        pool = [c for c in range(len(coh)) if edges[k] <= coh[c] <= edges[k + 1]]
        chosen += rng.sample(pool, min(JUDGE_PER_OCTILE, len(pool)))
    rng.shuffle(chosen)
    out = C.RESULTS / "b7_judge" / name
    out.mkdir(parents=True, exist_ok=True)
    key = {}
    for batch in range(4):
        lines = []
        for j, cell in enumerate(chosen[batch::4]):
            tag = f"C{batch}{j:02d}"
            key[tag] = int(cell)
            lines.append(f"=== {tag}\nCURRENT NAME: {labels[cell] if cell < len(labels) else ''}")
            members = picked[cell]
            sample = spread_sample(
                members, similarity[members], JUDGE_MEMBERS, spread_rng,
                exclude=frozenset(shown.get(str(cell), ())),
            )
            # Shuffled: readers judge the cell, not a centre-to-edge ordering.
            for i in spread_rng.permutation(sample):
                lines.append("  - " + " ".join(texts[int(i)][:220].split()))
            lines.append("")
        (out / f"batch_{batch}.txt").write_text("\n".join(lines), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"wrote 4 batches of {len(chosen)} cells to {out}; "
          "readers write answers_<k>.json beside them")


def score_judge(name: str) -> dict:
    out = C.RESULTS / "b7_judge" / name
    key = json.loads((out / "key.json").read_text())
    rows = []
    for batch in range(4):
        for answer in json.loads((out / f"answers_{batch}.json").read_text()):
            answer["cell"] = key[answer["tag"]]
            rows.append(answer)
    content = Counter(r["content"] for r in rows)
    verdict = Counter(r["name_verdict"] for r in rows)
    n = max(len(rows), 1)
    result = {
        "judged": len(rows),
        "content": dict(content),
        "name_verdict": dict(verdict),
        "grab_bag_share": content["grab-bag"] / n,
        "accurate_share": verdict["accurate"] / n,
        "soup_names": verdict["soup"],
        "median_fit_of_16": float(np.median([r["fit"] for r in rows])) if rows else None,
    }
    result["gates"] = {
        "grab_bag": result["grab_bag_share"] <= GATE_GRAB_BAG,
        "no_soup": result["soup_names"] == 0,
        "accurate": result["accurate_share"] >= GATE_ACCURATE,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("product", nargs="?", default=C.MAIN)
    parser.add_argument("--reservoir")
    parser.add_argument("--make-judge", action="store_true")
    parser.add_argument(
        "--naming-dir", help="worklist directory; its shown.json is kept from readers"
    )
    parser.add_argument("--score-judge", action="store_true")
    args = parser.parse_args()
    path = C.RESULTS / "b7_cells.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    entry = results.setdefault(args.product, {})
    if args.score_judge:
        entry["judge"] = score_judge(args.product)
    else:
        if not args.reservoir:
            raise SystemExit("--reservoir is the build reservoir the map was fitted with")
        entry["coherence"] = coherence(args.product, args.reservoir)
        entry["reservoir"] = args.reservoir
        if args.make_judge:
            make_judge(args.product, args.reservoir, args.naming_dir)
    path.write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
