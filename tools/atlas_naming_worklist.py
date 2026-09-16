#!/usr/bin/env python3
"""Write per-district naming worklists for an atlas build.

One markdown file per L1 district and one JSON answer skeleton beside it. For
every fine cell the worklist shows 30 members of the build reservoir **spread
from the centre of the cell to its edge** -- never the records nearest the
centroid, which is how the 13 Sep 2026 soup names were made: in a cell with no
shared subject the four nearest records are four unrelated subjects -- together
with the cell's contrastive terms, its source, language and axis mix, its size,
and its coherence under an encoder that is not the atlas's own
(:mod:`minilm_numpy`). Naming rules: ``tools/atlas-data/NAMING_GUIDE.md``.

    tools/atlas_naming_worklist.py --atlas release/atlas-v3.npz \
        --reservoir work/reservoir.jsonl --out naming/

The spread draw (:func:`spread_sample`) ranks a cell's members by similarity to
its centroid, cuts the ranking into 30 bands of equal count and takes one random
member from each band. Every band holds the same share of the cell, so the 30
still stand for the cell in proportion -- two thirds of them are two thirds of
the cell -- but no draw can come out all core or all edge by chance. Bands are
by rank, not by raw distance: a few stray records stretch the distance range,
and equal-width distance bands would give them as many slots as the dense core.

The rows shown are written to ``shown.json`` so the blind readers of B7 can be
given members the namer never saw. Placement goes through
:meth:`Atlas.bind_embedder`, so the reservoir is read with the build's own
encoder input policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from dropoutt.atlas import load_embedder  # noqa: E402
from dropoutt.atlas.apply import Atlas  # noqa: E402

SHOWN = 30
SHOWN_CHARS = 200
SEED = 20260913


def spread_sample(
    members: np.ndarray,
    similarity: np.ndarray,
    count: int,
    rng: np.random.Generator,
    exclude: set[int] | frozenset[int] = frozenset(),
) -> np.ndarray:
    """Pick ``count`` of ``members`` spread from the centre of a cell to its edge.

    ``similarity`` holds each member's cosine to the cell centroid, aligned with
    ``members``. Members are ranked closest first, the ranking is cut into
    ``count`` bands of equal size and one member is drawn at random from each
    band; the result is in rank order. Members in ``exclude`` are passed over
    while enough others remain, and a cell with too few members contributes
    the ones it has, topped up at random from the excluded.
    """
    members = np.asarray(members)
    similarity = np.asarray(similarity, dtype=np.float64)
    position = np.arange(len(members))
    free = np.fromiter((int(m) not in exclude for m in members), bool, len(members))
    ranked = position[free][np.argsort(-similarity[free], kind="stable")]
    if len(ranked) <= count:
        picked = ranked
    else:
        picked = np.array([band[rng.integers(len(band))] for band in np.array_split(ranked, count)])
    short = count - len(picked)
    if short > 0 and not free.all():
        taken = position[~free]
        picked = np.concatenate([picked, rng.choice(taken, min(short, len(taken)), replace=False)])
    picked = picked[np.argsort(-similarity[picked], kind="stable")]
    return members[picked]


def rank_percent(similarity: np.ndarray) -> np.ndarray:
    """Share of a cell's members closer to its centroid than each member, in %."""
    order = np.argsort(-np.asarray(similarity, dtype=np.float64), kind="stable")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order))
    return 100.0 * ranks / max(len(order), 1)


def read_reservoir(path: Path) -> list[tuple[int, str, str, str, str]]:
    with path.open(encoding="utf-8") as handle:
        handle.readline()
        return [tuple(json.loads(line)) for line in handle if line.strip()]


def _mix(items: list, members: np.ndarray, column: int) -> str:
    counts = Counter(items[i][column] for i in members)
    return ", ".join(
        f"{name} {count / max(len(members), 1):.0%}" for name, count in counts.most_common(4)
    )


def place(atlas: Atlas, texts: list[str], langs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest cell of every record and its cosine to that cell's centroid."""
    embedder = load_embedder(atlas.embed_model, offline=True, out_dim=atlas.dim)
    if embedder is None:
        raise SystemExit("encoder not cached; run `dropoutt fetch`")
    embedder = atlas.bind_embedder(embedder)
    profile = atlas.profile
    vectors = embedder.encode(
        texts, weighted=profile.pooling == "sif",
        max_chars=profile.max_chars, max_tokens=profile.max_tokens,
    )
    assigned = atlas.assign_all(vectors, langs)
    return np.asarray(assigned.nearest), np.asarray(assigned.score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--reservoir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    atlas = Atlas.load(args.atlas)
    items = read_reservoir(args.reservoir)
    texts = [item[1] for item in items]
    langs = [item[3] for item in items]
    cells, similarity = place(atlas, texts, langs)
    n_cells = atlas.n_regions
    parent = np.asarray(atlas.region_category)
    print(f"placed {len(items):,} reservoir records on {n_cells} cells")

    rng = np.random.default_rng(SEED)
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(n_cells + 1))
    shown: dict[int, np.ndarray] = {}
    percent: dict[int, float] = {}
    audit_rows: list[int] = []
    audit_cells: list[int] = []
    for cell in range(n_cells):
        members = order[bounds[cell]:bounds[cell + 1]]
        shown[cell] = spread_sample(members, similarity[members], SHOWN, rng)
        ranks = rank_percent(similarity[members]).tolist()
        percent.update(zip(members.tolist(), ranks, strict=True))
        # The coherence hint is read over the members the namer sees.
        audit_rows.extend(shown[cell].tolist())
        audit_cells.extend([cell] * len(shown[cell]))
    (args.out / "shown.json").write_text(
        json.dumps({str(cell): rows.tolist() for cell, rows in shown.items()}) + "\n",
        encoding="utf-8",
    )

    import minilm_numpy

    print(f"independent coherence over {len(audit_rows):,} members")
    vectors = minilm_numpy.encode_parallel([texts[i] for i in audit_rows], workers=args.workers)
    vectors = minilm_numpy.language_centred(vectors, [langs[i] for i in audit_rows])
    coherence = minilm_numpy.cell_coherence(vectors, np.asarray(audit_cells), n_cells)
    np.save(args.out / "coherence.npy", coherence)

    terms = atlas.region_terms
    sizes = atlas.region_size if atlas.region_size is not None else np.zeros(n_cells)
    for l1 in range(int(parent.max()) + 1):
        children = np.flatnonzero(parent == l1).tolist()
        lines = [f"# District {l1} — {len(children)} cells", ""]
        skeleton = {"l1": l1, "l1_name": "", "l1_kind": "", "cells": {}}
        for cell in children:
            members = order[bounds[cell]:bounds[cell + 1]]
            value = coherence[cell]
            hint = (
                "no member sample" if np.isnan(value)
                else "likely mixed" if value < 0.10
                else "check for a shared subject" if value < 0.15
                else "coherent"
            )
            lines += [
                f"## Cell {cell}",
                f"- build records: {int(sizes[cell]):,}; reservoir members: {len(members)}",
                f"- independent coherence: {value:.2f} ({hint}); "
                "reference: unrelated records 0.00, sibling cells 0.17, median cell 0.25",
                f"- sources: {_mix(items, members, 4)}",
                f"- languages: {_mix(items, members, 3)}; axes: {_mix(items, members, 2)}",
                f"- distinctive terms: {', '.join(str(terms[cell]).split(', ')[:16]) if cell < len(terms) else ''}",
                f"- {len(shown[cell])} members from centre to edge "
                "(n% = share of the cell's members closer to the centre):",
            ]
            for i in shown[cell]:
                excerpt = " ".join(texts[i][:SHOWN_CHARS].split())
                lines.append(f"  - [{percent[int(i)]:.0f}%] {excerpt}")
            lines.append("")
            skeleton["cells"][str(cell)] = {"name": "", "kind": ""}
        (args.out / f"l1_{l1:03d}.md").write_text("\n".join(lines), encoding="utf-8")
        (args.out / f"l1_{l1:03d}.json").write_text(
            json.dumps(skeleton, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "atlas": str(args.atlas),
        "corpus_hash": atlas.meta.get("corpus_hash"),
        "cells": n_cells,
        "shown_per_cell": SHOWN,
        "cells_with_fewer_members": int(sum(len(rows) < SHOWN for rows in shown.values())),
        # Over the shown members, as a hint for the namer; the release gate is
        # B7's coherence over its own random draw.
        "coherence_below_0.10": int(np.nansum(coherence < 0.10)),
        "coherence_below_0.15": int(np.nansum(coherence < 0.15)),
        "coherence_median": float(np.nanmedian(coherence)),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
