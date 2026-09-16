#!/usr/bin/env python3
"""Stamp hand-curated atlas-v3 fine-cell (L2) names into the label file and the packaged artifact.

Input is a directory of per-L1 files ``l1_NNN.json`` shaped
``{"l1": <int>, "cells": {"<cell id>": {"name": "<name>", "kind": "<kind>"}, ...}}``
or a single JSON file holding ``{"<cell id>": {"name": ..., "kind": ...}}`` for all
cells. Every name carries a kind -- ``subject``, ``form`` or ``mixed`` -- and is
checked against :mod:`atlas_label_rules`; see ``tools/atlas-data/NAMING_GUIDE.md``.
``--legacy-names`` accepts bare ``"<cell id>": "<name>"`` entries with no kinds,
which is how names written before the kinds existed are re-stamped.

The names are bound to the clustering they describe: ``--corpus-hash`` is the
annotator's claim about which build the names were written against, and it
must match the packaged ``atlas-v3.npz`` or nothing is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_label_rules import check_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tools" / "atlas-data"
OUT = DATA / "region_labels_atlas-v3.json"
PACKAGE = ROOT / "src" / "dropoutt" / "data" / "atlas"
NPZ = PACKAGE / "atlas-v3.npz"
SUMS = PACKAGE / "atlas-v3-SHA256SUMS"

COMMENT = [
    "Curated human-readable labels for the 4096 atlas-v3 fine cells.",
    "Each name was hand-written against members of the cell from the",
    "build's 500,000-record reservoir, spread from its centre to its edge,",
    "its contrastive terms, and its axis, language and source mix.",
    "Every name carries a kind. subject: most members share one subject, named",
    "at the level the members support. form: members share a format or template",
    "but not a subject, and the name says so. mixed: members share neither, and",
    "the name starts with Mixed. No name lists unrelated subjects or names a",
    "first letter. A language is named only when it is the subject of the cell.",
    "Names are chosen with respect to each other: no two cells under the same",
    "L1 parent share a name.",
    "Declarative curation: builds and promotion re-apply these names, and the",
    "builder applies this file only when corpus_hash matches the build.",
]


def _load_npz() -> tuple[dict, dict]:
    data = dict(np.load(NPZ, allow_pickle=True))
    meta = data["meta"].item()
    meta = json.loads(meta) if isinstance(meta, str) else meta
    return data, meta


def _entry(value, legacy: bool) -> tuple[str, str | None]:
    if isinstance(value, dict):
        return str(value.get("name", "")), value.get("kind")
    if not legacy:
        raise SystemExit("names need kinds: {\"name\": ..., \"kind\": ...}; pass --legacy-names to stamp bare names")
    return str(value), None


def load_names(source: Path, n_cells: int, legacy: bool) -> dict[int, tuple[str, str | None]]:
    names: dict[int, tuple[str, str | None]] = {}
    if source.is_dir():
        for path in sorted(source.glob("l1_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in payload["cells"].items():
                cell = int(key)
                if cell in names:
                    raise SystemExit(f"cell {cell} named twice (second time in {path.name})")
                names[cell] = _entry(value, legacy)
    else:
        raw = json.loads(source.read_text(encoding="utf-8"))
        names = {int(k): _entry(v, legacy) for k, v in raw.items()}
    missing = [i for i in range(n_cells) if i not in names]
    if missing:
        raise SystemExit(f"{len(missing)} cells unnamed; first {missing[:20]}")
    extra = sorted(k for k in names if not 0 <= k < n_cells)
    if extra:
        raise SystemExit(f"names for cells outside the map: {extra[:20]}")
    return names


def validate(labels: list[str], kinds: list[str | None], parent: np.ndarray, legacy: bool) -> None:
    problems: list[str] = []
    notes: list[str] = []
    for cell, (label, kind) in enumerate(zip(labels, kinds, strict=True)):
        if legacy and kind is None:
            if not label.strip():
                problems.append(f"cell {cell}: blank name")
            continue
        errors, warnings = check_name(label, kind)
        problems += [f"cell {cell} {label!r}: {e}" for e in errors]
        notes += [f"cell {cell} {label!r}: {w}" for w in warnings]
    for region in range(int(parent.max()) + 1):
        seen: dict[str, int] = {}
        for cell in np.flatnonzero(parent == region).tolist():
            key = labels[cell].strip().casefold()
            if key in seen:
                problems.append(
                    f"L1 {region}: cells {seen[key]} and {cell} share the name {labels[cell]!r}"
                )
            seen[key] = cell
    for note in notes:
        print(f"warning: {note}")
    if problems:
        for problem in problems[:60]:
            print(f"error: {problem}")
        raise SystemExit(f"{len(problems)} naming errors; nothing written")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", type=Path, help="directory of l1_NNN.json files, or one {cell: entry} JSON")
    parser.add_argument("--corpus-hash", required=True)
    parser.add_argument("--legacy-names", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data, meta = _load_npz()
    actual = str(meta.get("corpus_hash", ""))
    if args.corpus_hash != actual:
        raise SystemExit(
            f"refusing: names claim corpus {args.corpus_hash[:12]} but the packaged "
            f"artifact is corpus {actual[:12]}. Hand labels written for one "
            "clustering do not describe another; annotate the current build."
        )
    parent = np.asarray(data["l1_parent"], dtype=np.int64)
    n_cells = int(len(parent))
    names = load_names(args.names, n_cells, args.legacy_names)
    labels = [names[i][0].strip() for i in range(n_cells)]
    kinds = [names[i][1] for i in range(n_cells)]
    validate(labels, kinds, parent, args.legacy_names)
    has_kinds = all(kind is not None for kind in kinds)

    payload = {
        "_comment": COMMENT,
        "atlas_version": "atlas-v3",
        "corpus_hash": args.corpus_hash,
        "labels": {str(i): labels[i] for i in range(n_cells)},
    }
    if has_kinds:
        payload["kinds"] = {str(i): kinds[i] for i in range(n_cells)}
    if args.dry_run:
        counts = {k: kinds.count(k) for k in ("subject", "form", "mixed")} if has_kinds else {}
        print(f"dry run: {n_cells} names validated {counts}")
        return
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta["region_labels"] = labels
    meta["region_labels_source"] = f"curated:{OUT.name}"
    if has_kinds:
        meta["region_kinds"] = kinds
    else:
        meta.pop("region_kinds", None)
    data["meta"] = np.array([json.dumps(meta)])
    tmp = NPZ.parent / "atlas-v3.npz.l2patch.npz"
    np.savez_compressed(tmp, **data)
    shutil.copy2(tmp, NPZ)
    tmp.unlink(missing_ok=True)
    digest = hashlib.sha256(NPZ.read_bytes()).hexdigest()
    SUMS.write_text(f"{digest}  {NPZ.name}\n", encoding="utf-8")
    print(f"wrote {OUT}, patched {NPZ}, checksum {digest[:16]}")


if __name__ == "__main__":
    main()
