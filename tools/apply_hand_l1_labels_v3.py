#!/usr/bin/env python3
"""Stamp hand-curated atlas-v3 L1 region names into the label file and the packaged artifact.

Input is a JSON list of 256 ``{"name": ..., "kind": ...}`` entries or an
``{index: entry}`` object; kinds and wording rules are those of
:mod:`atlas_label_rules`. ``--legacy-names`` accepts bare name strings.

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
OUT = DATA / "l1_labels_atlas-v3.json"
PACKAGE = ROOT / "src" / "dropoutt" / "data" / "atlas"
NPZ = PACKAGE / "atlas-v3.npz"
SUMS = PACKAGE / "atlas-v3-SHA256SUMS"

COMMENT = [
    "Curated human-readable labels for the 256 atlas-v3 L1 scaffold regions.",
    "Each name was written against the names, kinds and centre-to-edge members",
    "of the region's fine cells, not against the records nearest its centroid.",
    "Every name carries a kind (subject, form or mixed) with the same wording",
    "rules as the fine cells: no lists of unrelated subjects, no first letters,",
    "and a mixed region's name starts with Mixed.",
    "Names are chosen with respect to each other: no two regions share a name.",
    "A language is named only when the language itself is the subject.",
    "Declarative curation: builds and promotion re-apply these names, and the",
    "builder applies this file only when corpus_hash matches the build.",
]


def _meta(npz: Path) -> dict:
    meta = np.load(npz, allow_pickle=True)["meta"].item()
    return json.loads(meta) if isinstance(meta, str) else meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", type=Path, help="JSON list of 256 entries, or {index: entry}")
    parser.add_argument("--corpus-hash", required=True)
    parser.add_argument("--legacy-names", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    actual = str(_meta(NPZ).get("corpus_hash", ""))
    if args.corpus_hash != actual:
        raise SystemExit(
            f"refusing: names claim corpus {args.corpus_hash[:12]} but the packaged "
            f"artifact is corpus {actual[:12]}."
        )
    raw = json.loads(args.names.read_text(encoding="utf-8"))
    entries = [raw[str(i)] for i in range(256)] if isinstance(raw, dict) else list(raw)
    if len(entries) != 256:
        raise SystemExit(f"expected 256 names, got {len(entries)}")
    labels: list[str] = []
    kinds: list[str | None] = []
    for entry in entries:
        if isinstance(entry, dict):
            labels.append(str(entry.get("name", "")).strip())
            kinds.append(entry.get("kind"))
        elif args.legacy_names:
            labels.append(str(entry).strip())
            kinds.append(None)
        else:
            raise SystemExit("names need kinds; pass --legacy-names to stamp bare names")

    problems: list[str] = []
    for i, (label, kind) in enumerate(zip(labels, kinds, strict=True)):
        if kind is None:
            if not label:
                problems.append(f"L1 {i}: blank name")
            continue
        errors, warnings = check_name(label, kind)
        problems += [f"L1 {i} {label!r}: {e}" for e in errors]
        for w in warnings:
            print(f"warning: L1 {i} {label!r}: {w}")
    dupes = {x for x in labels if labels.count(x) > 1}
    if dupes:
        problems.append(f"duplicate names: {sorted(dupes)}")
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        raise SystemExit(f"{len(problems)} naming errors; nothing written")
    has_kinds = all(kind is not None for kind in kinds)

    payload = {
        "_comment": COMMENT,
        "atlas_version": "atlas-v3",
        "corpus_hash": args.corpus_hash,
        "labels": {str(i): labels[i] for i in range(256)},
    }
    if has_kinds:
        payload["kinds"] = {str(i): kinds[i] for i in range(256)}
    if args.dry_run:
        print("dry run: names validated")
        return
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    data = dict(np.load(NPZ, allow_pickle=True))
    meta = _meta(NPZ)
    meta["l1_labels"] = labels
    meta["l1_labels_source"] = f"curated:{OUT.name}"
    if has_kinds:
        meta["l1_kinds"] = kinds
    else:
        meta.pop("l1_kinds", None)
    data["meta"] = np.array([json.dumps(meta)])
    tmp = NPZ.parent / "atlas-v3.npz.l1patch.npz"
    np.savez_compressed(tmp, **data)
    shutil.copy2(tmp, NPZ)
    tmp.unlink(missing_ok=True)
    digest = hashlib.sha256(NPZ.read_bytes()).hexdigest()
    SUMS.write_text(f"{digest}  {NPZ.name}\n", encoding="utf-8")
    print(f"wrote {OUT}, patched {NPZ}, checksum {digest[:16]}")


if __name__ == "__main__":
    main()
