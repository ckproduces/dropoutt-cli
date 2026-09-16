#!/usr/bin/env python3
"""Remove verbatim reference-record excerpts from a packaged atlas artifact.

The builder writes ``exemplar_texts`` -- a few hundred characters of the records
nearest each cell's centre -- as a labelling aid: the hand-written L1 and L2
names in ``tools/atlas-data`` were written against those excerpts. Nothing in
``dropoutt.atlas`` reads the array, so in the wheel it was 16,384 verbatim
excerpts of FineWeb, Wikipedia and forum text (21 MB unpacked on atlas-v3) with
no licence manifest, shipped to every install for no reader.

This drops the array and nothing else. Every other array is re-read from the
rewritten file and compared byte for byte with the original, and the SHA256SUMS
file beside the artifact is restamped. Keep the full artifact for the next
labelling pass; the release copy on the build volume is the right place.

    python tools/strip_atlas_exemplars.py src/dropoutt/data/atlas/atlas-v3.npz \
        --backup /Volumes/ck512/dropoutt-atlas-v2/release/with-exemplars/
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

TEXT_ARRAYS = ("exemplar_texts",)


def _sums_path(npz: Path) -> Path | None:
    version = npz.name.split(".", 1)[0]  # atlas-v3 / atlas-v2-lite -> atlas-v2
    family = "-".join(version.split("-")[:2])
    candidate = npz.parent / f"{family}-SHA256SUMS"
    return candidate if candidate.exists() else None


def _restamp(sums: Path, npz: Path) -> None:
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    lines = sums.read_text(encoding="utf-8").splitlines()
    out = []
    seen = False
    for line in lines:
        if line.strip().endswith(f"  {npz.name}"):
            out.append(f"{digest}  {npz.name}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{digest}  {npz.name}")
    sums.write_text("\n".join(out) + "\n", encoding="utf-8")


def strip(npz: Path, backup_dir: Path | None) -> list[str]:
    data = np.load(npz, allow_pickle=True)
    present = [name for name in TEXT_ARRAYS if name in data.files]
    if not present:
        return []
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(npz, backup_dir / npz.name)
    kept = {name: data[name] for name in data.files if name not in TEXT_ARRAYS}
    tmp = npz.parent / f"{npz.name}.stripping.npz"
    np.savez_compressed(tmp, **kept)
    check = np.load(tmp, allow_pickle=True)
    for name, array in kept.items():
        if not np.array_equal(check[name], array):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{npz.name}: array {name!r} changed during rewrite; aborted")
    shutil.copy2(tmp, npz)
    tmp.unlink(missing_ok=True)
    sums = _sums_path(npz)
    if sums is not None:
        _restamp(sums, npz)
    return present


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("npz", type=Path, nargs="+")
    parser.add_argument("--backup", type=Path, default=None,
                        help="directory to copy the unstripped artifact into first")
    args = parser.parse_args(argv)
    for npz in args.npz:
        before = npz.stat().st_size
        removed = strip(npz, args.backup)
        after = npz.stat().st_size
        if removed:
            print(f"{npz.name}: removed {', '.join(removed)}; {before/1e6:.2f} MB -> {after/1e6:.2f} MB")
        else:
            print(f"{npz.name}: nothing to remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
