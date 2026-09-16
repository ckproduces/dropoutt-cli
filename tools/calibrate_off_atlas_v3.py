#!/usr/bin/env python3
"""Calibrate atlas-v3's off-atlas cutoff and stamp it into the packaged artifact.

The rule, which docs/atlas.md promises and ``Atlas.off_threshold`` reads back:

    off_atlas_threshold = 2nd percentile of nearest-cell cosine over the
    build's own language-and-axis-balanced calibration draw.

That draw is ``balanced_calibration_draw`` from the builder -- the same rows,
same seed, that fit the global mean and the stripped PCA directions -- so the
cutoff and the normalization are calibrated on one sample. Every (language,
axis) stratum gets the same quota, so the cutoff is not an English-web cutoff:
a proportional draw is 53% English web, and a percentile of it would reject
ordinary records of every smaller stratum at a higher rate than 2%.

The similarities are computed from the build's ``emb.f16`` memmap, whose rows
are exactly what the runtime encoder produces for the same record (verified:
cosine 1.0 against re-encoding the reservoir text), projected through the
artifact's own per-language centering with each row's detected language. Rows
the memmap never wrote (all-zero blocks left by an interrupted ingest) are
dropped rather than scored; they would otherwise pull the percentile to zero.

Only ``meta`` changes. The tool re-reads the patched file and refuses to
finish if any other array differs from the original.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from build_atlas_v2 import NORM_SAMPLE_ROWS, SEED, balanced_calibration_draw  # noqa: E402

from dropoutt.atlas.apply import Atlas  # noqa: E402

PACKAGE = ROOT / "src" / "dropoutt" / "data" / "atlas"
NPZ = PACKAGE / "atlas-v3.npz"
SUMS = PACKAGE / "atlas-v3-SHA256SUMS"
DEFAULT_WORK = Path("/Volumes/ck512/dropoutt-atlas-v2/work")
PERCENTILE = 2.0
#: Rows per sequential read of the memmap (256 MB at 128 float16 columns).
SCAN_ROWS = 1_000_000
SCORE_ROWS = 65_536


def _meta(npz: Path) -> dict:
    meta = np.load(npz, allow_pickle=True)["meta"].item()
    return json.loads(meta) if isinstance(meta, str) else meta


def gather_rows(emb_path: Path, n: int, dim: int, idx: np.ndarray, log) -> tuple[np.ndarray, int]:
    """One sequential pass over ``emb.f16``; returns the rows at ``idx`` (sorted).

    Scattered reads of this file on the external drive run at a few hundred
    rows per second; a sequential pass over all of it takes under twenty
    minutes. Also counts all-zero rows across the whole file.
    """
    out = np.zeros((len(idx), dim), dtype=np.float16)
    holes = 0
    started = time.time()
    with emb_path.open("rb") as handle:
        start = 0
        while start < n:
            stop = min(start + SCAN_ROWS, n)
            chunk = np.fromfile(handle, dtype=np.float16, count=(stop - start) * dim)
            chunk = chunk.reshape(stop - start, dim)
            lo, hi = np.searchsorted(idx, start), np.searchsorted(idx, stop)
            if hi > lo:
                out[lo:hi] = chunk[idx[lo:hi] - start]
            holes += int((~chunk.view(np.uint16).any(axis=1)).sum())
            start = stop
            rate = start / max(time.time() - started, 1e-9)
            log(f"  {start:,}/{n:,} rows, {(n - start) / rate / 60:.1f} min left")
    return out, holes


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, q / 100.0 * cum[-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="build work dir with emb.f16, lang.u8, axis.u8, checkpoint.json")
    parser.add_argument("--npz", type=Path, default=NPZ)
    parser.add_argument("--rows", type=int, default=NORM_SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument("--cache", type=Path, help="npz of gathered rows; read if present and matching, written otherwise")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    meta = _meta(args.npz)
    atlas = Atlas.load(args.npz)
    checkpoint = json.loads((args.work / "checkpoint.json").read_text())
    n = int(checkpoint["n"])
    if n != int(meta.get("n_reference_records", -1)):
        raise SystemExit(
            f"refusing: work dir holds {n:,} rows but the artifact was built from "
            f"{meta.get('n_reference_records'):,} (corpus {str(meta.get('corpus_hash'))[:12]})."
        )
    # Same recipe as the builder: the manifest digest, the retained row count,
    # the logical byte count and the source table. A work dir from another
    # corpus cannot calibrate this artifact.
    manifest = args.work.parent / "corpus-cache" / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"refusing: {manifest} is missing, so the work dir cannot be tied to the artifact's corpus.")
    manifest_digest = hashlib.blake2b(manifest.read_bytes(), digest_size=16).hexdigest()
    corpus_hash = hashlib.blake2b(
        f"{manifest_digest}:{n}:{checkpoint['logical']}:{','.join(checkpoint['src_table'])}".encode(),
        digest_size=16,
    ).hexdigest()
    if corpus_hash != str(meta.get("corpus_hash", "")):
        raise SystemExit(
            f"refusing: work dir is corpus {corpus_hash[:12]} but the artifact is corpus "
            f"{str(meta.get('corpus_hash'))[:12]}."
        )
    lang_table = list(checkpoint["lang_table"])
    axis_table = list(checkpoint["axis_table"])
    missing = set(atlas.lang_labels) - set(lang_table)
    if missing:
        raise SystemExit(f"refusing: artifact language means {sorted(missing)} are not in this build's language table.")

    lang = np.fromfile(args.work / "lang.u8", dtype=np.uint8, count=n)
    axis = np.fromfile(args.work / "axis.u8", dtype=np.uint8, count=n)
    corpus = SimpleNamespace(lang_ids=lang, axis_ids=axis)
    idx = balanced_calibration_draw(corpus, None, n, args.rows, seed=args.seed)
    strata = len(set((lang[idx].astype(np.int64) * 256 + axis[idx]).tolist()))
    log(f"balanced draw: {len(idx):,} rows over {strata} (language, axis) strata, seed {args.seed}")

    emb = None
    if args.cache is not None and args.cache.is_file():
        cached = np.load(args.cache)
        if int(cached["n"]) == n and int(cached["seed"]) == args.seed and int(cached["rows"]) == args.rows \
                and np.array_equal(cached["idx"], idx):
            emb, holes = cached["emb"], int(cached["holes_total"])
            log(f"gathered rows read from {args.cache}")
        else:
            log(f"cache {args.cache} does not match this draw; gathering afresh")
    if emb is None:
        log("gathering rows from emb.f16 (one sequential pass)")
        emb, holes = gather_rows(args.work / "emb.f16", n, atlas.dim, idx, log)
        if args.cache is not None:
            np.savez(args.cache, idx=idx, emb=emb, lang=lang[idx], axis=axis[idx],
                     holes_total=holes, n=n, seed=args.seed, rows=args.rows)

    written = emb.view(np.uint16).any(axis=1)
    dropped = int((~written).sum())
    log(f"memmap holes: {holes:,} all-zero rows in the file, {dropped:,} of them in the draw (dropped)")
    keep = np.flatnonzero(written)
    rows_lang = [lang_table[int(i)] for i in lang[idx[keep]]]
    rows_axis = axis[idx[keep]]

    scores = np.empty(len(keep), dtype=np.float32)
    for start in range(0, len(keep), SCORE_ROWS):
        block = keep[start:start + SCORE_ROWS]
        _, score, _ = atlas.assign_full(emb[block].astype(np.float32), rows_lang[start:start + SCORE_ROWS])
        scores[start:start + len(block)] = score

    pct = {f"p{q:g}": round(float(np.percentile(scores, q)), 4) for q in (1, 2, 5, 10, 50)}
    threshold = round(float(np.percentile(scores, args.percentile)), 4)
    by_axis = {}
    for a, name in enumerate(axis_table):
        m = rows_axis == a
        if m.sum() >= 1000:
            by_axis[name] = {"rows": int(m.sum()), "p2": round(float(np.percentile(scores[m], 2)), 4),
                             "p50": round(float(np.percentile(scores[m], 50)), 4)}
    # What a proportional (uniform) draw would have said: weight each row by
    # its stratum's population over its quota.
    key_all = lang.astype(np.int64) * 256 + axis
    size = np.bincount(key_all, minlength=256 * 256)
    key_draw = lang[idx[keep]].astype(np.int64) * 256 + rows_axis
    quota = np.bincount(key_draw, minlength=256 * 256)
    weights = (size[key_draw] / quota[key_draw]).astype(np.float64)
    proportional_p2 = round(weighted_percentile(scores, weights, 2.0), 4)

    log(f"reference percentiles: {pct}")
    log(f"proportional-draw p2 estimate: {proportional_p2}")
    for name, row in by_axis.items():
        log(f"  {name:17s} rows {row['rows']:>8,}  p2 {row['p2']:.4f}  p50 {row['p50']:.4f}")
    log(f"off_atlas_threshold = {threshold} (p{args.percentile:g}); current artifact value: "
        f"{meta.get('off_atlas_threshold', 'absent, loader falls back to 0.35')}")
    if args.dry_run:
        return

    meta["off_atlas_threshold"] = threshold
    meta["off_atlas_calibration"] = {
        "rule": "percentile of nearest-cell cosine over the build's language-and-axis-balanced calibration draw",
        "percentile": args.percentile,
        "draw": "language-axis-balanced",
        "draw_seed": args.seed,
        "draw_rows": int(args.rows),
        "strata": strata,
        "rows_scored": int(len(keep)),
        "rows_dropped_zero_embedding": dropped,
        "reference_percentiles": pct,
        "proportional_p2_estimate": proportional_p2,
        "by_axis": by_axis,
        "corpus_hash": corpus_hash,
        "embeddings": "build emb.f16 rows, projected with per-language centering on the detected language",
        "tool": "tools/calibrate_off_atlas_v3.py",
        "calibrated_at": dt.date.today().isoformat(),
    }

    original = dict(np.load(args.npz, allow_pickle=True))
    data = dict(original)
    data["meta"] = np.array([json.dumps(meta)])
    tmp = args.npz.parent / f"{args.npz.name}.cutoffpatch.npz"
    np.savez_compressed(tmp, **data)
    check = np.load(tmp, allow_pickle=True)
    for key, value in original.items():
        if key != "meta" and not np.array_equal(check[key], value):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"refusing: array {key!r} changed while patching meta")
    if _meta(tmp).get("off_atlas_threshold") != threshold:
        tmp.unlink(missing_ok=True)
        raise SystemExit("refusing: patched meta does not carry the cutoff")
    shutil.copy2(tmp, args.npz)
    tmp.unlink(missing_ok=True)
    digest = hashlib.sha256(args.npz.read_bytes()).hexdigest()
    if args.npz == NPZ:
        SUMS.write_text(f"{digest}  {args.npz.name}\n", encoding="utf-8")
    log(f"patched {args.npz}: off_atlas_threshold={threshold}, checksum {digest[:16]}")


if __name__ == "__main__":
    main()
