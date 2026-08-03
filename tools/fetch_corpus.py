#!/usr/bin/env python3
"""Fetch the atlas reference corpus to disk, once.

Collection and clustering used to be the same program, which had three costs.
Collection was 87% of a 23-minute build (1,229s of 1,406s) and was paid again on
every re-cluster. It ran one source at a time against forty-odd independent HTTP
streams. And nothing survived the process, so no two builds saw the same corpus
and an encoder swap meant re-downloading everything.

This script is the collection half. It writes one gzipped JSONL shard per source
plus a manifest, and it is resumable: a source that already has a complete shard
is skipped, so a second run costs a directory listing. The builder reads the
cache and never touches the network.

Nothing here aborts the build. A dataset that is gone, gated, renamed or simply
slow is recorded in its shard metadata with the error, and the run continues.
The manifest is where a missing axis becomes visible, and the builder prints it
against ``AXIS_FLOORS`` rather than discovering it in the map afterwards.

Usage:
    python tools/fetch_corpus.py --cache .atlas-cache --workers 10
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_sources import SOURCES, Source  # noqa: E402

#: Bump when the on-disk record format changes. Shards written by an older
#: fetcher are re-fetched rather than silently mixed with new ones.
CACHE_FORMAT = "atlas-cache-v1"

#: Characters below which a row is not worth storing. Matches the client's
#: ATLAS_MIN_CHARS so the reference corpus and a user scan agree on what counts
#: as a record at all.
MIN_CHARS = 80

#: Stored per record. Longer text is truncated here rather than at build time so
#: the cache size is predictable; the client truncates identically.
MAX_CHARS = 2000

_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# row extraction


def field_text(value: object) -> str:
    """Flatten one row field to text.

    Conversational datasets hold a list of ``{"role": ..., "content": ...}``
    dicts where a prose dataset holds a string. Serialising those as JSON would
    put braces and role names into the embedding, which is the exact failure the
    extractor exists to prevent, so they are flattened to their text here.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return ""
    if isinstance(value, dict):
        for key in ("value", "content", "text", "answer", "response"):
            if isinstance(value.get(key), str):
                return value[key]
        return "\n".join(v for v in value.values() if isinstance(v, str))
    if isinstance(value, (list, tuple)):
        parts = [field_text(item) for item in value]
        return "\n".join(p for p in parts if p)
    return ""


def row_text(row: dict, fields: tuple[str, ...]) -> str:
    parts = [field_text(row.get(f)) for f in fields]
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


# ---------------------------------------------------------------------------
# dataset opening


def _parquet_url(hf_id: str, path: str, revision: str = "main") -> str:
    """An ``hf://`` URI, not an https URL.

    Both address the same file, and an https URL with correct percent-escapes
    even returns 200 to a plain HTTP client. But ``datasets`` rewrites whatever
    it is given back into an ``hf://`` URI without unescaping first, so
    ``c%23`` is looked up as a directory literally named ``c%23`` and the load
    fails with a FileNotFoundError naming a path that does exist. Handing it the
    URI it wants, with the characters raw, avoids the round trip: this is what
    lost CommitPackFT's C# and C++ shards on the first run.
    """
    ref = "refs/convert/parquet" if revision == "convert" else revision
    return f"hf://datasets/{hf_id}@{ref}/{path}"


def open_dataset(src: Source, spec: dict | None = None):
    """Open a streaming dataset for ``src`` (or one of its fallbacks)."""
    from datasets import load_dataset

    hf_id = (spec or {}).get("hf_id", src.hf_id)
    config = (spec or {}).get("config", src.config)
    split = (spec or {}).get("split", src.split)
    loader = (spec or {}).get("loader", src.loader if spec is None else "hub")
    path = (spec or {}).get("path", src.path if spec is None else None)
    revision = (spec or {}).get("revision", src.revision if spec is None else "main")

    if loader in ("parquet", "json"):
        if not path:
            raise ValueError(f"{hf_id}: the {loader} loader needs a path")
        # ``path`` may name several shards, comma-separated: one shard of a
        # sharded repo is often far short of a source's row target.
        urls = [
            p if p.startswith(("http", "hf://")) else _parquet_url(hf_id, p, revision)
            for p in path.split(",")
        ]
        return load_dataset(loader, data_files={split: urls}, split=split, streaming=True)
    if loader == "data_dir":
        return load_dataset(hf_id, data_dir=path, split=split, streaming=True)
    return load_dataset(hf_id, config, split=split, streaming=True)


def _iter_rows(ds, *, deadline: float, stall_seconds: float):
    """Yield rows, giving up if a single fetch stalls.

    A hung shard used to take the whole build down with it. The worker thread
    keeps the iterator, and the reader gives up on it rather than joining.
    """
    q: queue.Queue = queue.Queue(maxsize=64)
    sentinel = object()

    def worker() -> None:
        try:
            for row in ds:
                q.put(row)
                if time.time() > deadline:
                    break
            q.put(sentinel)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            q.put(exc)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        if time.time() > deadline:
            raise TimeoutError("source wall-clock budget exhausted")
        try:
            item = q.get(timeout=min(stall_seconds, max(1.0, deadline - time.time())))
        except queue.Empty:
            raise TimeoutError(f"no row within {stall_seconds:.0f}s") from None
        if item is sentinel:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def dataset_card(hf_id: str) -> dict:
    """Best-effort licence and revision, so the cache records provenance."""
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(hf_id, timeout=20)
        tags = list(getattr(info, "tags", []) or [])
        licence = next(
            (t.split(":", 1)[1] for t in tags if t.startswith("license:")), None
        )
        return {
            "revision": getattr(info, "sha", None),
            "license": licence,
            "downloads": getattr(info, "downloads", None),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# one source


def fetch_source(src: Source, cache: Path, *, budget: float, stall: float,
                 scale: float, refresh: bool) -> dict:
    """Fetch one source into ``cache/<slug>/``. Never raises."""
    target = max(50, int(src.target * scale))
    out_dir = cache / src.slug
    meta_path = out_dir / "meta.json"
    shard = out_dir / "records.jsonl.gz"

    if not refresh and meta_path.exists() and shard.exists():
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
        fresh_enough = (
            previous.get("cache_format") == CACHE_FORMAT
            and previous.get("complete")
            and previous.get("rows", 0) >= min(target, previous.get("target", 0))
        )
        if fresh_enough:
            log(f"  cached {src.slug[:52]:<52} {previous.get('rows', 0):>8,} rows")
            return previous

    out_dir.mkdir(parents=True, exist_ok=True)
    part = shard.with_suffix(".gz.part")
    started = time.time()
    deadline = started + budget
    rows = 0
    chars = 0
    error: str | None = None
    note: str | None = None
    used: dict = {"hf_id": src.hf_id, "config": src.config, "split": src.split}

    attempts: list[dict | None] = [None, *[dict(fb) for fb in src.fallbacks]]
    ds = None
    for spec in attempts:
        try:
            ds = open_dataset(src, spec)
            if spec:
                used = {
                    "hf_id": spec.get("hf_id", src.hf_id),
                    "config": spec.get("config", src.config),
                    "split": spec.get("split", src.split),
                    "via": "fallback",
                }
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            ds = None

    if ds is not None:
        error = None
        try:
            with gzip.open(part, "wt", encoding="utf-8", compresslevel=6) as fh:
                for row in _iter_rows(ds, deadline=deadline, stall_seconds=stall):
                    text = row_text(row, src.fields)
                    if len(text) < MIN_CHARS:
                        continue
                    text = text[:MAX_CHARS]
                    rid = hashlib.blake2b(
                        f"{src.slug}:{rows}:{text[:200]}".encode(), digest_size=8
                    ).hexdigest()
                    fh.write(json.dumps(
                        {"id": rid, "text": text},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ) + "\n")
                    rows += 1
                    chars += len(text)
                    if rows >= target:
                        break
        except TimeoutError as exc:
            note = f"stopped early: {exc}"
        except Exception as exc:  # noqa: BLE001
            if rows:
                note = f"partial: {type(exc).__name__}: {str(exc)[:120]}"
            else:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"

    if rows:
        part.replace(shard)
        digest = hashlib.blake2b(digest_size=16)
        with shard.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        shard_hash = digest.hexdigest()
    else:
        part.unlink(missing_ok=True)
        shard.unlink(missing_ok=True)
        shard_hash = ""

    meta = {
        "cache_format": CACHE_FORMAT,
        "slug": src.slug,
        "requested": {k: v for k, v in asdict(src).items() if k != "fallbacks"},
        "used": used,
        "target": target,
        "rows": rows,
        "chars": chars,
        "mean_chars": round(chars / rows, 1) if rows else 0,
        "complete": bool(rows >= target),
        "shortfall": max(0, target - rows),
        "seconds": round(time.time() - started, 1),
        "shard_hash": shard_hash,
        "note": note,
        "error": error,
        "card": dataset_card(used["hf_id"]) if rows else {},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    status = "ok  " if rows >= target else ("part" if rows else "MISS")
    detail = note or error or ""
    log(f"  {status} {src.slug[:52]:<52} {rows:>8,}/{target:<8,} "
        f"{meta['seconds']:>6.1f}s {detail[:60]}")
    return meta


# ---------------------------------------------------------------------------


def salvage_part(cache: Path, src: Source) -> dict | None:
    """Recover an interrupted shard into a usable one.

    A killed fetch leaves ``records.jsonl.gz.part``: a valid gzip stream with
    its tail cut off. Everything before the cut decodes fine, so the rows are
    there — they are just in a file nothing reads. Decoding what survives and
    closing the file properly turns an abandoned download back into a source
    rather than re-fetching it.
    """
    d = cache / src.slug
    part = d / "records.jsonl.gz.part"
    shard = d / "records.jsonl.gz"
    if not part.exists():
        return None
    if part.stat().st_size == 0:
        part.unlink()
        return None

    chunks: list[bytes] = []
    try:
        with gzip.open(part, "rb") as fh:
            while True:
                block = fh.read(1 << 20)
                if not block:
                    break
                chunks.append(block)
    except Exception:
        # Truncated stream: keep whatever decoded before the cut.
        pass
    lines = b"".join(chunks).split(b"\n")
    rows: list[str] = []
    chars = 0
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue  # the final line is usually a partial write
        if isinstance(record, dict) and record.get("text"):
            rows.append(line.decode("utf-8", "replace"))
            chars += len(record["text"])

    existing = 0
    if shard.exists():
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                existing = sum(1 for _ in fh)
        except Exception:
            existing = 0
    if len(rows) <= existing:
        # The completed shard from an earlier run is better than this remnant.
        part.unlink()
        return None

    tmp = d / "records.jsonl.gz.rebuild"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        fh.write("\n".join(rows) + "\n")
    tmp.replace(shard)
    part.unlink()

    digest = hashlib.blake2b(digest_size=16)
    with shard.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    meta = {
        "cache_format": CACHE_FORMAT, "slug": src.slug,
        "requested": {k: v for k, v in asdict(src).items() if k != "fallbacks"},
        "used": {"hf_id": src.hf_id, "config": src.config, "split": src.split},
        "target": src.target, "rows": len(rows), "chars": chars,
        "mean_chars": round(chars / len(rows), 1) if rows else 0,
        "complete": len(rows) >= src.target,
        "shortfall": max(0, src.target - len(rows)),
        "seconds": 0.0, "shard_hash": digest.hexdigest(),
        "note": "recovered from an interrupted fetch", "error": None, "card": {},
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def reindex(cache: Path) -> Path:
    """Rebuild the manifest from what is on disk, without fetching.

    The manifest is written once, at the end of a run, so a fetch that is
    stopped part-way leaves the previous run's index in place. The builder
    reads that index and skips any source it lists with zero rows — so a
    source recovered since would be silently ignored even though its shard is
    sitting right there. This makes the index agree with the disk.
    """
    metas: list[dict] = []
    salvaged = 0
    for src in SOURCES:
        d = cache / src.slug
        if not d.is_dir():
            continue
        recovered = salvage_part(cache, src)
        if recovered is not None:
            salvaged += 1
            log(f"  recovered {src.slug[:48]:<48} {recovered['rows']:>8,} rows")
            metas.append(recovered)
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    if salvaged:
        log(f"  salvaged {salvaged} interrupted shards")
    return write_manifest(cache, metas, {"reindexed": True, "fetched": False})


def write_manifest(cache: Path, metas: list[dict], settings: dict) -> Path:
    """Rewrite the manifest atomically; it is the builder's only index."""
    from atlas_sources import AXIS_FLOORS, axis_totals

    rows_by_slug = {m["slug"]: m.get("rows", 0) for m in metas}
    axes = axis_totals(rows_by_slug)
    manifest = {
        "cache_format": CACHE_FORMAT,
        "settings": settings,
        "totals": {
            "sources": len(metas),
            "sources_with_rows": sum(1 for m in metas if m.get("rows")),
            "rows": sum(m.get("rows", 0) for m in metas),
            "chars": sum(m.get("chars", 0) for m in metas),
        },
        "axes": {
            axis: {
                "rows": count,
                "floor": AXIS_FLOORS.get(axis, 0),
                "meets_floor": count >= AXIS_FLOORS.get(axis, 0),
            }
            for axis, count in sorted(axes.items())
        },
        "sources": sorted(metas, key=lambda m: m["slug"]),
    }
    path = cache / "manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path(".atlas-cache"))
    ap.add_argument("--workers", type=int, default=10,
                    help="Concurrent sources (default 10). These are independent "
                         "HTTP streams; the wall clock is the slowest source, not the sum.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply every per-source row target.")
    ap.add_argument("--budget", type=float, default=900.0,
                    help="Wall-clock seconds per source (default 900).")
    ap.add_argument("--stall", type=float, default=600.0,
                    help="Give up on a source after this long with no row "
                         "(default 600). This is not idle time — ten workers "
                         "saturating the link means a source waiting on its "
                         "first row group can legitimately go minutes without "
                         "yielding a row. At 120s it cost a full run 410,372 "
                         "rows across nine sources, including the largest "
                         "scientific and instruction sets, all of which "
                         "downloaded fine on a retry.")
    ap.add_argument("--refresh", action="store_true",
                    help="Re-fetch sources that already have a complete shard.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Fetch only sources whose slug contains one of these.")
    ap.add_argument("--reindex", action="store_true",
                    help="Rebuild manifest.json from the shards already on disk "
                         "and recover any interrupted ones. No network. Run this "
                         "after stopping a fetch early, or the builder will "
                         "ignore everything the interrupted run had added.")
    args = ap.parse_args()

    if args.reindex:
        log(f"Reindexing {args.cache} …")
        path = reindex(args.cache)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        log(f"\n{manifest['totals']['rows']:,} rows from "
            f"{manifest['totals']['sources_with_rows']} sources → {path}")
        for axis, info in manifest["axes"].items():
            log(f"  {'ok ' if info['meets_floor'] else 'LOW'} {axis:<14} "
                f"{info['rows']:>9,} / floor {info['floor']:,}")
        return 0

    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    selected = SOURCES
    if args.only:
        needles = [n.lower() for n in args.only]
        selected = [s for s in SOURCES
                    if any(n in s.slug.lower() or n in s.axis for n in needles)]

    args.cache.mkdir(parents=True, exist_ok=True)
    settings = {
        "workers": args.workers, "scale": args.scale, "budget_s": args.budget,
        "stall_s": args.stall, "min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
        "sources_selected": len(selected),
    }

    log(f"Fetching {len(selected)} sources into {args.cache} "
        f"with {args.workers} workers")
    t0 = time.time()
    metas: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_source, src, args.cache, budget=args.budget,
                        stall=args.stall, scale=args.scale, refresh=args.refresh): src
            for src in selected
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                metas.append(future.result())
            except Exception as exc:  # noqa: BLE001 - a worker must not kill the run
                metas.append({
                    "cache_format": CACHE_FORMAT, "slug": src.slug, "rows": 0,
                    "chars": 0, "complete": False, "target": src.target,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                })

    # Sources not selected this run still belong in the manifest if cached.
    if args.only:
        for src in SOURCES:
            if any(m["slug"] == src.slug for m in metas):
                continue
            cached = args.cache / src.slug / "meta.json"
            if cached.exists():
                try:
                    metas.append(json.loads(cached.read_text(encoding="utf-8")))
                except Exception:
                    pass

    path = write_manifest(args.cache, metas, settings)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    total = manifest["totals"]["rows"]
    log(f"\nCached {total:,} rows from {manifest['totals']['sources_with_rows']} "
        f"sources in {time.time() - t0:.0f}s → {path}")
    log("\nAxis coverage (floors are advisory; nothing here failed the run):")
    for axis, info in manifest["axes"].items():
        mark = "ok " if info["meets_floor"] else "LOW"
        log(f"  {mark} {axis:<14} {info['rows']:>9,} / floor {info['floor']:,}")
    missing = [m for m in manifest["sources"] if not m.get("rows")]
    if missing:
        log(f"\n{len(missing)} sources returned nothing:")
        for m in missing[:20]:
            log(f"  - {m['slug'][:56]:<56} {str(m.get('error'))[:70]}")

    # Exit rather than return. Every result is already on disk and the manifest
    # is written, but fsspec's HTTP layer leaves a non-daemon worker behind and
    # the interpreter then waits on it at shutdown — the run looks hung for
    # minutes after it has actually finished, which is worse than abrupt.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
