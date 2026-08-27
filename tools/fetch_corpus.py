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
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_sources import (
    BASELINE_CATALOGUE_COMMIT,
    BASELINE_SCALE,
    LOGICAL_BYTE_TARGET,
    SOURCES,
    SUPPLEMENTAL_LANGUAGES,
    Source,
)

#: Bump when the on-disk record format changes. Shards written by an older
#: fetcher are re-fetched rather than silently mixed with new ones.
CACHE_FORMAT = "atlas-cache-v3"

#: Characters below which a row is not worth storing. Matches the client's
#: ATLAS_MIN_CHARS so the reference corpus and a user scan agree on what counts
#: as a record at all.
MIN_CHARS = 80

#: Default stored per record; override with ``--max-chars`` (4000 for atlas-v2).
MAX_CHARS_DEFAULT = 4000

FINEWEB2_ID = "HuggingFaceFW/fineweb-2"
FINEWEB2_PREFIX = "data/{language}/train/"

_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


# ---------------------------------------------------------------------------


def logical_bytes(metas: list[dict]) -> int:
    """Return retained UTF-8 bytes; never use compressed shard size as corpus size."""
    return sum(int(meta.get("logical_bytes", 0)) for meta in metas)


def _used_fineweb2_paths() -> set[str]:
    paths: set[str] = set()
    for src in SOURCES:
        if src.hf_id == FINEWEB2_ID and src.path:
            paths.update(path.strip() for path in src.path.split(","))
    return paths


def list_fineweb2_paths() -> list[str]:
    """List public FineWeb-2 parquet paths in stable Hub order."""
    from huggingface_hub import list_repo_files

    return sorted(list_repo_files(FINEWEB2_ID, repo_type="dataset"))


def supplemental_sources(paths: list[str]) -> dict[str, list[Source]]:
    """Turn public FineWeb-2 paths into the only allowed supplemental sources."""
    excluded = _used_fineweb2_paths()
    result: dict[str, list[Source]] = {lang: [] for lang, _ in SUPPLEMENTAL_LANGUAGES}
    for lang, script in SUPPLEMENTAL_LANGUAGES:
        prefix = FINEWEB2_PREFIX.format(language=script)
        for path in sorted(path for path in paths if path.startswith(prefix) and path.endswith(".parquet")):
            if path in excluded:
                continue
            result[lang].append(Source(
                FINEWEB2_ID, None, "train", ("text",), "web", 1_000_000_000,
                lang, loader="parquet", path=path,
            ))
    return result


def byte_quotas(total: int, languages: list[str]) -> dict[str, int]:
    """Split bytes exactly, assigning indivisible remainder in language order."""
    if not languages:
        return {}
    quotient, remainder = divmod(total, len(languages))
    return {language: quotient + int(index < remainder)
            for index, language in enumerate(languages)}


def _supplemental_unavailable(language: str, error: str) -> dict:
    return {
        "cache_format": CACHE_FORMAT,
        "slug": f"supplemental__fineweb2__{language}",
        "requested": {"hf_id": FINEWEB2_ID, "language": language},
        "used": {"hf_id": FINEWEB2_ID},
        "source_role": "supplemental",
        "target": 0,
        "rows": 0,
        "chars": 0,
        "logical_bytes": 0,
        "byte_limit": 0,
        "complete": False,
        "shortfall": 0,
        "seconds": 0,
        "shard_hash": "",
        "checksum": {"algorithm": "blake2b-128", "value": ""},
        "status": "unavailable",
        "error": error,
        "note": None,
        "card": {},
    }


def fetch_supplemental(cache: Path, *, current_bytes: int, budget: float,
                       stall: float, refresh: bool, max_chars: int,
                       paths: list[str] | None = None,
                       existing: list[dict] | None = None,
                       target_logical_bytes: int = LOGICAL_BYTE_TARGET) -> list[dict]:
    """Fill the logical-byte deficit with public FineWeb-2 shards.

    The initial quota is equal across the 22 requested languages. Exhausted
    languages are removed after each deterministic wave and their shortfall is
    split equally among the remaining languages. Fetching is serial here: that
    makes the global final-record overshoot bounded to one record.
    """
    if current_bytes >= target_logical_bytes:
        return []
    try:
        discovered = list_fineweb2_paths() if paths is None else paths
    except Exception as exc:
        return [_supplemental_unavailable(lang, f"path listing {type(exc).__name__}: {exc}")
                for lang, _ in SUPPLEMENTAL_LANGUAGES]
    used_slugs = {str(meta.get("slug")) for meta in existing or []
                  if meta.get("source_role") == "supplemental" and meta.get("rows")}
    queues = supplemental_sources(discovered)
    for language, items in queues.items():
        queues[language] = [source for source in items if source.slug not in used_slugs]
    languages = [lang for lang, _ in SUPPLEMENTAL_LANGUAGES]
    metas: list[dict] = []
    positions = {lang: 0 for lang in languages}
    exhausted: set[str] = set()
    total = current_bytes

    while total < target_logical_bytes:
        active = [lang for lang in languages if lang not in exhausted]
        if not active:
            break
        quotas = byte_quotas(target_logical_bytes - total, active)
        progress = False
        for lang in active:
            needed = quotas[lang]
            if needed <= 0 or total >= target_logical_bytes:
                continue
            queue_for_language = queues[lang]
            if positions[lang] >= len(queue_for_language):
                exhausted.add(lang)
                continue
            src = queue_for_language[positions[lang]]
            positions[lang] += 1
            # Only the final global allocation may retain a record over target.
            final_allocation = len(active) == 1 and needed == target_logical_bytes - total
            meta = fetch_source(
                src, cache, budget=budget, stall=stall, scale=1.0,
                refresh=refresh, max_chars=max_chars, byte_limit=needed,
                allow_overshoot=final_allocation, source_role="supplemental",
            )
            metas.append(meta)
            got = int(meta.get("logical_bytes", 0))
            total += got
            progress = progress or bool(got)
            if positions[lang] >= len(queue_for_language) and got < needed:
                exhausted.add(lang)
            if total >= target_logical_bytes:
                break
        if not progress:
            break

    for lang in languages:
        if not queues[lang]:
            metas.append(_supplemental_unavailable(lang, "no public unused FineWeb-2 parquet path"))
    return metas


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
    q: queue.Queue = queue.Queue(maxsize=8)
    sentinel = object()
    stop = threading.Event()

    def worker() -> None:
        try:
            for row in ds:
                if stop.is_set() or time.time() > deadline:
                    break
                q.put(row)
            q.put(sentinel)
        except Exception as exc:
            q.put(exc)

    threading.Thread(target=worker, daemon=True).start()
    try:
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
    finally:
        stop.set()
        with suppress(queue.Empty):
            while True:
                q.get_nowait()


def iter_local(src: Source, *, deadline: float):
    """Yield rows from a repo-local JSON file (probes, curated lists)."""
    root = Path(__file__).resolve().parent.parent
    path = Path(src.path) if Path(src.path).is_absolute() else root / src.path
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("probes", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list or {{probes: [...]}}")
    for row in rows:
        if time.time() > deadline:
            break
        if isinstance(row, dict):
            yield row


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


def preflight_source(src: Source, spec: dict | None = None) -> dict:
    """Reject gated/private Hub repositories before opening their payload.

    The ledger must distinguish an unavailable public source from a source that
    was never attempted.  Hub metadata is deliberately the only preflight
    network operation; row streaming remains in :func:`open_dataset`.
    """
    if src.loader == "local":
        return {"public": True, "card": {}}
    hf_id = (spec or {}).get("hf_id", src.hf_id)
    try:
        from huggingface_hub import dataset_info

        info = dataset_info(hf_id, timeout=20)
        if getattr(info, "gated", False) or getattr(info, "private", False):
            return {
                "public": False,
                "card": dataset_card(hf_id),
                "error": "source is gated or private; public-only build excludes it",
            }
        return {"public": True, "card": dataset_card(hf_id)}
    except Exception as exc:
        return {
            "public": False,
            "card": {},
            "error": f"preflight {type(exc).__name__}: {str(exc)[:160]}",
        }


# ---------------------------------------------------------------------------
# one source


def fetch_source(src: Source, cache: Path, *, budget: float, stall: float,
                 scale: float, refresh: bool, max_chars: int,
                 byte_limit: int | None = None, allow_overshoot: bool = False,
                 source_role: str = "baseline") -> dict:
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
            and previous.get("max_chars") == max_chars
            and previous.get("complete")
            and (
                previous.get("rows", 0) >= target
                if byte_limit is None
                else previous.get("logical_bytes", 0) >= byte_limit
            )
            and previous.get("byte_limit") == byte_limit
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
    logical_bytes = 0
    error: str | None = None
    note: str | None = None
    preflight_card: dict = {}
    used: dict = {"hf_id": src.hf_id, "config": src.config, "split": src.split}

    attempts: list[dict | None] = [None, *[dict(fb) for fb in src.fallbacks]]
    ds = None
    local = src.loader == "local"
    if local:
        try:
            ds = iter_local(src, deadline=deadline)
            used = {"hf_id": src.hf_id, "config": src.config, "split": src.split,
                    "path": src.path, "via": "local"}
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            ds = None
    else:
        for spec in attempts:
            try:
                preflight = preflight_source(src, spec)
                preflight_card = preflight.get("card", {})
                if preflight.get("error"):
                    error = preflight["error"]
                    continue
                ds = open_dataset(src, spec)
                if spec:
                    used = {
                        "hf_id": spec.get("hf_id", src.hf_id),
                        "config": spec.get("config", src.config),
                        "split": spec.get("split", src.split),
                        "via": "fallback",
                    }
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                ds = None

    if ds is not None:
        error = None
        try:
            with gzip.open(part, "wt", encoding="utf-8", compresslevel=6) as fh:
                row_iter = ds if local else _iter_rows(ds, deadline=deadline, stall_seconds=stall)
                for row in row_iter:
                    text = row_text(row, src.fields)
                    if len(text) < MIN_CHARS:
                        continue
                    text = text[:max_chars]
                    text_bytes = len(text.encode("utf-8"))
                    if byte_limit is not None and logical_bytes + text_bytes > byte_limit:
                        if allow_overshoot and logical_bytes < byte_limit:
                            pass
                        else:
                            break
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
                    logical_bytes += text_bytes
                    if byte_limit is not None and logical_bytes >= byte_limit:
                        break
                    if rows >= target:
                        break
        except TimeoutError as exc:
            note = f"stopped early: {exc}"
        except Exception as exc:
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
        if shard.exists() and meta_path.exists() and not refresh:
            # A failed re-fetch must not wipe a good shard from an earlier run.
            try:
                previous = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                previous = None
            if previous and previous.get("rows"):
                log(f"  kept  {src.slug[:52]:<52} {previous.get('rows', 0):>8,} rows "
                    f"(fetch failed: {(error or note or 'unknown')[:40]})")
                return previous
        shard.unlink(missing_ok=True)
        shard_hash = ""

    meta = {
        "cache_format": CACHE_FORMAT,
        "slug": src.slug,
        "requested": {k: v for k, v in asdict(src).items() if k != "fallbacks"},
        "used": used,
        "target": target,
        "source_role": source_role,
        "max_chars": max_chars,
        "rows": rows,
        "chars": chars,
        "logical_bytes": logical_bytes,
        "byte_limit": byte_limit,
        "mean_chars": round(chars / rows, 1) if rows else 0,
        "complete": bool(rows and (rows >= target if byte_limit is None else logical_bytes >= byte_limit)),
        "shortfall": max(0, target - rows) if byte_limit is None else max(0, byte_limit - logical_bytes),
        "seconds": round(time.time() - started, 1),
        "shard_hash": shard_hash,
        "checksum": {"algorithm": "blake2b-128", "value": shard_hash},
        "status": "complete" if rows and (byte_limit is None or logical_bytes >= byte_limit)
        else ("partial" if rows else "unavailable"),
        "note": note,
        "error": error,
        "card": preflight_card if not local else {},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")

    status = "ok  " if meta["complete"] else ("part" if rows else "MISS")
    detail = note or error or ""
    log(f"  {status} {src.slug[:52]:<52} {rows:>8,}/{target:<8,} "
        f"{meta['seconds']:>6.1f}s {detail[:60]}")
    return meta


# ---------------------------------------------------------------------------
# stream-and-delete helpers (local 40 GiB build)


def wipe_tree(path: Path) -> None:
    """Best-effort recursive delete. Missing paths are not an error."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@contextmanager
def isolate_hf_home(root: Path):
    """Point HuggingFace caches at a disposable directory for one build.

    Streaming parquet still materialises the current shard in the hub cache.
    Isolating that cache lets the builder delete it after each source without
    touching the user's existing ``~/.cache/huggingface``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    keys = ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
    try:
        yield root
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def delete_cached_source(cache: Path, slug: str) -> None:
    wipe_tree(cache / slug)


def cached_shard_path(cache: Path, slug: str) -> Path:
    return cache / slug / "records.jsonl.gz"


def iter_cached_texts(cache: Path, slug: str, max_chars: int):
    """Yield retained texts from an on-disk shard. Does not delete it."""
    shard = cached_shard_path(cache, slug)
    if not shard.is_file():
        return
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("text", "")).strip()
            if len(text) < MIN_CHARS:
                continue
            yield text[:max_chars]


def iter_streamed_texts(
    src: Source,
    *,
    budget: float,
    stall: float,
    max_chars: int,
    row_limit: int,
    byte_limit: int | None = None,
):
    """Yield texts from a live Hub stream. Never writes a shard."""
    deadline = time.time() + budget
    attempts: list[dict | None] = [None, *[dict(fb) for fb in src.fallbacks]]
    ds = None
    if src.loader == "local":
        try:
            ds = iter_local(src, deadline=deadline)
        except Exception:
            return
    else:
        for spec in attempts:
            try:
                preflight = preflight_source(src, spec)
                if preflight.get("error"):
                    continue
                ds = open_dataset(src, spec)
                break
            except Exception:
                ds = None
    if ds is None:
        return
    rows = 0
    logical = 0
    try:
        row_iter = ds if src.loader == "local" else _iter_rows(
            ds, deadline=deadline, stall_seconds=stall
        )
        for row in row_iter:
            text = row_text(row, src.fields)
            if len(text) < MIN_CHARS:
                continue
            text = text[:max_chars]
            text_bytes = len(text.encode("utf-8"))
            if byte_limit is not None and logical + text_bytes > byte_limit:
                break
            yield text
            rows += 1
            logical += text_bytes
            if byte_limit is not None and logical >= byte_limit:
                break
            if rows >= row_limit:
                break
    except (TimeoutError, Exception):
        return


def iter_supplemental_sources(paths: list[str] | None = None):
    """Yield FineWeb-2 sources not already named in the baseline catalogue."""
    discovered = list_fineweb2_paths() if paths is None else paths
    queues = supplemental_sources(discovered)
    for lang, _ in SUPPLEMENTAL_LANGUAGES:
        for src in queues.get(lang, []):
            yield src


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
    recovered_bytes = 0
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
            recovered_bytes += len(str(record["text"]).encode("utf-8"))

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
        "target": src.target, "source_role": "baseline", "rows": len(rows), "chars": chars,
        "logical_bytes": recovered_bytes, "byte_limit": None,
        "mean_chars": round(chars / len(rows), 1) if rows else 0,
        "complete": len(rows) >= src.target,
        "shortfall": max(0, src.target - len(rows)),
        "seconds": 0.0, "shard_hash": digest.hexdigest(),
        "checksum": {"algorithm": "blake2b-128", "value": digest.hexdigest()},
        "status": "partial", "note": "recovered from an interrupted fetch", "error": None, "card": {},
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def reindex(cache: Path, *, source_ledger: Path | None = None,
            target_logical_bytes: int = LOGICAL_BYTE_TARGET) -> Path:
    """Rebuild the manifest from what is on disk, without fetching.

    The manifest is written once, at the end of a run, so a fetch that is
    stopped part-way leaves the previous run's index in place. The builder
    reads that index and skips any source it lists with zero rows — so a
    source recovered since would be silently ignored even though its shard is
    sitting right there. This makes the index agree with the disk.
    """
    metas_by_slug: dict[str, dict] = {}
    salvaged = 0
    for src in SOURCES:
        d = cache / src.slug
        if not d.is_dir():
            continue
        recovered = salvage_part(cache, src)
        if recovered is not None:
            salvaged += 1
            log(f"  recovered {src.slug[:48]:<48} {recovered['rows']:>8,} rows")
            metas_by_slug[recovered["slug"]] = recovered
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            with suppress(Exception):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                metas_by_slug[meta["slug"]] = meta
    for meta_path in cache.rglob("meta.json"):
        with suppress(Exception):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            metas_by_slug[meta["slug"]] = meta
    if salvaged:
        log(f"  salvaged {salvaged} interrupted shards")
    return write_manifest(
        cache, list(metas_by_slug.values()),
        {"reindexed": True, "fetched": False,
         "target_logical_bytes": target_logical_bytes,
         "baseline_catalogue_commit": BASELINE_CATALOGUE_COMMIT},
        source_ledger,
    )


def write_source_ledger(path: Path, metas: list[dict], settings: dict) -> Path:
    """Write the durable, text-free record of every attempted source."""
    ledger = {
        "format": "atlas-source-ledger-v1",
        "baseline_catalogue_commit": BASELINE_CATALOGUE_COMMIT,
        "settings": settings,
        "sources": sorted(metas, key=lambda meta: str(meta.get("slug", ""))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def write_manifest(cache: Path, metas: list[dict], settings: dict,
                   source_ledger: Path | None = None) -> Path:
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
            "logical_bytes": logical_bytes(metas),
            "logical_byte_target": int(settings.get("target_logical_bytes", LOGICAL_BYTE_TARGET)),
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
    if source_ledger is not None:
        write_source_ledger(source_ledger, metas, settings)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path(".atlas-cache"))
    ap.add_argument("--workers", type=int, default=10,
                    help="Concurrent sources (default 10). These are independent "
                         "HTTP streams; the wall clock is the slowest source, not the sum.")
    ap.add_argument("--scale", type=float, default=BASELINE_SCALE,
                    help="Multiply the dc12d8a baseline row targets (default 3.0).")
    ap.add_argument("--target-logical-bytes", type=int, default=LOGICAL_BYTE_TARGET,
                    help="Retained UTF-8 text-byte target before JSON framing or compression.")
    ap.add_argument("--source-ledger", type=Path, default=None,
                    help="Persistent text-free source ledger path (default CACHE/source-ledger.json).")
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
    ap.add_argument("--pending", action="store_true",
                    help="Fetch only sources whose cache shard is missing or "
                         "does not meet the scaled row target.")
    ap.add_argument("--reindex", action="store_true",
                    help="Rebuild manifest.json from the shards already on disk "
                         "and recover any interrupted ones. No network. Run this "
                         "after stopping a fetch early, or the builder will "
                         "ignore everything the interrupted run had added.")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS_DEFAULT,
                    help=f"Truncate each record to this many characters "
                         f"(default {MAX_CHARS_DEFAULT}; atlas-v1 used 2000).")
    args = ap.parse_args()
    max_chars = max(MIN_CHARS + 1, args.max_chars)
    if args.scale <= 0 or args.target_logical_bytes <= 0:
        ap.error("--scale and --target-logical-bytes must be positive")
    source_ledger = args.source_ledger or args.cache / "source-ledger.json"

    if args.reindex:
        log(f"Reindexing {args.cache} …")
        path = reindex(args.cache, source_ledger=source_ledger,
                       target_logical_bytes=args.target_logical_bytes)
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

    if args.pending:
        pending: list[Source] = []
        for src in selected:
            target = max(50, int(src.target * args.scale))
            meta_path = args.cache / src.slug / "meta.json"
            shard = args.cache / src.slug / "records.jsonl.gz"
            if not meta_path.exists() or not shard.exists():
                pending.append(src)
                continue
            try:
                previous = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pending.append(src)
                continue
            if (
                previous.get("cache_format") != CACHE_FORMAT
                or previous.get("max_chars") != max_chars
                or previous.get("rows", 0) < target
            ):
                pending.append(src)
        selected = pending

    args.cache.mkdir(parents=True, exist_ok=True)
    settings = {
        "workers": args.workers, "scale": args.scale, "budget_s": args.budget,
        "stall_s": args.stall, "min_chars": MIN_CHARS, "max_chars": max_chars,
        "sources_selected": len(selected),
        "baseline_catalogue_commit": BASELINE_CATALOGUE_COMMIT,
        "target_logical_bytes": args.target_logical_bytes,
        "supplemental_reservoir": FINEWEB2_ID,
        "supplemental_languages": [lang for lang, _ in SUPPLEMENTAL_LANGUAGES],
    }

    log(f"Fetching {len(selected)} sources into {args.cache} "
        f"with {args.workers} workers (scale={args.scale}, max_chars={max_chars})")
    t0 = time.time()
    metas: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_source, src, args.cache, budget=args.budget,
                        stall=args.stall, scale=args.scale, refresh=args.refresh,
                        max_chars=max_chars, source_role="baseline"): src
            for src in selected
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                metas.append(future.result())
            except Exception as exc:
                metas.append({
                    "cache_format": CACHE_FORMAT, "slug": src.slug, "rows": 0,
                    "chars": 0, "logical_bytes": 0, "complete": False,
                    "source_role": "baseline", "target": int(src.target * args.scale),
                    "status": "unavailable",
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                })

    # Preserve all previous ledger entries during a resume, including failed
    # attempts and supplemental shards. The baseline phase always remains first.
    metas_by_slug = {meta["slug"]: meta for meta in metas}
    for meta_path in args.cache.rglob("meta.json"):
        with suppress(Exception):
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            metas_by_slug.setdefault(cached["slug"], cached)
    metas = list(metas_by_slug.values())

    if not args.only:
        baseline_bytes = logical_bytes(
            [meta for meta in metas if meta.get("source_role", "baseline") == "baseline"]
        )
        existing_supplemental = [meta for meta in metas if meta.get("source_role") == "supplemental"]
        supplemental = fetch_supplemental(
            args.cache,
            current_bytes=baseline_bytes + logical_bytes(existing_supplemental),
            budget=args.budget,
            stall=args.stall,
            refresh=args.refresh,
            max_chars=max_chars,
            existing=existing_supplemental,
            target_logical_bytes=args.target_logical_bytes,
        )
        for meta in supplemental:
            metas_by_slug[meta["slug"]] = meta
        metas = list(metas_by_slug.values())

    path = write_manifest(args.cache, metas, settings, source_ledger)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    total = manifest["totals"]["rows"]
    log(f"\nCached {total:,} rows / {manifest['totals']['logical_bytes']:,} logical UTF-8 bytes "
        f"from {manifest['totals']['sources_with_rows']} "
        f"sources in {time.time() - t0:.0f}s → {path}")
    log(f"Source ledger → {source_ledger}")
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
