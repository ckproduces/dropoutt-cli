#!/usr/bin/env python3
"""Report live byte-based progress for the frozen Atlas corpus fetch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_sources import (
    AXIS_TARGET_BYTES,
    DEFAULT_CACHE,
    DEFAULT_LOGS,
    GIB,
    LOGICAL_BYTE_TARGET,
    PLANNED_LANGUAGE_BYTES,
)


def cache_bytes(cache: Path) -> int:
    return sum(path.stat().st_size for path in cache.rglob("*") if path.is_file())


def pct_bar(value: int, target: int, width: int = 24) -> str:
    percent = 100 * value / target if target else 0.0
    shown = max(0.0, min(100.0, percent))
    filled = int(round(width * shown / 100))
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:6.2f}%"


def _load(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def current_entries(cache: Path) -> list[dict]:
    """Merge completed metadata with live progress, keeping the larger byte count.

    ``progress.json`` is truncated to zero when a source reopens at the start of
    a fetch round, so it must never be allowed to overwrite the ``meta.json``
    left by an earlier completed pass. A supervisor killed mid-round used to
    leave the monitor reporting books at 0.24% while the ledger and the shards
    on disk both held 28.6% — the bytes were never lost, only the accounting.
    Take the live status either way, so a reopened source still counts as
    running, but keep whichever record actually saw more bytes.
    """
    entries: dict[str, dict] = {}
    for path in cache.rglob("meta.json"):
        payload = _load(path)
        if payload and payload.get("slug"):
            entries[str(payload["slug"])] = payload
    for path in cache.rglob("progress.json"):
        payload = _load(path)
        if not (payload and payload.get("slug")):
            continue
        slug = str(payload["slug"])
        settled = entries.get(slug)
        if settled is None:
            entries[slug] = payload
            continue
        live_bytes = int(payload.get("logical_bytes", 0) or 0)
        settled_bytes = int(settled.get("logical_bytes", 0) or 0)
        # A source raised above its old quota keeps the shard it already has
        # and appends the difference, so meta.json and progress.json describe
        # two different, additive spans -- taking the larger hides every new
        # byte until the new run passes the old total, which reads as a stall
        # for hours. A closed records.jsonl.gz beside a live .part is what
        # tells the two cases apart: with both, add; with only the part file,
        # progress.json is the whole story and max still guards a reopen that
        # truncated it to zero.
        directory = path.parent
        extending = (
            (directory / "records.jsonl.gz").exists()
            and (directory / "records.jsonl.gz.part").exists()
        )
        if extending:
            merged = dict(payload)
            merged["logical_bytes"] = settled_bytes + live_bytes
            merged["rows"] = int(settled.get("rows", 0) or 0) + int(
                payload.get("rows", 0) or 0
            )
            entries[slug] = merged
        elif live_bytes >= settled_bytes:
            entries[slug] = payload
        else:
            merged = dict(settled)
            merged["status"] = payload.get("status", settled.get("status"))
            entries[slug] = merged
    return list(entries.values())


ETA_WINDOW_S = 1800.0
ETA_MIN_SPAN_S = 120.0


def _eta(cache: Path, logical: int) -> str:
    """Time left at the rate actually observed over the last half hour.

    Rate is measured, not derived from a plan: sources differ in throughput by
    two orders of magnitude (Wikipedia streams, common-pile books managed about
    84 KiB/s), so any estimate from remaining bytes alone would be fiction.
    Samples persist next to the cache, so the estimate survives a restart of
    this monitor -- though not of the fetch, whose rate genuinely changes.
    """
    path = cache.parent / "logs" / "fetch-v3-eta.json"
    now = time.time()
    try:
        samples = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        samples = []
    samples = [s for s in samples if isinstance(s, list) and now - s[0] <= ETA_WINDOW_S]
    samples.append([now, logical])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(samples[-400:]), encoding="utf-8")
    except OSError:
        pass
    remaining = LOGICAL_BYTE_TARGET - logical
    if remaining <= 0:
        return "eta      target reached"
    if len(samples) < 2 or now - samples[0][0] < ETA_MIN_SPAN_S:
        return f"eta      measuring rate ({remaining / GIB:.1f} GiB to go)"
    span = now - samples[0][0]
    gained = logical - samples[0][1]
    if gained <= 0:
        return f"eta      no progress in {span / 60:.0f} min ({remaining / GIB:.1f} GiB to go)"
    rate = gained / span
    left = remaining / rate
    hours, minutes = divmod(int(left // 60), 60)
    when = datetime.fromtimestamp(now + left, timezone.utc).strftime("%H:%M UTC")
    return (
        f"eta      {hours}h{minutes:02d}m left at {rate * 3600 / GIB:.2f} GiB/h"
        f"  ({remaining / GIB:.1f} GiB to go, ~{when})"
    )


def render(cache: Path, *, all_languages: bool, supervisor: Path | None = None) -> str:
    entries = current_entries(cache)
    logical = sum(int(item.get("logical_bytes", 0)) for item in entries)
    rows = sum(int(item.get("rows", 0)) for item in entries)
    axes = dict.fromkeys(AXIS_TARGET_BYTES, 0)
    languages = dict.fromkeys(PLANNED_LANGUAGE_BYTES, 0)
    running = complete = partial = 0
    for item in entries:
        requested = item.get("requested", {})
        axis = requested.get("axis")
        lang = requested.get("lang", requested.get("language"))
        retained = int(item.get("logical_bytes", 0))
        if axis in axes:
            axes[axis] += retained
        if lang in languages:
            languages[lang] += retained
        status = item.get("status")
        if status in {"opening", "fetching"}:
            running += 1
        elif item.get("complete") or status == "complete":
            complete += 1
        elif retained:
            partial += 1

    lines = [
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        f"overall  {pct_bar(logical, LOGICAL_BYTE_TARGET)}  "
        f"{logical / GIB:.2f} / {LOGICAL_BYTE_TARGET / GIB:.0f} GiB retained",
        f"disk     {cache_bytes(cache) / GIB:.2f} GiB compressed cache",
        f"rows     {rows:,}",
        f"sources  {complete} complete, {partial} partial, {running} running",
        _eta(cache, logical),
    ]
    sup = _load(supervisor) if supervisor else _load(DEFAULT_LOGS / "fetch-v3-progress.json")
    if sup:
        stale = ""
        updated = str(sup.get("updated_at") or "")
        if updated:
            try:
                stamp = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                lag = (datetime.now(timezone.utc) - stamp).total_seconds()
                if lag > 90:
                    stale = f"  (stale {int(lag)}s; ignore its % — overall bar is live)"
            except ValueError:
                stale = "  (timestamp unreadable)"
        lines.append(
            f"supervisor  status={sup.get('status')}  round={sup.get('round')}  "
            f"pending={sup.get('pending_count')}  pid={sup.get('pid')}  "
            f"updated={updated}{stale}"
        )
        running_slugs = sup.get("running") or []
        if running_slugs:
            lines.append("  live: " + ", ".join(str(slug)[:48] for slug in running_slugs[:8]))
    lines.extend(["", "axis completion:"])
    for axis, target in AXIS_TARGET_BYTES.items():
        lines.append(
            f"  {axis:<18} {pct_bar(axes[axis], target, 18)}  "
            f"{axes[axis] / GIB:6.2f}/{target / GIB:6.2f} GiB"
        )

    # Every language, three to a row. This used to show the twelve largest,
    # which hides exactly the ones worth watching: a language only gets its own
    # mean vector fitted if it lands enough records, and it is the small ones
    # that fall short. `!` marks a language under a tenth of its target.
    done = sum(
        1 for lang, target in PLANNED_LANGUAGE_BYTES.items()
        if target and languages[lang] >= target
    )
    lines.extend(["", f"language completion  ({done}/{len(PLANNED_LANGUAGE_BYTES)} at target):"])
    ordered = list(PLANNED_LANGUAGE_BYTES)
    cells = []
    for lang in ordered:
        target = PLANNED_LANGUAGE_BYTES[lang]
        got = languages[lang]
        ratio = got / target if target else 1.0
        flag = "!" if ratio < 0.10 else (" " if ratio < 1.0 else "*")
        cells.append(
            f"{lang:<3}{flag}{pct_bar(got, target, 8)} {got / GIB:6.2f}/{target / GIB:5.2f}G"
        )
    per_row = 1 if all_languages else 3
    for start in range(0, len(cells), per_row):
        lines.append("  " + "  ".join(cells[start:start + per_row]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--watch", type=float, default=0.0, help="Reprint every N seconds. Does not clear the screen."
    )
    parser.add_argument("--all-languages", action="store_true")
    parser.add_argument(
        "--pct",
        action="store_true",
        help="Print one live percentage line from the cache and exit.",
    )
    parser.add_argument(
        "--supervisor",
        type=Path,
        default=None,
        help="Supervisor heartbeat JSON. Default: storage logs/fetch-v3-progress.json.",
    )
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be non-negative")
    try:
        while True:
            if args.pct:
                entries = current_entries(args.cache)
                logical = sum(int(item.get("logical_bytes", 0)) for item in entries)
                print(
                    f"{100 * logical / LOGICAL_BYTE_TARGET:.2f}%  "
                    f"{logical / GIB:.2f}/{LOGICAL_BYTE_TARGET / GIB:.0f} GiB",
                    flush=True,
                )
                return 0
            if args.watch:
                print("\n-----", datetime.now(timezone.utc).strftime("%H:%M:%S UTC"), "-----", flush=True)
            print(
                render(
                    args.cache,
                    all_languages=args.all_languages,
                    supervisor=args.supervisor,
                ),
                flush=True,
            )
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
