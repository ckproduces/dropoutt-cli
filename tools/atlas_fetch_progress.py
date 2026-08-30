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
    LANGUAGE_TARGET_BYTES,
    LOGICAL_BYTE_TARGET,
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
    """Use live progress when present and completed metadata otherwise."""
    entries: dict[str, dict] = {}
    for path in cache.rglob("meta.json"):
        payload = _load(path)
        if payload and payload.get("slug"):
            entries[str(payload["slug"])] = payload
    for path in cache.rglob("progress.json"):
        payload = _load(path)
        if payload and payload.get("slug"):
            entries[str(payload["slug"])] = payload
    return list(entries.values())


def render(cache: Path, *, all_languages: bool, supervisor: Path | None = None) -> str:
    entries = current_entries(cache)
    logical = sum(int(item.get("logical_bytes", 0)) for item in entries)
    rows = sum(int(item.get("rows", 0)) for item in entries)
    axes = dict.fromkeys(AXIS_TARGET_BYTES, 0)
    languages = dict.fromkeys(LANGUAGE_TARGET_BYTES, 0)
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
    ]
    sup = _load(supervisor) if supervisor else _load(DEFAULT_LOGS / "fetch-v3-progress.json")
    if sup:
        lines.append(
            f"supervisor  status={sup.get('status')}  round={sup.get('round')}  "
            f"pending={sup.get('pending_count')}  pid={sup.get('pid')}  "
            f"updated={sup.get('updated_at')}"
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

    lines.extend(["", "language completion:"])
    ordered = list(LANGUAGE_TARGET_BYTES)
    if not all_languages:
        ordered = ordered[:12]
        if "tr" not in ordered:
            ordered.append("tr")
    for lang in ordered:
        target = LANGUAGE_TARGET_BYTES[lang]
        lines.append(
            f"  {lang:<4} {pct_bar(languages[lang], target, 18)}  "
            f"{languages[lang] / GIB:6.2f}/{target / GIB:6.2f} GiB"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--watch", type=float, default=0.0, help="Refresh interval in seconds; zero prints once."
    )
    parser.add_argument("--all-languages", action="store_true")
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
            if args.watch:
                print("\033[2J\033[H", end="")
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
