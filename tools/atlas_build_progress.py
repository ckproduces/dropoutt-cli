#!/usr/bin/env python3
"""Show live progress for the local Atlas v2 shared-corpus build."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from atlas_sources import DEFAULT_CACHE, DEFAULT_RELEASE, DEFAULT_WORK, GIB


def directory_bytes(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def process_paused(pid: int | None) -> bool:
    """True when the process exists but is stopped (SIGSTOP), i.e. paused.

    A paused build still answers ``kill(pid, 0)``, so without this a monitor
    would report it as running with an ever-growing update age.
    """
    if not pid:
        return False
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except Exception:
        return False
    return out.startswith("T")


def bar(percent: float, width: int = 30) -> str:
    shown = max(0.0, min(100.0, percent))
    filled = int(round(width * shown / 100.0))
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent:6.2f}%"


def elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_progress(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


class SizeSampler(threading.Thread):
    """Directory sizes, measured off the render path.

    Every ``stat()`` on the build volume queues behind the build's own reads;
    measured mid-build, sizing the cache took 7-55 s and even the dozen files
    in ``work/`` cost tens of seconds, all of it blocked I/O. A frame that
    waits on that is blank for longer than its refresh interval and cannot be
    interrupted until the syscall returns. So sizes are sampled here, slowly,
    and the frame prints whatever was last measured with its age.
    """

    def __init__(self, paths: dict[str, Path], every: float = 300.0) -> None:
        super().__init__(daemon=True)
        self.paths = paths
        self.every = every
        self.sizes: dict[str, int] = {}
        self.measured_at: dict[str, float] = {}
        self._cache_done = False

    def run(self) -> None:
        while True:
            for name, path in self.paths.items():
                if name == "cache" and self._cache_done:
                    continue  # the corpus does not change during a build
                self.sizes[name] = directory_bytes(path)
                self.measured_at[name] = time.time()
                if name == "cache":
                    self._cache_done = True
            time.sleep(self.every)

    def line(self, name: str, path: Path, now: float) -> str:
        if name not in self.sizes:
            return f"{name:<8} (measuring in background)  {path}"
        age = now - self.measured_at[name]
        return f"{name:<8} {self.sizes[name] / GIB:8.2f} GiB  {path}  ({elapsed(age)} ago)"


def render(
    work: Path, cache: Path, release: Path, sampler: SizeSampler | None = None
) -> str:
    """One frame: a single small JSON read plus whatever the sampler has."""
    path = work / "build-progress.json"
    payload = load_progress(path)
    now = time.time()
    lines = [datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")]
    if payload is None:
        lines.extend(
            [
                "status   waiting for build-progress.json",
                f"path     {path}",
            ]
        )
    else:
        overall = float(payload.get("overall_pct", 0.0))
        stage = str(payload.get("stage", "unknown"))
        stage_pct = float(payload.get("stage_pct", 0.0))
        pid = int(payload.get("pid", 0) or 0)
        status = str(payload.get("status", "running"))
        if status == "running" and not process_alive(pid):
            status = "stopped"
        elif status == "running" and process_paused(pid):
            status = "paused (SIGSTOP) -- resume with: kill -CONT -- -<pgid>"
        started = float(payload.get("started_at", now))
        updated = float(payload.get("updated_at", now))
        retained = int(payload.get("retained_rows", 0))
        input_rows = int(payload.get("input_rows", 0))
        if stage == "idf":
            row_line = f"rows     {retained:,} selected for stratified IDF"
        elif stage == "dedup-recovery":
            row_line = f"rows     {retained:,} dedup hashes reconstructed"
        else:
            row_line = f"rows     {retained:,} retained / {input_rows:,} manifest rows"
        # Linear in overall_pct, which the builder already weights by stage
        # cost, so the estimate is honest across stages rather than within one.
        run_s = max(now - started, 1.0)
        base = float(payload.get("started_pct", 0.0) or 0.0)
        gain = overall - base
        if stage == "dedup-recovery" and status == "running" and 0.5 < stage_pct < 100.0:
            # The recovery tool runs before the builder and is measured on its
            # own stage bar; the overall figure is the checkpoint's, unchanged.
            left = (100.0 - stage_pct) * run_s / stage_pct
            eta_line = (
                f"eta      {elapsed(left)} left in dedup recovery, then the build resumes  "
                f"(~{datetime.fromtimestamp(now + left, timezone.utc).strftime('%H:%M UTC')})"
            )
        elif status == "running" and gain > 0.5 and overall < 100.0:
            rate = gain / run_s
            left = (100.0 - overall) / rate
            eta_line = (
                f"eta      {elapsed(left)} left at {rate * 3600:.1f}%/h  "
                f"(~{datetime.fromtimestamp(now + left, timezone.utc).strftime('%H:%M UTC')})"
            )
        elif status == "complete":
            eta_line = f"eta      done in {elapsed(updated - started)}"
        elif status.startswith("paused"):
            eta_line = "eta      paused"
        else:
            eta_line = "eta      measuring rate"
        lines.extend(
            [
                f"overall  {bar(overall)}",
                eta_line,
                f"stage    {stage:<14} {bar(stage_pct)}",
                f"status   {status}  pid={pid}  elapsed={elapsed(now - started)}  update-age={elapsed(now - updated)}",
                f"detail   {payload.get('detail', '')}",
                row_line,
            ]
        )
    lines.append("")
    if sampler is None:
        lines.append("sizes    (skipped for a one-shot; --watch samples them in the background)")
    else:
        for name, path in (("cache", cache), ("work", work), ("release", release)):
            lines.append(sampler.line(name, path, now))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--watch", type=float, default=0.0, help="Refresh interval in seconds; zero prints once."
    )
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be non-negative")
    sampler = None
    if args.watch:
        sampler = SizeSampler(
            {"cache": args.cache, "work": args.work, "release": args.release}
        )
        sampler.start()
    try:
        while True:
            # Build the frame first, then clear and paint it in one write, so
            # the screen is never blank while the frame is being made.
            frame = render(args.work, args.cache, args.release, sampler)
            prefix = "\033[2J\033[H" if args.watch else ""
            print(prefix + frame, flush=True)
            if not args.watch:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
