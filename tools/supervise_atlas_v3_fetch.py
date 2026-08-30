#!/usr/bin/env python3
"""Supervise Atlas v3 corpus fetch until the 200 GiB plan is filled.

Does not exit on source failures, child crashes, or transient Hub errors.
Writes a heartbeat/progress JSON the watch script can tail. Stop with SIGTERM
or SIGINT; anything else is retried.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from atlas_sources import (  # noqa: E402
    AXIS_FLOORS,
    AXIS_TARGET_BYTES,
    DEFAULT_CACHE,
    DEFAULT_LOGS,
    DEFAULT_STORAGE_ROOT,
    GIB,
    LOGICAL_BYTE_TARGET,
    SOURCES,
)
from atlas_fetch_progress import current_entries  # noqa: E402
from fetch_corpus import BYTE_COMPLETE_SLACK  # noqa: E402

STOP = False


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def handle_stop(signum: int, _frame) -> None:
    global STOP
    STOP = True
    print(f"received signal {signum}; will stop after the current child", flush=True)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def snapshot(cache: Path) -> dict:
    entries = current_entries(cache)
    logical = sum(int(item.get("logical_bytes", 0)) for item in entries)
    rows = sum(int(item.get("rows", 0)) for item in entries)
    axes: dict[str, dict] = {}
    for axis, target in AXIS_TARGET_BYTES.items():
        retained = sum(
            int(item.get("logical_bytes", 0))
            for item in entries
            if (item.get("requested") or {}).get("axis") == axis
        )
        axis_rows = sum(
            int(item.get("rows", 0))
            for item in entries
            if (item.get("requested") or {}).get("axis") == axis
        )
        axes[axis] = {
            "logical_bytes": retained,
            "target_bytes": target,
            "pct": round(100 * retained / target, 4) if target else 0.0,
            "rows": axis_rows,
            "floor": AXIS_FLOORS[axis],
            "meets_floor": axis_rows >= AXIS_FLOORS[axis],
        }
    running = [
        item["slug"]
        for item in entries
        if item.get("status") in {"opening", "fetching"}
    ]
    pending = []
    for source in SOURCES:
        match = next((item for item in entries if item.get("slug") == source.slug), None)
        have = int((match or {}).get("logical_bytes", 0))
        if have + BYTE_COMPLETE_SLACK < source.target_bytes:
            pending.append(
                {
                    "slug": source.slug,
                    "axis": source.axis,
                    "have_bytes": have,
                    "target_bytes": source.target_bytes,
                    "shortfall_bytes": source.target_bytes - have,
                }
            )
    return {
        "updated_at": _now(),
        "logical_bytes": logical,
        "logical_target": LOGICAL_BYTE_TARGET,
        "pct": round(100 * logical / LOGICAL_BYTE_TARGET, 4),
        "rows": rows,
        "disk_bytes": sum(path.stat().st_size for path in cache.rglob("*") if path.is_file()),
        "axes": axes,
        "running": running,
        "pending_sources": pending,
        "pending_count": len(pending),
        "floors_ok": all(info["meets_floor"] for info in axes.values()),
        "bytes_ok": logical >= int(0.995 * LOGICAL_BYTE_TARGET),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dump_fetch_list(path: Path, cache: Path) -> dict:
    snap = snapshot(cache)
    payload = {
        "plan": "atlas-v3-200gib",
        "written_at": _now(),
        "logical_target_bytes": LOGICAL_BYTE_TARGET,
        "logical_target_gib": LOGICAL_BYTE_TARGET / GIB,
        "axis_targets_gib": {axis: target / GIB for axis, target in AXIS_TARGET_BYTES.items()},
        "current_pct": snap["pct"],
        "pending": snap["pending_sources"],
        "sources": [
            {
                "slug": source.slug,
                "hf_id": source.hf_id,
                "config": source.config,
                "axis": source.axis,
                "lang": source.lang,
                "target_bytes": source.target_bytes,
                "target_gib": round(source.target_bytes / GIB, 4),
                "revision": source.revision,
                "role": source.source_role,
            }
            for source in SOURCES
        ],
        "fineweb2": "remaining non-English web quota; extra parquet files, not a rewrite",
    }
    write_json(path, payload)
    return payload


def run_fetch(cache: Path, log_path: Path, workers: int, budget: float, stall: float) -> int:
    command = [
        sys.executable,
        str(ROOT / "tools" / "fetch_corpus.py"),
        "--cache",
        str(cache),
        "--pending",
        "--workers",
        str(workers),
        "--budget",
        str(budget),
        "--stall",
        str(stall),
        "--max-chars",
        "4000",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        handle.write(f"\n--- fetch child {_now()} ---\n".encode())
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--budget", type=float, default=43_200.0)
    parser.add_argument("--stall", type=float, default=900.0)
    parser.add_argument("--heartbeat", type=float, default=20.0)
    parser.add_argument("--idle-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.workers < 1 or args.budget <= 0 or args.stall <= 0:
        parser.error("workers/budget/stall must be positive")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    args.logs.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    lock_path = args.storage / ".atlas-v3-fetch.lock"
    progress_path = args.logs / "fetch-v3-progress.json"
    list_path = args.storage / "atlas-v3-fetch-list.json"
    repo_list = ROOT / "tools" / "atlas-data" / "atlas-v3-fetch-list.json"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    child_log = args.logs / f"fetch-v3-{stamp}.log"

    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(existing.get("pid", 0))
        except Exception:
            pid = 0
        if pid and process_alive(pid):
            print(f"another supervisor is running (pid {pid})", flush=True)
            return 2
    write_json(lock_path, {"pid": os.getpid(), "started_at": _now(), "log": str(child_log)})

    rounds = 0
    try:
        dump_fetch_list(list_path, args.cache)
        dump_fetch_list(repo_list, args.cache)
        print(f"fetch list: {list_path}", flush=True)
        print(f"progress:   {progress_path}", flush=True)
        print(f"child log:  {child_log}", flush=True)
        print(
            f"watch: uv run python tools/atlas_fetch_progress.py --watch 20 --supervisor {progress_path}",
            flush=True,
        )
        while not STOP:
            rounds += 1
            snap = snapshot(args.cache)
            snap.update(
                {
                    "round": rounds,
                    "pid": os.getpid(),
                    "child_log": str(child_log),
                    "status": "complete" if snap["bytes_ok"] and snap["floors_ok"] and not snap["pending_count"] else "fetching",
                    "stop_requested": STOP,
                }
            )
            write_json(progress_path, snap)
            dump_fetch_list(list_path, args.cache)
            print(
                f"round {rounds}  {snap['pct']:.2f}%  "
                f"{snap['logical_bytes'] / GIB:.2f}/{LOGICAL_BYTE_TARGET / GIB:.0f} GiB  "
                f"pending={snap['pending_count']}  floors={'ok' if snap['floors_ok'] else 'LOW'}",
                flush=True,
            )
            if snap["bytes_ok"] and not snap["pending_count"]:
                snap["status"] = "idle-complete"
                write_json(progress_path, snap)
                print("plan filled; idling (SIGTERM to stop)", flush=True)
                slept = 0.0
                while not STOP and slept < args.idle_s:
                    time.sleep(min(args.heartbeat, args.idle_s - slept))
                    slept += args.heartbeat
                    heart = snapshot(args.cache)
                    heart.update({"round": rounds, "pid": os.getpid(), "status": "idle-complete"})
                    write_json(progress_path, heart)
                continue
            started = time.time()
            try:
                code = run_fetch(args.cache, child_log, args.workers, args.budget, args.stall)
            except Exception:
                code = 99
                with child_log.open("a", encoding="utf-8") as handle:
                    handle.write(traceback.format_exc())
                    handle.write("\n")
            elapsed = time.time() - started
            snap = snapshot(args.cache)
            snap.update(
                {
                    "round": rounds,
                    "pid": os.getpid(),
                    "last_child_exit": code,
                    "last_child_seconds": round(elapsed, 1),
                    "status": "retrying" if not STOP else "stopping",
                }
            )
            write_json(progress_path, snap)
            print(f"child exit {code} after {elapsed:.0f}s", flush=True)
            if STOP:
                break
            time.sleep(min(args.idle_s, 30.0))
    finally:
        lock_path.unlink(missing_ok=True)
        final = snapshot(args.cache)
        final.update({"pid": os.getpid(), "status": "stopped", "round": rounds})
        write_json(progress_path, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
