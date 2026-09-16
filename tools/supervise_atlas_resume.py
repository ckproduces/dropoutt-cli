#!/usr/bin/env python3
"""Wait for dedup recovery, then run build and package promotion detached."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORAGE = Path("/Volumes/ck512/dropoutt-atlas-v2")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_logged(command: list[str], log: Path) -> None:
    with log.open("ab", buffering=0) as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {command}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-pid", type=int, required=True)
    parser.add_argument("--storage", type=Path, default=DEFAULT_STORAGE)
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    work = args.storage / "work"
    release = args.storage / "release"
    logs = args.storage / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    build_log = logs / f"build-resumed-{stamp}.log"
    print(f"waiting for recovery PID {args.recovery_pid}", flush=True)
    while process_alive(args.recovery_pid):
        time.sleep(args.interval)

    summary_path = work / "recovery-summary.json"
    if not summary_path.is_file():
        raise RuntimeError("dedup recovery exited without recovery-summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads((work / "checkpoint.json").read_text(encoding="utf-8"))
    if int(summary["checkpoint_rows"]) != int(checkpoint["n"]):
        raise RuntimeError("recovery summary does not match the active checkpoint")
    print(
        f"recovery complete: {summary['dedup_hashes']:,} hashes; starting build",
        flush=True,
    )

    build_command = [
        "/usr/bin/caffeinate",
        "-dimsu",
        sys.executable,
        str(ROOT / "tools" / "build_atlas_v2.py"),
        "--cache",
        str(args.storage / "corpus-cache"),
        "--work",
        str(work),
        "--out-dir",
        str(release),
    ]
    run_logged(build_command, build_log)
    print("build complete; validating and promoting artifacts", flush=True)
    promote_command = [
        sys.executable,
        str(ROOT / "tools" / "promote_atlas_v2.py"),
        "--release",
        str(release),
        "--package-dir",
        str(ROOT / "src" / "dropoutt" / "data" / "atlas"),
    ]
    run_logged(promote_command, build_log)
    print(f"resume pipeline complete; log={build_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
