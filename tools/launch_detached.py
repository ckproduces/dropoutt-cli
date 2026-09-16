#!/usr/bin/env python3
"""Run a command in its own session, out of reach of the launching shell.

``nohup`` only ignores SIGHUP. When the shell that started a background job
is torn down and its whole process group is killed, the job dies with it --
which is how a 38-hour atlas build and its bash wrapper both vanished without
a word in any log. ``setsid`` makes the child the leader of a new session and
process group, so a group signal aimed at the launcher never reaches it. macOS
ships no ``setsid`` binary; this is that binary.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: launch_detached.py <command> [args...]", file=sys.stderr)
        return 2
    # setsid() is refused for a process that already leads a group -- which a
    # shell with job control makes every background job. Fork first: the
    # child is never a leader, so it can always start a session of its own,
    # and the parent returns at once so the caller's `&` and `$!` stay cheap.
    if os.fork():
        return 0
    os.setsid()
    os.execvp(sys.argv[1], sys.argv[1:])
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
