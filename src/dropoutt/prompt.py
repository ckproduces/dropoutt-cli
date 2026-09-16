"""Interactive terminal choices. No extra dependency: stdin bytes and a redraw."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from rich.console import Console


class PromptError(RuntimeError):
    """The terminal cannot ask, or the user cancelled."""


def next_index(current: int, n: int, key: str) -> int:
    """Move a 0-based selection. ``key`` is ``up``, ``down``, or anything else."""
    if n <= 0:
        return 0
    if key == "up":
        return (current - 1) % n
    if key == "down":
        return (current + 1) % n
    return current


def can_prompt() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def select(
    title: str,
    choices: Sequence[tuple[str, str]],
    *,
    default: int = 0,
    console: Console | None = None,
) -> str:
    """Arrow-key menu. Returns the id of the chosen row. Raises PromptError."""
    if not choices:
        raise PromptError("nothing to choose")
    if not can_prompt():
        raise PromptError("not a terminal")
    out = console or Console()
    index = max(0, min(default, len(choices) - 1))
    drawn = False
    while True:
        if drawn:
            _clear(out, len(choices) + 2)
        _draw(out, title, choices, index)
        drawn = True
        key = _read_key()
        if key in ("enter", "space"):
            _clear(out, len(choices) + 2)
            return choices[index][0]
        if key in ("esc", "q", "ctrl-c"):
            _clear(out, len(choices) + 2)
            raise PromptError("cancelled")
        index = next_index(index, len(choices), key)


def _draw(console: Console, title: str, choices: Sequence[tuple[str, str]], index: int) -> None:
    console.print(f"  {title}")
    console.print("  [dim]↑↓ to move, enter to choose[/dim]")
    for i, (_key, label) in enumerate(choices):
        mark = "[bold cyan]>[/bold cyan]" if i == index else " "
        style = "bold cyan" if i == index else "dim"
        console.print(f"  {mark} [{style}]{label}[/{style}]")


def _clear(console: Console, lines: int) -> None:
    for _ in range(lines):
        console.file.write("\x1b[1A\x1b[2K")
    console.file.flush()


def _read_key() -> str:
    if sys.platform == "win32":
        return _read_key_windows()
    return _read_key_posix()


def _read_key_posix() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        first = sys.stdin.read(1)
        if first == "\x03":
            return "ctrl-c"
        if first in ("\r", "\n"):
            return "enter"
        if first == " ":
            return "space"
        if first in ("q", "Q"):
            return "q"
        if first == "\x1b":
            rest = sys.stdin.read(2)
            if rest == "[A":
                return "up"
            if rest == "[B":
                return "down"
            return "esc"
        return first
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key_windows() -> str:
    # typeshed declares ``msvcrt`` empty off Windows, so the body is guarded
    # by the platform rather than by a per-line ignore.
    if sys.platform != "win32":  # pragma: no cover - dispatch above prevents it
        raise PromptError("not a Windows console")
    import msvcrt

    first = msvcrt.getwch()
    if first in ("\r", "\n"):
        return "enter"
    if first == " ":
        return "space"
    if first in ("q", "Q"):
        return "q"
    if first == "\x03":
        return "ctrl-c"
    if first in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        if code == "H":
            return "up"
        if code == "P":
            return "down"
        return "esc"
    if first == "\x1b":
        return "esc"
    return first
