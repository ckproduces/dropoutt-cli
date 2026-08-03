"""What a scan shows while it is working.

Two things live here, and neither is about the command surface: how the scan
reports progress, and how it starts loading model files before anything needs
them. Both are about the shape of a long-running job.
"""

from __future__ import annotations

import contextlib
import sys

from rich.console import Console
from rich.markup import escape

console = Console()


class ProgressDisplay:
    """A bar while records are streaming, a spinner for everything else.

    Two modes on purpose. In a terminal the scan is one long phase with a
    countable unit, so it gets a real bar with a rate and a remaining time —
    "2,000 records" told a user nothing about whether to wait or make coffee.
    When output is redirected there is no bar to redraw, so phases print once
    and record counts print at intervals, which is what CI logs want.

    The record total is an estimate from bytes on disk, so the bar is capped
    rather than allowed to run past 100% on a corpus of unusually short records.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._progress = None
        self._status = None
        self._task = None
        self._last_phase = ""
        self._last_printed = 0
        self._live = enabled and console.is_terminal and sys.stdout.isatty()

    def __enter__(self) -> ProgressDisplay:
        # Rich's Console is created at import time; under CliRunner / pipes
        # `console.is_terminal` can stay True while stdout is not a TTY. Gate
        # the live display on the real stream so redirected output gets stable
        # phase lines (what CI and tests assert on).
        if self._live:
            self._status = console.status("", spinner="dots")
            self._status.start()
        return self

    def phase(self, label: str) -> None:
        if not self.enabled or label == self._last_phase:
            return
        self._last_phase = label
        self._close_bar()
        if self._status is not None:
            self._status.update(f"[cyan]working[/cyan] {escape(label)}")
        else:
            console.print(f"  [dim]{escape(label)}...[/dim]")

    def records(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        if not self._live:
            # One line per 20k records. Redirected output has no bar to redraw,
            # so this is what a CI log gets.
            if done and done >= self._last_printed + 20_000:
                self._last_printed = done
                console.print(f"  [dim]Scanned {done:,} of ~{total:,} records[/dim]")
            return
        if self._progress is None:
            self._start_bar(total)
        self._progress.update(self._task, completed=min(done, total), total=max(total, done))

    def _start_bar(self, total: int) -> None:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        if self._status is not None:
            self._status.stop()
            self._status = None
        self._progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[cyan]scanning[/cyan]"),
            BarColumn(bar_width=28, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]records[/dim]"),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=True,
        )
        self._progress.start()
        self._task = self._progress.add_task("scan", total=total)

    def _close_bar(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task = None
            if self._live and self._status is None:
                self._status = console.status("", spinner="dots")
                self._status.start()

    def finish(self, label: str | None = None) -> None:
        self._close_bar()
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self.enabled and label:
            console.print(f"  [green]done[/green] {escape(label)}")

    def __exit__(self, *_args) -> None:
        self.finish()


class _NullWarmup:
    def shutdown(self, wait: bool = True) -> None:
        return


def start_warmup(*, offline: bool, want_embedder: bool, want_panel: bool):
    """Begin loading the model files a scan will need, off the critical path.

    Returns something with ``shutdown(wait=True)``. Nothing here touches scan
    state or raises into the scan: a warm-up that fails simply means the load
    happens later, where it would have happened anyway.
    """
    jobs = []
    if want_embedder:
        def _embedder() -> None:
            from .atlas import DEFAULT_MODEL, EMBED_DIM, load_embedder

            with contextlib.suppress(Exception):
                load_embedder(DEFAULT_MODEL, offline=offline, out_dim=EMBED_DIM)

        jobs.append(_embedder)
    if want_panel:
        from .tokenizer_panel import warm_panel

        jobs.append(lambda: warm_panel(offline=offline))
    if not jobs:
        return _NullWarmup()

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="dropoutt-warm")
    for job in jobs:
        pool.submit(job)
    return pool
