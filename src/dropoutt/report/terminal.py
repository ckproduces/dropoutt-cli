"""Terminal output: triage, and a map of where the rest of it went.

This is not a short version of the report. It answers one question — *is
anything wrong, and do I need to go and look* — and then says where looking
happens. Everything it used to print in full is written to a file in the same
directory milliseconds earlier: the detail and fix for each finding, the
excerpts, the dataset table, the tokenizer panel, the off-map diagnosis, the
whole density grid. Reprinting any of it made the scroll-back longer without
making the decision easier, and taught people to skim the part that *is* the
decision.

So: a verdict, one line of facts, one line per finding, three lines about the
map, what could not run, and the list of files. Roughly twenty lines whatever
the corpus. The files are named with a word each, because "where is the fix
for T0-ROLE-002" should not require knowing which of four artifacts holds it.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape

from ..models import Severity
from ..runner import ScanResult
from ..tokenizer_panel import BudgetReport
from .atlas_story import format_reach
from .summary import ScanSummary, build

TONE = {"block": "red", "warn": "yellow", "clean": "green"}
SEVERITY_MARK = {
    Severity.BLOCKING: ("[red]●[/red]", "red"),
    Severity.WARNING: ("[yellow]●[/yellow]", "yellow"),
    Severity.INFO: ("[cyan]●[/cyan]", "cyan"),
}

#: How many problems get their own line before the rest are counted. A terminal
#: report that lists twenty findings is read as a wall and skipped.
SHOWN = 6

#: Width of the consequence column, sized for its longest word. Fixed, because
#: it is the column a reader scans down and a ragged one cannot be scanned.
FLAG_WIDTH = 11


def _m(value: object) -> str:
    """Escape a value before it is interpolated into a rich markup string.

    Two distinct things are handled by the same call. Install hints are written
    as `pip install 'dropoutt[tokenizer]'`, and rich reads `[tokenizer]` as a
    style tag and silently deletes it, so the user is told to install a package
    that does not exist. Separately, dataset names and reasons come from scanned
    data, and a folder named `[red]` would otherwise inject styling into our own
    output. Anything derived from data or carrying an extras name goes here.
    """
    return escape(str(value))


def render(
    console: Console,
    result: ScanResult,
    *,
    budget: BudgetReport | None = None,
    show_evidence: bool = True,
    summary: ScanSummary | None = None,
    out_dir: Path | str | None = None,
    written: list[str] | None = None,
) -> None:
    from ..branding import mark, supports_unicode

    s = summary or build(result, budget=budget, include_evidence=show_evidence)
    colour = TONE.get(s.tone, "white")

    console.print()
    console.print(f"  [bold cyan]{mark(unicode_ok=supports_unicode())}[/bold cyan] "
                  f"[bold {colour}]{_m(s.verdict)}[/bold {colour}]")
    if s.subtitle:
        console.print(f"     {_m(s.subtitle)}")

    console.print()
    facts = [
        f"[bold]{s.records:,}[/bold] records",
        f"[bold]{s.datasets}[/bold] dataset{'' if s.datasets == 1 else 's'}",
    ]
    if s.tokens:
        facts.append(f"[bold]{_short(s.tokens)}[/bold] tokens")
    if s.language_line:
        facts.append(_m(s.language_line))
    console.print("  " + "   [dim]·[/dim]   ".join(facts))

    _render_problems(console, s)
    _render_atlas(console, s)
    _render_unavailable(console, result)
    _render_outputs(console, s, out_dir, written)


def _short(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1e9:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.1f}M"
    if count >= 1_000:
        return f"{count / 1e3:.0f}k"
    return f"{count:,}"


def _render_problems(console: Console, s: ScanSummary) -> None:
    """One line per finding, most critical first.

    It used to print the scale, the detail, the fix and an excerpt for each of
    six findings, which is most of a screen and all of it duplicated in the
    files written beside it. What a terminal is for is deciding whether to look
    further, and that decision needs a title, a consequence and an id.
    """
    console.print()
    if not s.problems:
        console.print("  [green]No problems found.[/green]")
        return

    console.print("  [bold]What would go wrong[/bold]")
    # Laid out by hand rather than with a `Table`. Rich sizes a flexible column
    # against the widest cell in it and then shrinks *every* column to fit,
    # including the fixed ones — which truncates "would block" to "would b…",
    # losing the word that carries the decision to save a title that nobody has
    # to read in full. The title is the column that should give.
    scale_width = max(
        (len(p.scale.split(" · ")[0]) for p in s.problems[:SHOWN] if p.scale),
        default=0,
    )
    title_width = max(20, console.width - FLAG_WIDTH - scale_width - 8)
    for problem in s.problems[:SHOWN]:
        mark, style = SEVERITY_MARK.get(problem.severity, ("●", "white"))
        if problem.is_blocking:
            flag, tone = "blocks", "red"
        elif problem.would_block:
            flag, tone = "would block", "yellow"
        else:
            flag, tone = problem.severity.value, "dim"
        title = problem.title
        if len(title) > title_width:
            title = title[: title_width - 1] + "…"
        scale = problem.scale.split(" · ")[0] if problem.scale else ""
        console.print(
            f"  {mark} [bold]{_m(title):<{title_width}}[/bold] "
            f"[{tone}]{flag:<{FLAG_WIDTH}}[/{tone}] "
            f"[{style}]{_m(scale):>{scale_width}}[/{style}]"
        )

    if len(s.problems) > SHOWN:
        rest = len(s.problems) - SHOWN
        console.print(f"    [dim]and {rest} more[/dim]")


def _render_atlas(console: Console, s: ScanSummary) -> None:
    """Three lines about the map: how broad, how lopsided, how much fell off.

    The place list and the subject bars that used to be here are the grid in
    the report, and the grid needs colour and a hundred rows to say what it
    says. Three sentences is what survives being read in a scroll-back.
    """
    atlas = s.atlas
    if atlas is None:
        return
    console.print()
    console.print("  [bold]Where your data sits[/bold] "
                  f"[dim]({_m(atlas.version or 'atlas')})[/dim]")
    if not atlas.available:
        console.print(f"    [dim]{_m(atlas.unavailable_reason)}[/dim]")
        return

    console.print(
        f"    Effective coverage [bold]{format_reach(atlas.effective)}[/bold] of "
        f"{atlas.regions_total} "
        f"({atlas.regions_touched} hold any records)"
        + (f" [dim]({_m(atlas.shape)})[/dim]" if atlas.shape else "")
    )
    if atlas.insights:
        console.print(f"    {_m(atlas.insights[0].headline)}")
    if atlas.off_count:
        console.print(f"    [dim]{_m(atlas.off_line.split('. ')[0])}.[/dim]")


def _render_unavailable(console: Console, result: ScanResult) -> None:
    """What could not run, and the single flag that unlocks each.

    Kept because it is the progressive-disclosure ladder made visible: it is the
    reason a first run with no flags is still worth reading, and the reason the
    second run is better.
    """
    seen: set[str] = set()
    rows = []
    for entry in result.skipped:
        if entry.reason in seen:
            continue
        seen.add(entry.reason)
        rows.append(entry)
    if not rows:
        return
    console.print()
    console.print(f"  [bold]{len(rows)} thing"
                  f"{'' if len(rows) == 1 else 's'} could not be checked[/bold]")
    for entry in rows[:5]:
        console.print(f"    [dim]{_m(entry.reason)}[/dim]"
                      + (f"  [dim]→[/dim] {_m(entry.unlock)}" if entry.unlock else ""))


#: What each artifact is for, in the fewest words that let someone pick one.
#: Keyed by filename so a file that stops being written stops being advertised.
ARTIFACTS = {
    "report.html": "the map, the excerpts, every finding in full",
    "report.md": "the same, as text — paste into a PR or a ticket",
    "findings.jsonl": "one finding per line, for scripts",
    "fingerprint.json": "comparable measurements, for diffing runs",
}


def _render_outputs(
    console: Console,
    s: ScanSummary,
    out_dir: Path | str | None,
    written: list[str] | None,
) -> None:
    """Where everything this screen left out has been put.

    The screen above is a decision, not a report. This is the part that makes
    that honest: it names each file and what it is for, so nobody has to guess
    which of four artifacts holds the fix for the finding they just read.
    """
    console.print()
    if s.notes:
        console.print(f"  [dim]{len(s.notes)} more observation"
                      f"{'' if len(s.notes) == 1 else 's'} in the report.[/dim]")
    if out_dir is not None and written:
        width = max(len(name) for name in written)
        console.print(f"  [dim]Written to {_m(out_dir)}[/dim]")
        for name in written:
            purpose = ARTIFACTS.get(name, "")
            console.print(f"    [bold]{_m(name):<{width}}[/bold]"
                          + (f"  [dim]{purpose}[/dim]" if purpose else ""))
    if not s.blocking_enabled:
        console.print("  [dim]No pass-or-fail verdict: no target declared. "
                      "Pass --target sft to turn findings into an exit code.[/dim]")
    console.print(f"  [dim]{s.records:,} records in {s.elapsed:.1f}s.[/dim]")
    console.print()
