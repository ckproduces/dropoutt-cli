"""Terminal output.

Same reading as the HTML page, cut to what fits on a screen someone is about to
scroll past. The order is the verdict, what would go wrong, where the data sits
on the map, and what could not be checked — and nothing else, because everything
that was previously printed and never acted on was competing with the four
things that are.

Three things moved out of here into the page rather than being repeated in both.
The dataset table, the tokenizer comparison beyond its headline, and the full
off-map diagnosis: all of them are worth reading once and none of them is worth
scrolling past every run. The last line says where to find them.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..models import Severity
from ..runner import ScanResult
from ..tokenizer_panel import BudgetReport
from .escaping import safe_snippet
from .summary import ScanSummary, build

TONE = {"block": "red", "warn": "yellow", "clean": "green"}
SEVERITY_MARK = {
    Severity.BLOCKING: ("[red]●[/red]", "red"),
    Severity.WARNING: ("[yellow]●[/yellow]", "yellow"),
    Severity.INFO: ("[cyan]●[/cyan]", "cyan"),
}

#: How many problems get their own block before the rest are summarised. A
#: terminal report that lists twenty findings is read as a wall and skipped.
SHOWN = 6


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
    report_path: str | None = None,
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
    facts.append(f"looks like [bold]{_m(s.profile)}[/bold]")
    if s.language_line:
        facts.append(_m(s.language_line))
    console.print("  " + "   [dim]·[/dim]   ".join(facts))

    _render_problems(console, s, show_evidence=show_evidence)
    _render_atlas(console, s)
    _render_unavailable(console, result)

    console.print()
    if s.notes:
        console.print(f"  [dim]{len(s.notes)} more observation"
                      f"{'' if len(s.notes) == 1 else 's'} in the report.[/dim]")
    if report_path:
        where = "Full report with the map" if (s.atlas and s.atlas.available) else "Full report"
        console.print(f"  [dim]{where}: {_m(report_path)}[/dim]")
    if not s.blocking_enabled:
        console.print("  [dim]No pass-or-fail verdict: no target declared. "
                      "Pass --target sft to turn findings into an exit code.[/dim]")
    console.print(f"  [dim]{s.records:,} records in {s.elapsed:.1f}s.[/dim]")
    console.print()


def _short(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1e9:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.1f}M"
    if count >= 1_000:
        return f"{count / 1e3:.0f}k"
    return f"{count:,}"


def _render_problems(console: Console, s: ScanSummary, *, show_evidence: bool) -> None:
    console.print()
    if not s.problems:
        console.print("  [green]No problems found.[/green]")
        return

    console.print("  [bold]What would go wrong[/bold]")
    for problem in s.problems[:SHOWN]:
        mark, style = SEVERITY_MARK.get(problem.severity, ("●", "white"))
        flag = ""
        if problem.is_blocking:
            flag = " [red]blocks this run[/red]"
        elif problem.would_block:
            flag = f" [dim]would block under {', '.join(problem.would_block)}[/dim]"
        console.print()
        console.print(f"  {mark} [bold]{_m(problem.title)}[/bold]{flag}"
                      f"  [dim]{problem.check_id}[/dim]")
        scale = problem.scale
        if problem.cost:
            scale = f"{scale}   [{style}]{problem.cost} wasted[/{style}]" if scale \
                else f"[{style}]{problem.cost} wasted[/{style}]"
        if scale:
            console.print(f"      {scale}")
        console.print(f"      [dim]{_m(problem.detail)}[/dim]")
        console.print(f"      [dim]→[/dim] {_m(problem.fix)}")
        if show_evidence and problem.evidence:
            ev = problem.evidence[0]
            console.print(f"      [dim]{_m(_where(ev))}[/dim]  "
                          f"[dim]{_m(safe_snippet(ev.excerpt, 96))}[/dim]")

    if len(s.problems) > SHOWN:
        rest = len(s.problems) - SHOWN
        console.print()
        console.print(f"  [dim]and {rest} more: "
                      f"{_m(', '.join(p.check_id for p in s.problems[SHOWN:]))}[/dim]")


def _where(evidence) -> str:
    path = evidence.source_file
    if len(path) > 44:
        path = "…" + path[-43:]
    return f"{path}:{evidence.source_index}"


def _render_atlas(console: Console, s: ScanSummary) -> None:
    """Four lines about the map, and nothing that needs a picture to make sense."""
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
        f"    Reaches [bold]{atlas.regions_touched}[/bold] of {atlas.regions_total} "
        f"areas on the map, as spread out as {atlas.effective:.0f} even ones"
        + (f" [dim]({_m(atlas.shape)})[/dim]" if atlas.shape else "")
    )
    if atlas.crowding:
        console.print(f"    [yellow]{_m(atlas.crowding)}[/yellow]")
    if atlas.twins_line and len(atlas.twins) > 0:
        console.print(f"    {_m(atlas.twins_line)}")
    console.print(f"    [dim]{_m(atlas.off_line)}[/dim]")

    if atlas.places:
        console.print()
        console.print("    [dim]Your biggest areas, named by your own record "
                      "closest to the centre of each[/dim]")
        table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1, 0, 4))
        # A floor on the share column, because rich will otherwise shrink it to
        # an ellipsis when the excerpt beside it wants the room.
        table.add_column(justify="right", no_wrap=True, min_width=5)
        table.add_column(overflow="ellipsis", no_wrap=True, ratio=1)
        for place in atlas.places[:4]:
            text = _m(place.yours or f"{place.records:,} records")
            if place.repetitive:
                text += f"  [yellow]({place.cohesion:.2f} alike)[/yellow]"
            table.add_row(f"{place.share:.0%}", text)
        console.print(table)
    if atlas.categories:
        console.print()
        console.print("    [dim]Subject areas[/dim]")
        for cat in atlas.categories[:8]:
            name = _m(str(cat.get("name", "")))
            share = float(cat.get("share", 0.0))
            map_share = cat.get("map_share")
            suffix = (
                f"  [dim](map {float(map_share):.0%})[/dim]"
                if map_share is not None else ""
            )
            console.print(f"      {share:>4.0%}  {name}{suffix}")
    if atlas.gaps_line:
        console.print(f"    [dim]{_m(atlas.gaps_line)}[/dim]")
        if atlas.gaps:
            shown = ", ".join(_m(str(g.get("name", ""))) for g in atlas.gaps[:8])
            more = len(atlas.gaps) - 8
            if more > 0:
                shown = f"{shown}, and {more} more"
            console.print(f"      [dim]{_m(shown)}[/dim]")


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
