"""The report, on a terminal.

This used to be triage and nothing else: a verdict, a line of facts, one line
per finding and a note saying where the rest had been written. The reasoning was
that reprinting what is already in a file makes the scroll-back longer without
making the decision easier.

That reasoning was right about the *default* and wrong about the ceiling. The
files it pointed at are opened by roughly nobody: a scan runs on a compute node
and the person watching it has a terminal and no browser, or it runs in CI and
what a human sees is the job log. Telling them the detail exists in
``report.html`` and then not printing it means the detail does not exist.

So the full report prints here — the same sections as the page, in the same
order, drawn with what a terminal has. Where the page draws a hundred coloured
squares this prints the ten largest rows and says how many it left out; where
the page has a `<details>` element this prints the contents.

``--brief`` restores the triage view for anyone who preferred it, and ``--quiet``
still prints nothing at all.
"""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from typing import Literal

from rich.cells import cell_len, set_cell_size
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ..models import Severity
from ..runner import ScanResult
from ..tokenizer_panel import BudgetReport
from .payload import build as build_payload
from .summary import ScanSummary, build

TONE = {"block": "red", "warn": "yellow", "clean": "green"}
SEVERITY_MARK = {
    Severity.BLOCKING: ("[red]●[/red]", "red"),
    Severity.WARNING: ("[yellow]●[/yellow]", "yellow"),
    Severity.INFO: ("[cyan]●[/cyan]", "cyan"),
}
CONSEQUENCE_STYLE = {"will block": "red", "would block": "yellow"}

#: How many problems get their own line in the brief view.
SHOWN = 6

#: Width of the consequence column, sized for its longest word. Fixed, because
#: it is the column a reader scans down and a ragged one cannot be scanned.
FLAG_WIDTH = 11

#: Rows printed in full before a list says how many it left out. A terminal
#: report that prints fifty subject areas is a wall; ten is a paragraph.
LIST_ROWS = 10

#: Findings printed in full in the complete view. Past this the tail is named
#: and counted, exactly as the Markdown report does.
DETAILED = 8


def _m(value: object) -> str:
    """Escape a value before it is interpolated into a rich markup string.

    Anything derived from scanned data goes through here: a dataset named
    ``[red]`` would otherwise inject styling into our own output, and an install
    hint written as ``dropoutt[tokenizer]`` would have the bracketed word
    silently deleted as an unknown style tag.
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
    brief: bool = False,
    fingerprint=None,
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

    if brief or fingerprint is None:
        # The triage view. Also the fallback when the caller has no fingerprint
        # to build a full payload from, which is every caller that only wanted
        # a verdict.
        _render_problems_brief(console, s)
        _render_atlas_brief(console, s)
        _render_unavailable(console, result)
        _render_outputs(console, s, out_dir, written, brief=True)
        return

    data = build_payload(result, fingerprint, budget, include_evidence=show_evidence,
                         summary=s)
    _section(console, "What this corpus is")
    _render_composition(console, data["composition"])
    _section(console, "What would go wrong")
    _render_problems(console, data)
    if data["token_budget"]["tokenizers"]:
        _section(console, "What it costs to tokenize")
        _render_budget(console, data["token_budget"])
    if data["atlas"] is not None:
        _section(console, "Where your data sits")
        _render_atlas(console, data["atlas"])
    if data["notes"]:
        _section(console, "Worth knowing")
        _render_notes(console, data["notes"])
    _section(console, "The small print")
    _render_small_print(console, data)
    _render_outputs(console, s, out_dir, written, brief=False)


def render_atlas(
    console: Console,
    result: ScanResult,
    *,
    show_evidence: bool = True,
    summary: ScanSummary | None = None,
    out_dir: Path | str | None = None,
    written: list[str] | None = None,
) -> None:
    """The map, on a terminal, with nothing else on the page.

    `dropoutt atlas` prints this instead of the scan report. It is the same
    section a scan used to carry, given the whole screen: what the corpus is
    made of comes first in one line, because a coverage number is a statement
    about a corpus and the reader has to know which one.
    """
    from ..branding import mark, supports_unicode
    from .payload import build_atlas

    s = summary or build(result, include_evidence=show_evidence)
    data = build_atlas(result, include_evidence=show_evidence, summary=s)
    atlas = data["atlas"]

    console.print()
    console.print(f"  [bold cyan]{mark(unicode_ok=supports_unicode())}[/bold cyan] "
                  "[bold]Where this corpus sits on the map[/bold]")
    identity = (atlas or {}).get("identity") or {}
    if identity.get("version"):
        console.print(f"     [dim]{_m(identity['version'])}[/dim]")

    console.print()
    facts = [
        f"[bold]{data['records']:,}[/bold] records",
        f"[bold]{data['datasets']}[/bold] dataset{'' if data['datasets'] == 1 else 's'}",
    ]
    if data["language_line"]:
        facts.append(_m(data["language_line"]))
    console.print("  " + "   [dim]·[/dim]   ".join(facts))

    console.print()
    if atlas is None:
        console.print("  [dim]No records could be placed on the map.[/dim]")
    else:
        _render_atlas(console, atlas)

    if data["findings"]:
        _section(console, "What the shape means")
        for finding in data["findings"]:
            mark_, colour = SEVERITY_MARK.get(
                Severity(finding["severity"]), ("[cyan]●[/cyan]", "cyan")
            )
            console.print()
            console.print(f"    {mark_} [bold {colour}]{_m(finding['title'])}[/bold {colour}]"
                          f"  [dim]{finding['check_id']}[/dim]")
            console.print(f"      [dim]{_m(finding['detail'])}[/dim]")
            if finding["fix"]:
                console.print(f"      [green]→[/green] {_m(finding['fix'])}")

    if data["degraded"]:
        console.print()
        for note in data["degraded"]:
            console.print(f"  [yellow]note[/yellow] {_m(note)}")

    if out_dir and written:
        console.print()
        width = max(len(name) for name in written)
        console.print(f"  [dim]Written to {_m(out_dir)}[/dim]")
        for name in written:
            purpose = ARTIFACTS.get(name, "")
            console.print(f"    [bold]{_m(name):<{width}}[/bold]"
                          + (f"  [dim]{purpose}[/dim]" if purpose else ""))
    console.print(f"  [dim]{data['records']:,} records in {data['elapsed_seconds']:.1f}s. "
                  f"Run `dropoutt scan` for the checks.[/dim]")
    console.print()


# --------------------------------------------------------------------------


def _section(console: Console, title: str) -> None:
    console.print()
    console.print(f"  [bold]{title}[/bold]")


#: Column alignments a `_table` column may use. Spelled out because rich types
#: `justify` as a Literal and a bare `str` does not satisfy it.
Justify = Literal["left", "right"]


#: Subject areas whose subregions are named, before the tail is counted. The
#: map is one list now — each area's subregions sit under it rather than in a
#: second table — and a terminal cannot take four thousand lines, so the
#: nesting is capped here. The HTML page carries all of them.
AREAS_NAMED = 8


def _print_area_cells(console: Console, reached: list[dict]) -> None:
    """Each subject area, then the subregions inside it, two to a row.

    This used to be two lists printed one after another: a table of areas, then
    a flat table of the densest cells anywhere on the map. Joining them was left
    to the reader, who had no column to join on. The cells are printed under the
    area they belong to instead, so "what does *Video games* mean for my corpus"
    is answered where the question is asked.

    Two columns because a cell name is short and a single column of sixteen is a
    wall where two of eight is a block. Density leads each entry: it is what the
    cells are ranked on and what the colour used to say on the page.
    """
    named = [area for area in reached if any(c["records"] for c in area["cells"])]
    if not named:
        return
    # Two columns share the console, minus the indent and the density prefix
    # each entry carries. Clipping to a constant instead would either wrap on an
    # eighty-column terminal or waste half a wide one.
    width = max(18, (console.width - 12) // 2 - 8)
    for area in named[:AREAS_NAMED]:
        cells = sorted(
            (c for c in area["cells"] if c["records"]),
            key=lambda c: (-c.get("density", 0.0), -c["records"]),
        )
        console.print()
        console.print(
            f"    [bold]{_m(area['name'])}[/bold]  "
            f"[dim]{area['reach_label']}/{area['subregions']} reach · "
            f"{_pct(area['share'])} share · "
            f"{area['records']:,} record{'' if area['records'] == 1 else 's'}[/dim]"
        )
        table = _table(("", "left"), (" ", "left"), headers=False)
        half = (len(cells) + 1) // 2
        for left, right in zip_longest(cells[:half], cells[half:]):
            table.add_row(_cell_entry(left, width), _cell_entry(right, width))
        console.print(table)
    if len(named) > AREAS_NAMED:
        rest = len(named) - AREAS_NAMED
        console.print()
        console.print(
            f"    [dim]{rest:,} further area{'' if rest == 1 else 's'} reached; "
            f"their subregions are named in the report files[/dim]"
        )


def _cell_entry(cell: dict | None, width: int) -> str:
    """``1.5x  flowers, plant, perennial`` — density inline, name clipped to fit.

    Density shares the column rather than taking its own: two name columns and
    two number columns leaves each name about twenty-four characters on an
    eighty-column terminal, which wraps every one of them.

    The clip is by display width rather than by character count. The captions
    are multilingual and a CJK glyph occupies two terminal cells, so clipping
    ``自己中心的で心づかいができない`` to forty characters produced an eighty-cell
    string that wrapped out of its column and back to the left margin.
    """
    if cell is None:
        return ""
    caption = cell.get("caption") or f"cell {cell['region']}"
    ratio = cell.get("density") or 0.0
    density = f"{ratio:.0f}x" if ratio >= 10 else f"{ratio:.1f}x"
    caption = " ".join(str(caption).split())
    if cell_len(caption) > width:
        caption = set_cell_size(caption, max(1, width - 1)).rstrip() + "…"
    return f"[dim]{density:>5}[/dim]  {_m(caption)}"


def _table(*columns: tuple[str, Justify], headers: bool = True) -> Table:
    """A borderless table in the house style. ``(header, justify)`` per column.

    ``headers=False`` is for a table whose columns were already named by the line
    printed above it — the map prints one block per subject area, and a
    ``subregion | subregion`` header repeated over every block is noise.
    """
    table = Table(show_header=headers, header_style="dim", box=None,
                  padding=(0, 2))
    for header, justify in columns:
        table.add_column(header, justify=justify, overflow="fold")
    return table


def _short(count: int | None) -> str:
    if not count:
        return "0"
    if count >= 1_000_000_000:
        return f"{count / 1e9:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.1f}M"
    if count >= 1_000:
        return f"{count / 1e3:.0f}k"
    return f"{count:,}"


def _pct(value: float | None, places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{places}f}%"


#: Longest quoted record in a terminal table cell. The payload keeps 240
#: characters, which is right for a page and for a file; inside a five-column
#: table it wraps to eight lines and turns a ten-row table into a screen.
CELL_QUOTE = 90


def _clip(text: str, limit: int = CELL_QUOTE) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _more(console: Console, shown: int, total: int, noun: str) -> None:
    if total > shown:
        console.print(f"    [dim]and {total - shown} more {noun}[/dim]")


# --------------------------------------------------------------------------


def _render_composition(console: Console, comp: dict) -> None:
    console.print(
        f"    {comp['records']:,} records in {comp['files']} file"
        f"{'' if comp['files'] == 1 else 's'}, averaging "
        f"{comp['mean_characters_per_record']:,} characters"
        + (f" [dim]· tokens {_m(comp['token_note'])}[/dim]"
           if comp["token_note"] else "")
    )
    if comp["structured_line"]:
        console.print(f"    [dim]{_m(comp['structured_line'])}[/dim]")

    if comp["languages"]:
        table = _table(("language", "left"), ("share", "right"))
        for row in comp["languages"][:LIST_ROWS]:
            table.add_row(_m(row["code"]), _pct(row["share"]))
        console.print()
        console.print(table)
        _more(console, LIST_ROWS, len(comp["languages"]), "languages")

    if comp["layouts"]:
        table = _table(("layout", "left"), ("share", "right"), ("confidence", "right"))
        for row in comp["layouts"][:LIST_ROWS]:
            table.add_row(_m(row["label"]), _pct(row["share"], 0),
                          f"{row['confidence']:.2f}")
        console.print()
        console.print(table)

    if comp["chat_templates_in_text"]:
        console.print()
        console.print("    [yellow]Chat templates already in the text.[/yellow] "
                      "[dim]A trainer applying its own on top produces a doubled "
                      "format the model never sees at inference.[/dim]")
        table = _table(("family", "left"), ("records", "right"), ("share", "right"))
        for row in comp["chat_templates_in_text"]:
            table.add_row(_m(row["family"]), f"{row['records']:,}", _pct(row["share"]))
        console.print(table)
        if comp["target_template"]:
            console.print(f"    [dim]Your target model uses "
                          f"{_m(comp['target_template'])}.[/dim]")

    if comp["dataset_table"]:
        total = comp["records"] or 1
        table = _table(("dataset", "left"), ("layout", "left"), ("records", "right"),
                       ("share", "right"), ("licence", "left"))
        for row in comp["dataset_table"][:LIST_ROWS * 2]:
            table.add_row(
                _m(row.get("name", "")),
                _m(row.get("layout") or "—"),
                f"{row.get('records', 0):,}",
                _pct((row.get("records") or 0) / total),
                _m(row.get("licence") or "not recorded"),
            )
        console.print()
        console.print(table)
        _more(console, LIST_ROWS * 2, len(comp["dataset_table"]), "datasets")

    if comp["dataset_overlap"]:
        console.print()
        console.print("    [dim]Datasets that repeat each other. Directional: the "
                      "share of the first that also appears in the second.[/dim]")
        table = _table(("from", "left"), ("also in", "left"), ("share", "right"))
        for row in comp["dataset_overlap"][:LIST_ROWS]:
            table.add_row(_m(row.get("from", "")), _m(row.get("to", "")),
                          _pct(row.get("fraction", 0.0)))
        console.print(table)


def _render_problems(console: Console, data: dict) -> None:
    problems = data["problems"]
    if not problems:
        console.print("    [green]Every check that could run came back clean.[/green]")
        return
    for problem in problems[:DETAILED]:
        style = CONSEQUENCE_STYLE.get(problem["consequence"], "dim")
        console.print()
        console.print(
            f"    [{style}]{problem['consequence']}[/{style}]  "
            f"[bold]{_m(problem['title'])}[/bold]  [dim]{problem['check_id']}[/dim]"
        )
        line = problem["scale"]
        if problem["wasted_tokens"]:
            line += f" · {_short(problem['wasted_tokens'])} tokens wasted"
        if line:
            console.print(f"      {_m(line)}")
        console.print(f"      [dim]{_m(problem['detail'])}[/dim]")
        console.print(f"      [bold]→[/bold] {_m(problem['fix'])}")
        by_dataset = problem["by_dataset"]
        if len(by_dataset) > 1:
            worst = sorted(by_dataset.items(), key=lambda kv: -kv[1])[:4]
            console.print(
                "      [dim]worst: "
                + ", ".join(f"{_m(name)} ({count:,})" for name, count in worst)
                + "[/dim]"
            )
        for ev in problem["evidence"][:2]:
            console.print(f"      [dim]{_m(ev['source_file'])}:"
                          f"{ev['source_index']}[/dim]")
            console.print(f"      [italic]{_m(_clip(ev['excerpt'], 160))}[/italic]")
            if ev["partner_excerpt"]:
                score = f" ({_pct(ev['score'], 0)} alike)" if ev["score"] else ""
                console.print(f"      [dim]matches{score}[/dim] "
                              f"[italic]{_m(_clip(ev['partner_excerpt'], 160))}[/italic]")
    rest = problems[DETAILED:]
    if rest:
        console.print()
        console.print("    [dim]and " + ", ".join(p["check_id"] for p in rest) + "[/dim]")


def _render_budget(console: Console, budget: dict) -> None:
    table = _table(("tokenizer", "left"), ("total tokens", "right"),
                   ("± sampling", "right"), ("vs cheapest", "right"))
    for row in budget["tokenizers"]:
        name = row["name"] + (" (cheapest)" if row["cheapest"] else "")
        table.add_row(
            _m(name),
            f"{row['total_tokens']:,}",
            _pct(row["margin_share"], 2) if row["margin_share"] else "exact",
            f"+{_pct(row['premium_vs_cheapest'])}"
            if row["premium_vs_cheapest"] > 0 else "—",
        )
    console.print(table)
    if budget["method"]:
        console.print(f"    [dim]{_m(budget['method'])}[/dim]")
    for note in budget["notes"]:
        console.print(f"    [dim]{_m(note)}[/dim]")


def _render_atlas(console: Console, atlas: dict) -> None:
    if not atlas["available"]:
        console.print(f"    [dim]{_m(atlas['reason'])}[/dim]")
        return
    console.print(
        f"    Effective coverage [bold]{atlas['effective_reach_label']}[/bold] of "
        f"{atlas['subregions_total']} "
        f"({atlas['subregions_touched']} subregions hold any records)"
        + (f" [dim]({_m(atlas['shape'])})[/dim]" if atlas["shape"] else "")
    )
    console.print(
        f"    [dim]{atlas['placed_records']:,} of {atlas['sampled_records']:,} "
        f"sampled records placed · {atlas['off_map_records']:,} off the map "
        f"({_pct(atlas['off_map_rate'])}) · {atlas['too_short_to_place']:,} too "
        f"short to place[/dim]"
    )

    reached = [area for area in atlas["subject_areas"] if area["records"]]
    if reached:
        console.print()
        console.print("    [dim]Each subject area you reached, then the subregions "
                      "inside it. Density is your share of a subregion against the "
                      "map's own. 1.0× is parity.[/dim]")
        _print_area_cells(console, reached)
        empty = len(atlas["subject_areas"]) - len(reached)
        if empty:
            console.print()
            console.print(
                f"    [dim]{empty} of the map's {len(atlas['subject_areas'])} "
                f"subject areas never reached[/dim]"
            )

    for insight in atlas["insights"][:5]:
        console.print()
        console.print(f"    [bold]{_m(insight['headline'])}[/bold]")
        console.print(f"      [dim]{_m(insight['detail'])}[/dim]")
        if insight["evidence"]:
            console.print(f"      [italic]“{_m(_clip(insight['evidence'], 160))}”[/italic]")

    _render_places(console, "What you have most of", atlas["most_of"])
    _render_places(console, "What you have least of", atlas["least_of"])

    if atlas["imbalances"]:
        console.print()
        console.print("    [dim]Farthest from the map. Denser: cut volume. "
                      "Empty or thinner: grow. Grow starts at 0×.[/dim]")
        table = _table(("density", "right"), ("do", "left"), ("subject area", "left"),
                       ("share", "right"), ("one of your records", "left"))
        for item in atlas["imbalances"][:LIST_ROWS]:
            sample = _clip(item["yours"]) if item.get("yours") else ""
            if not sample:
                sample = (
                    f"{item['records']:,} records" if item["records"]
                    else "never reached"
                )
            table.add_row(
                _m(item["density_label"]), item["action"], _m(item["area"] or "—"),
                _pct(item["share"]) if item["records"] else "—",
                _m(sample),
            )
        console.print(table)

    if atlas["off_map_records"] and atlas["off_map_line"]:
        console.print()
        console.print(f"    [dim]{_m(atlas['off_map_line'])}[/dim]")
        for example in atlas["off_map_examples"][:2]:
            console.print(f"      [dim]similarity {example['score']:.2f}[/dim] "
                          f"[italic]{_m(_clip(example['excerpt'], 160))}[/italic]")


def _render_places(console: Console, title: str, places: list[dict]) -> None:
    if not places:
        return
    console.print()
    console.print(f"    [dim]{title}, against the map.[/dim]")
    table = _table(("density", "right"), ("subject area", "left"), ("share", "right"),
                   ("one of your records", "left"))
    for place in places[:LIST_ROWS]:
        text = _clip(place["yours"]) or f"{place['records']:,} records"
        if place["repetitive"]:
            text += f" — {place['cohesion']:.2f} alike, likely one template"
        table.add_row(_m(place["density_label"]), _m(place["area"] or "—"),
                      _pct(place["share"]), _m(text))
    console.print(table)


def _render_notes(console: Console, notes: list[dict]) -> None:
    for note in notes:
        console.print(f"    [cyan]note[/cyan]  [bold]{_m(note['title'])}[/bold]  "
                      f"[dim]{note['check_id']}[/dim]")
        console.print(f"      [dim]{_m(note['detail'])}[/dim]")


def _render_small_print(console: Console, data: dict) -> None:
    if data["not_checked"]:
        table = _table(("could not run", "left"), ("why not", "left"), ("unlock", "left"))
        for row in data["not_checked"]:
            table.add_row(_m(row["title"]), _m(row["reason"]), _m(row["unlock"]))
        console.print(table)
    if data["degraded"]:
        console.print(f"    [dim]{len(data['degraded'])} thing"
                      f"{'' if len(data['degraded']) == 1 else 's'} fell back rather "
                      f"than failing:[/dim]")
        for line in data["degraded"][:LIST_ROWS]:
            console.print(f"      [dim]{_m(line)}[/dim]")
        _more(console, LIST_ROWS, len(data["degraded"]), "notes")

    provenance = [(k, v) for k, v in sorted(data["provenance"].items())
                  if v not in (None, "")]
    if provenance:
        table = _table(("reproduce with", "left"), ("", "left"))
        for key, value in provenance:
            table.add_row(_m(key), _m(value))
        console.print(table)

    identity = (data["atlas"] or {}).get("identity")
    if identity:
        built = identity.get("encoder_built_with") or ""
        console.print(
            f"    [dim]Coverage measured against {_m(identity['version'])}, encoder "
            f"{_m(identity['embed_model'])} "
            f"{_m(identity['encoder_weight_hash'][:12])}"
            + (f" (quantised from {_m(built[:12])})" if built else "")
            + f", pipeline {_m(identity['pipeline_hash'][:12])}. Comparable only to "
            "numbers measured against the same three.[/dim]"
        )
    console.print(
        "    [dim]Findings are structural observations about the files. No "
        "measured effect on model quality is attached to acting on any of "
        "them.[/dim]"
    )


# -- the brief view -------------------------------------------------------


def _render_problems_brief(console: Console, s: ScanSummary) -> None:
    """One line per finding, most critical first."""
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


def _render_atlas_brief(console: Console, s: ScanSummary) -> None:
    """Three lines about the map: how broad, how lopsided, how much fell off."""
    from .atlas_story import format_reach

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
    """What could not run, and the single flag that unlocks each."""
    from .payload import skipped_checks

    rows = skipped_checks(result)
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
    "report.html": "the excerpts and every finding in full",
    "report.md": "the same, as text — paste into a PR or a ticket",
    "report.json": "the same, as data — for a dashboard or a gate",
    "findings.jsonl": "one finding per line, for scripts",
    "fingerprint.json": "comparable measurements, for diffing runs",
    "atlas.html": "the map, drawn",
    "atlas.md": "the same, as text — paste into a PR or a ticket",
    "atlas.json": "the same, as data — for a dashboard or a gate",
}


def _render_outputs(
    console: Console,
    s: ScanSummary,
    out_dir: Path | str | None,
    written: list[str] | None,
    *,
    brief: bool,
) -> None:
    """Where everything is, and what each file is for."""
    console.print()
    if brief and s.notes:
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
