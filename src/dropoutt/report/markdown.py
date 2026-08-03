"""The same reading of a scan, as text a machine put in front of a human.

This exists because the HTML report is read in a browser and most of the
decisions it informs are not made in one. A scan runs in CI; what a reviewer
sees is a pull-request comment or a job log. Attaching a 60 KB HTML file to
either produces something nobody opens.

So this is the third rendering of :class:`dropoutt.report.summary.ScanSummary`,
after the terminal and the page, and it is deliberately the *pasteable* one:
GitHub-flavoured Markdown, no HTML, no images, no colour, and short enough that
a reviewer reads it in the comment rather than clicking through. Where the page
draws the density grid as coloured squares this prints the numbers, because a
table of ratios is what survives being quoted back at you.

Two rules it shares with the page. Everything quoted from the corpus passes
through :func:`safe_snippet` first, and ``--no-evidence`` removes excerpts and
source locations from here exactly as it does there.
"""

from __future__ import annotations

from typing import Any

from ..fingerprint import Fingerprint
from ..runner import ScanResult
from .atlas_story import density_ratio, format_reach
from .escaping import safe_snippet
from .summary import ScanSummary, budget_rows, build

#: Findings listed in full before the rest are named and counted. A comment
#: nobody scrolls is a comment nobody reads, and the tail is in findings.jsonl.
DETAILED = 8

#: Rows of the density grid printed. The page draws all 48 because an empty row
#: is a fact; a comment cannot afford 48 rows to say "you reached nine".
GRID_ROWS = 12

_TONE = {"block": "🔴", "warn": "🟡", "clean": "🟢"}


def _pipe(value: object) -> str:
    """Escape a value for a Markdown table cell.

    A dataset called ``a|b`` would otherwise add a column to the row it sits in
    and misalign every cell after it.
    """
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[str]], align: str = "") -> list[str]:
    if not rows:
        return []
    rule = [
        "---:" if align[i:i + 1] == "r" else "---"
        for i in range(len(header))
    ]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    out += ["| " + " | ".join(_pipe(c) for c in row) + " |" for row in rows]
    return out


def render(
    result: ScanResult,
    fp: Fingerprint,
    budget: Any = None,
    *,
    include_evidence: bool = True,
    summary: ScanSummary | None = None,
) -> str:
    s = summary or build(result, budget=budget, include_evidence=include_evidence)
    root = result.ctx.root
    out: list[str] = []

    # -- the verdict, and nothing above it ---------------------------------
    out.append(f"## {_TONE.get(s.tone, '')} {s.verdict}".strip())
    if s.subtitle:
        out.append("")
        out.append(s.subtitle)

    facts = [
        f"**{s.records:,}** records",
        f"**{s.datasets}** dataset{'' if s.datasets == 1 else 's'}",
    ]
    if s.tokens:
        facts.append(f"**{_short(s.tokens)}** tokens")
    if s.language_line:
        facts.append(s.language_line)
    out += ["", " · ".join(facts), "", f"`{root}`"]

    # -- what would go wrong ------------------------------------------------
    out += ["", "### What would go wrong", ""]
    if not s.problems:
        out.append("Every check that could run came back clean.")
    else:
        for problem in s.problems[:DETAILED]:
            flag = (
                "**will block**" if problem.is_blocking
                else "**would block**" if problem.would_block
                else problem.severity.value
            )
            out.append(f"**{problem.title}** — {flag} · `{problem.check_id}`  ")
            if problem.scale:
                cost = f" · {problem.cost} wasted" if problem.cost else ""
                out.append(f"{problem.scale}{cost}  ")
            out.append(f"{problem.detail}  ")
            out.append(f"→ {problem.fix}")
            if include_evidence and problem.evidence:
                ev = problem.evidence[0]
                out += [
                    "",
                    f"> `{ev.source_file}:{ev.source_index}`  ",
                    f"> {safe_snippet(str(ev.excerpt).replace(chr(10), ' '), 160)}",
                ]
            out.append("")
        rest = s.problems[DETAILED:]
        if rest:
            out.append(
                f"…and {len(rest)} more: "
                + ", ".join(f"`{p.check_id}`" for p in rest)
            )
            out.append("")

    # -- what it costs -----------------------------------------------------
    rows = budget_rows(budget)
    if rows:
        out += ["### What it costs to tokenize", ""]
        out += _table(
            ["tokenizer", "total tokens", "vs cheapest"],
            [
                [
                    r["name"] + (" (cheapest)" if i == 0 else ""),
                    f"{r['total']:,}",
                    f"+{r['premium'] * 100:.1f}%" if r["premium"] > 0 else "—",
                ]
                for i, r in enumerate(rows)
            ],
            align="lrr",
        )
        out.append("")

    out += _atlas(s)

    # -- what could not run ------------------------------------------------
    seen: set[str] = set()
    skipped = []
    for entry in result.skipped:
        if entry.reason in seen:
            continue
        seen.add(entry.reason)
        skipped.append(entry)
    if skipped:
        out += ["### What could not be checked", ""]
        out += _table(
            ["check", "why not", "unlock"],
            [[e.title, e.reason, f"`{e.unlock}`"] for e in skipped],
        )
        out.append("")

    out += [
        "---",
        "",
        f"dropoutt {fp.pipeline_version} · fingerprint `{fp.fingerprint_id}` · "
        f"{s.records:,} records in {s.elapsed:.1f}s. "
        "Findings are structural observations about the files; no measured "
        "effect on model quality is attached to acting on any of them.",
    ]
    if not include_evidence:
        out.append("")
        out.append(
            "Record excerpts and source locations were omitted with "
            "`--no-evidence`."
        )
    return "\n".join(out).rstrip() + "\n"


def _atlas(s: ScanSummary) -> list[str]:
    """Where the corpus sits, as numbers rather than as colour."""
    atlas = s.atlas
    if atlas is None:
        return []
    out = ["### Where your data sits", ""]
    if not atlas.available:
        return [*out, atlas.unavailable_reason, ""]

    out.append(
        f"**{format_reach(atlas.effective)} of {atlas.regions_total}** in "
        f"effective coverage ({atlas.regions_touched} subregions hold any "
        f"records)"
        + (f" — {atlas.shape}." if atlas.shape else ".")
    )
    out.append("")

    if atlas.grid:
        reached = [area for area in atlas.grid if area.records]
        out.append(
            "Density is your share of a subject area against the reference "
            "corpus's share of the same one: 1.0× is as common in your data as "
            "it is on the map. Reach sums min(1, density) over subregions: "
            "parity is a full score, and over-representation does not add more."
        )
        out.append("")
        out += _table(
            ["subject area", "share", "density", "reach"],
            [
                [
                    area.name,
                    f"{area.share * 100:.1f}%",
                    density_ratio(area.ratio),
                    f"{format_reach(area.effective_reach)}/{len(area.cells)}",
                ]
                for area in reached[:GRID_ROWS]
            ],
            align="lrrr",
        )
        out.append("")
        hidden = len(reached) - GRID_ROWS
        empty = len(atlas.grid) - len(reached)
        tail = []
        if hidden > 0:
            tail.append(f"{hidden} further area{'' if hidden == 1 else 's'} reached")
        if empty:
            tail.append(
                f"{empty} of the map's {len(atlas.grid)} subject areas never reached"
            )
        if tail:
            out += ["; ".join(tail).capitalize() + ".", ""]

    for insight in atlas.insights[:3]:
        out.append(f"- **{insight.headline}** — {insight.detail}")
    if atlas.insights[:3]:
        out.append("")
    if atlas.off_line:
        out += [atlas.off_line, ""]
    return out


def _short(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1e9:.1f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.1f}M"
    if count >= 1_000:
        return f"{count / 1e3:.0f}k"
    return f"{count:,}"
