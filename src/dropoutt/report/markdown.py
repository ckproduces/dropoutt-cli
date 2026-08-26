"""The same reading of a scan, as text a machine put in front of a human.

This exists because the HTML report is read in a browser and most of the
decisions it informs are not made in one. A scan runs in CI; what a reviewer
sees is a pull-request comment or a job log. Attaching a 60 KB HTML file to
either produces something nobody opens.

So this is the Markdown rendering of :mod:`dropoutt.report.payload`, alongside
the page, the terminal and the JSON file, and it is deliberately the
*pasteable* one: GitHub-flavoured Markdown, no HTML, no images, no colour.

**It says everything the page says.** It used to say about a third of it — no
dataset table, no language breakdown, no structure, no places, no imbalances,
no off-map diagnosis, no provenance — which meant the reader without a browser
was handed a strictly worse report and never told what was missing. Where the
page draws the density grid as coloured squares this prints the numbers,
because a table of ratios is what survives being quoted back at you; where the
page can afford a hundred rows this prints the largest and says how many it left
out. What it no longer does is drop a section.

Two rules it shares with the page. Everything quoted from the corpus passes
through :func:`safe_snippet` first, and ``--no-evidence`` removes excerpts and
source locations from here exactly as it does there.
"""

from __future__ import annotations

from typing import Any

from ..fingerprint import Fingerprint
from ..runner import ScanResult
from .payload import build as build_payload
from .summary import ScanSummary

#: Findings listed in full before the rest are named and counted. A comment
#: nobody scrolls is a comment nobody reads, and the tail is in findings.jsonl.
DETAILED = 8

#: Rows of the density grid printed. The page draws all of them because an empty
#: row is a fact; a comment cannot afford fifty rows to say "you reached nine".
GRID_ROWS = 15

#: Longest lists that get printed in full before being counted instead.
LIST_ROWS = 10

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


def _pct(value: float | None, places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{places}f}%"


def render(
    result: ScanResult,
    fp: Fingerprint,
    budget: Any = None,
    *,
    include_evidence: bool = True,
    summary: ScanSummary | None = None,
) -> str:
    data = build_payload(result, fp, budget, include_evidence=include_evidence,
                         summary=summary)
    out: list[str] = []

    # -- the verdict, and nothing above it ---------------------------------
    verdict = data["verdict"]
    out.append(f"## {_TONE.get(verdict['tone'], '')} {verdict['headline']}".strip())
    if verdict["detail"]:
        out += ["", verdict["detail"]]

    comp = data["composition"]
    facts = [
        f"**{comp['records']:,}** records",
        f"**{comp['datasets']}** dataset{'' if comp['datasets'] == 1 else 's'}",
    ]
    if comp["tokens"]:
        facts.append(f"**{_short(comp['tokens'])}** tokens")
    if comp["language_line"]:
        facts.append(comp["language_line"])
    out += ["", " · ".join(facts), "", f"`{data['root']}`"]

    out += _composition(comp)
    out += _problems(data)
    out += _budget(data["token_budget"])
    out += _atlas(data["atlas"])
    out += _notes(data)
    out += _small_print(data)
    return "\n".join(out).rstrip() + "\n"


def render_atlas(
    result: ScanResult,
    *,
    include_evidence: bool = True,
    summary: ScanSummary | None = None,
) -> str:
    """The map as Markdown, for a pull request comment or a pipeline stage."""
    from .payload import build_atlas

    data = build_atlas(result, include_evidence=include_evidence, summary=summary)
    out: list[str] = ["## Where this corpus sits on the map"]

    identity = (data["atlas"] or {}).get("identity") or {}
    if identity.get("version"):
        out += ["", f"Measured against `{identity['version']}`."]

    facts = [
        f"**{data['records']:,}** records",
        f"**{data['datasets']}** dataset{'' if data['datasets'] == 1 else 's'}",
    ]
    if data["language_line"]:
        facts.append(data["language_line"])
    out += ["", " · ".join(facts), "", f"`{data['root']}`", ""]

    if data["atlas"] is None:
        out += ["No records could be placed on the map.", ""]
    else:
        out += _atlas(data["atlas"], heading="")

    if data["findings"]:
        out += ["### What the shape means", ""]
        for finding in data["findings"]:
            out.append(f"**{finding['title']}** · `{finding['check_id']}`")
            out += ["", finding["detail"]]
            if finding["fix"]:
                out += ["", f"→ {finding['fix']}"]
            out.append("")

    if data["degraded"]:
        out += ["### What degraded", ""]
        out += [f"- {note}" for note in data["degraded"]]
        out.append("")

    out += [
        "---",
        "",
        f"Placed in {data['elapsed_seconds']:.1f}s. This is the map only — run "
        "`dropoutt scan` for the checks.",
    ]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------


def _composition(comp: dict) -> list[str]:
    """What the files contain, before any question of what is wrong with it."""
    out = ["", "### What this corpus is", ""]
    out.append(
        f"{comp['records']:,} records in {comp['files']} file"
        f"{'' if comp['files'] == 1 else 's'}, averaging "
        f"{comp['mean_characters_per_record']:,} characters "
        f"({comp['total_characters']:,} in total)."
        + (f" Tokens: {comp['token_note']}." if comp["token_note"] else "")
    )
    out.append("")

    if comp["languages"]:
        out += _table(
            ["language", "share"],
            [[row["code"], _pct(row["share"])] for row in comp["languages"][:LIST_ROWS]],
            align="lr",
        )
        hidden = len(comp["languages"]) - LIST_ROWS
        if hidden > 0:
            out.append("")
            out.append(f"…and {hidden} more language{'' if hidden == 1 else 's'}.")
        out.append("")
    else:
        out += ["Language was not measured.", ""]

    if comp["structured_line"]:
        out += [comp["structured_line"], ""]
    if comp["layouts"]:
        out += _table(
            ["layout", "share", "confidence"],
            [
                [row["label"], _pct(row["share"], 0), f"{row['confidence']:.2f}"]
                for row in comp["layouts"][:LIST_ROWS]
            ],
            align="lrr",
        )
        out.append("")

    if comp["chat_templates_in_text"]:
        out.append(
            "**Chat templates already in the text.** These records carry turn "
            "delimiters. A trainer that applies its own template on top produces "
            "a doubled format the model never sees at inference."
            + (f" Your target model uses **{comp['target_template']}**."
               if comp["target_template"] else "")
        )
        out.append("")
        out += _table(
            ["family", "records", "share"],
            [
                [row["family"], f"{row['records']:,}", _pct(row["share"])]
                for row in comp["chat_templates_in_text"]
            ],
            align="lrr",
        )
        out.append("")

    if comp["dataset_table"]:
        total = comp["records"] or 1
        out += _table(
            ["dataset", "layout", "records", "share", "licence"],
            [
                [
                    row.get("name", ""),
                    row.get("layout") or "—",
                    f"{row.get('records', 0):,}",
                    _pct((row.get("records") or 0) / total),
                    row.get("licence") or "not recorded",
                ]
                for row in comp["dataset_table"][:25]
            ],
            align="llrrl",
        )
        hidden = len(comp["dataset_table"]) - 25
        if hidden > 0:
            out += ["", f"…and {hidden} more dataset{'' if hidden == 1 else 's'}."]
        out.append("")

    if comp["dataset_overlap"]:
        out.append(
            "**Datasets that repeat each other.** Directional: the share of the "
            "first that also appears in the second."
        )
        out.append("")
        out += _table(
            ["from", "also in", "share"],
            [
                [row.get("from", ""), row.get("to", ""), _pct(row.get("fraction", 0.0))]
                for row in comp["dataset_overlap"][:LIST_ROWS]
            ],
            align="llr",
        )
        out.append("")
    return out


def _problems(data: dict) -> list[str]:
    out = ["", "### What would go wrong", ""]
    problems = data["problems"]
    if not problems:
        out += ["Every check that could run came back clean.", ""]
        return out

    for problem in problems[:DETAILED]:
        out.append(f"**{problem['title']}** — {problem['consequence']} · "
                   f"`{problem['check_id']}`  ")
        if problem["scale"]:
            cost = (
                f" · {_short(problem['wasted_tokens'])} tokens wasted"
                if problem["wasted_tokens"] else ""
            )
            out.append(f"{problem['scale']}{cost}  ")
        out.append(f"{problem['detail']}  ")
        out.append(f"→ {problem['fix']}")
        by_dataset = problem["by_dataset"]
        if len(by_dataset) > 1:
            worst = sorted(by_dataset.items(), key=lambda kv: -kv[1])[:5]
            out.append("")
            out.append(
                "Worst datasets: "
                + ", ".join(f"`{name}` ({count:,})" for name, count in worst)
            )
        for ev in problem["evidence"][:2]:
            out += [
                "",
                f"> `{ev['source_file']}:{ev['source_index']}`  ",
                f"> {ev['excerpt']}",
            ]
            if ev["partner_excerpt"]:
                score = f" · {_pct(ev['score'], 0)} alike" if ev["score"] else ""
                out.append(f"> _matches{score}_ {ev['partner_excerpt']}")
        out.append("")

    rest = problems[DETAILED:]
    if rest:
        out += [
            f"…and {len(rest)} more: "
            + ", ".join(f"`{p['check_id']}`" for p in rest),
            "",
        ]
    return out


def _budget(budget: dict) -> list[str]:
    if not budget["tokenizers"]:
        return []
    out = ["### What it costs to tokenize", ""]
    out += _table(
        ["tokenizer", "total tokens", "± sampling", "vs cheapest"],
        [
            [
                row["name"] + (" (cheapest)" if row["cheapest"] else ""),
                f"{row['total_tokens']:,}",
                _pct(row["margin_share"], 2) if row["margin_share"] else "exact",
                f"+{_pct(row['premium_vs_cheapest'])}"
                if row["premium_vs_cheapest"] > 0 else "—",
            ]
            for row in budget["tokenizers"]
        ],
        align="lrrr",
    )
    out.append("")
    if budget["method"]:
        out += [budget["method"], ""]
    for note in budget["notes"]:
        out += [note, ""]
    return out


def _atlas(atlas: dict | None, *, heading: str = "### Where your data sits") -> list[str]:
    """Where the corpus sits, as numbers rather than as colour.

    ``heading`` is empty on the atlas page, whose title already says this.
    """
    if atlas is None:
        return []
    out = ([heading, ""] if heading else [])
    if not atlas["available"]:
        return [*out, atlas["reason"], ""]

    out.append(
        f"**{atlas['effective_reach_label']} of {atlas['subregions_total']}** in "
        f"effective coverage ({atlas['subregions_touched']} subregions hold any "
        f"records)" + (f" — {atlas['shape']}." if atlas["shape"] else ".")
    )
    out.append("")
    out += _table(
        ["", ""],
        [
            ["Placed", f"{atlas['placed_records']:,} of "
                       f"{atlas['sampled_records']:,} sampled records"],
            ["Too short to place", f"{atlas['too_short_to_place']:,}"],
            ["Off the map", f"{atlas['off_map_records']:,} "
                            f"({_pct(atlas['off_map_rate'])})"],
            ["Shape", atlas["shape"] or "—"],
            ["Evenness", _pct(atlas["concentration"], 0)
                         + " of perfectly even coverage"
                         if atlas["concentration"] is not None else "—"],
        ],
    )
    out.append("")

    areas = atlas["subject_areas"]
    if areas:
        reached = [area for area in areas if area["records"]]
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
                    area["name"],
                    _pct(area["share"]),
                    area["density_label"],
                    f"{area['reach_label']}/{area['subregions']}",
                ]
                for area in reached[:GRID_ROWS]
            ],
            align="lrrr",
        )
        out.append("")
        hidden = len(reached) - GRID_ROWS
        empty = len(areas) - len(reached)
        tail = []
        if hidden > 0:
            tail.append(f"{hidden} further area{'' if hidden == 1 else 's'} reached")
        if empty:
            tail.append(f"{empty} of the map's {len(areas)} subject areas never reached")
        if tail:
            out += ["; ".join(tail).capitalize() + ".", ""]

    if atlas["insights"]:
        out += ["**What the map says.**", ""]
        for insight in atlas["insights"]:
            line = f"- **{insight['headline']}** — {insight['detail']}"
            if insight["evidence"]:
                line += f" “{insight['evidence']}”"
            out.append(line)
        out.append("")

    out += _places("What you have most of, against the map", atlas["most_of"])
    out += _places("What you have least of, against the map", atlas["least_of"])

    if atlas["imbalances"]:
        out.append(
            "**Where you are farthest from the map.** Denser than the map means "
            "cut volume there; thinner means add more of that kind of record."
        )
        out.append("")
        out += _table(
            ["density", "do", "subject area", "share", "one of your records"],
            [
                [
                    item["density_label"],
                    item["action"],
                    item["area"] or "—",
                    _pct(item["share"]),
                    item["yours"] or f"{item['records']:,} records",
                ]
                for item in atlas["imbalances"][:LIST_ROWS]
            ],
            align="rllrl",
        )
        out.append("")

    if atlas["off_map_records"]:
        out += [f"**Why {atlas['off_map_records']:,} records could not be placed.**", ""]
        if atlas["off_map_line"]:
            out += [atlas["off_map_line"], ""]
        out += _off_detail(atlas["off_map_detail"])
        for example in atlas["off_map_examples"]:
            out.append(f"> _similarity {example['score']:.2f}_ {example['excerpt']}")
        if atlas["off_map_examples"]:
            out.append("")
    return out


def _places(title: str, places: list[dict]) -> list[str]:
    if not places:
        return []
    out = [f"**{title}.**", ""]
    out += _table(
        ["density", "subject area", "share", "one of your records"],
        [
            [
                place["density_label"],
                place["area"] or "—",
                _pct(place["share"]),
                (place["yours"] or f"{place['records']:,} records")
                + (f" — {place['cohesion']:.2f} alike, likely one template"
                   if place["repetitive"] else ""),
            ]
            for place in places[:LIST_ROWS]
        ],
        align="rlrl",
    )
    out.append("")
    return out


def _off_detail(detail: dict) -> list[str]:
    rows: list[list[str]] = []
    length = detail.get("length") or {}
    if length:
        rows.append([
            "Length",
            f"off-map median {length.get('off_median_chars')} characters, "
            f"placed median {length.get('placed_median_chars')}",
        ])
    surface = detail.get("surface") or {}
    if surface and surface.get("placed_whitespace") is not None:
        rows.append([
            "Surface",
            f"{_pct(surface['off_whitespace'], 0)} whitespace and "
            f"{_pct(surface['off_non_letter'], 0)} non-letter, against "
            f"{_pct(surface['placed_whitespace'], 0)} and "
            f"{_pct(surface['placed_non_letter'], 0)} for placed records",
        ])
    score = detail.get("score") or {}
    if score:
        line = (
            f"median similarity {score.get('off_median', 0):.2f} against a cutoff "
            f"of {score.get('cutoff', 0):.2f}"
        )
        if score.get("near_miss_share"):
            line += f"; {_pct(score['near_miss_share'], 0)} were near misses"
        rows.append(["Distance", line])
    for name, stats in sorted(
        (detail.get("by_dataset") or {}).items(), key=lambda kv: -kv[1]["rate"]
    )[:5]:
        rows.append([
            name,
            f"{_pct(stats['rate'], 0)} of its records "
            f"({stats['off_atlas']:,} of {stats['records']:,})",
        ])
    if not rows:
        return []
    return [*_table(["", ""], rows), ""]


def _notes(data: dict) -> list[str]:
    if not data["notes"]:
        return []
    out = ["### Worth knowing", ""]
    for note in data["notes"]:
        out.append(f"**{note['title']}** · `{note['check_id']}`  ")
        out += [f"{note['detail']}", ""]
    return out


def _small_print(data: dict) -> list[str]:
    out: list[str] = []
    if data["not_checked"]:
        out += ["### What could not be checked", ""]
        out += _table(
            ["check", "why not", "unlock"],
            [
                [row["title"], row["reason"], f"`{row['unlock']}`"]
                for row in data["not_checked"]
            ],
        )
        out.append("")

    if data["degraded"]:
        out += ["### What fell back rather than failing", ""]
        out += [f"- {line}" for line in data["degraded"]]
        out.append("")

    out += ["### How to reproduce this scan", ""]
    out += _table(
        ["", ""],
        [[key, f"`{value}`"] for key, value in sorted(data["provenance"].items())
         if value not in (None, "")],
    )
    out.append("")

    atlas = data["atlas"]
    identity = (atlas or {}).get("identity")
    if identity:
        out.append(
            f"Subject areas and coverage were measured against "
            f"**{identity['version']}**"
            + (f" ({identity['n_l1']} subject areas over {identity['n_regions']} cells)"
               if identity.get("n_l1") else "")
            + f", encoder `{identity['embed_model']}`"
            + (f", {identity['normalization_variant']} normalization"
               if identity.get("normalization_variant") else "")
            + f". Pipeline `{identity['pipeline_hash'][:12]}`, encoder weights "
              f"`{identity['encoder_weight_hash'][:12]}`"
            + (f", quantised from the weights the atlas was fitted on "
               f"(`{identity['encoder_built_with'][:12]}`)"
               if identity.get("encoder_built_with") else "")
            + ". Coverage numbers are only "
              "comparable to numbers measured against the same three."
        )
        out.append("")

    shards = data["shards"]
    out += [
        "---",
        "",
        f"dropoutt {data['pipeline_version']} · fingerprint "
        f"`{data['fingerprint_id']}` · {data['composition']['records']:,} records "
        f"in {data['elapsed_seconds']:.1f}s"
        + (f" across {shards} shards" if shards > 1 else "") + ". "
        "Findings are structural observations about the files; no measured "
        "effect on model quality is attached to acting on any of them.",
    ]
    if not data["blocking_enabled"]:
        out += [
            "",
            "No pass-or-fail verdict was issued because no target was declared — "
            "pass `--target sft` to turn findings into an exit code.",
        ]
    if not data["includes_evidence"]:
        out += [
            "",
            "Record excerpts and source locations were omitted with `--no-evidence`.",
        ]
    return out


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
