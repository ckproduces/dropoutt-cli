"""The four output formats have to say the same things.

They had drifted. The HTML page carried the dataset table, the density grid, the
places lists, the imbalances, the off-map diagnosis, the degradations and the
provenance block; the Markdown file carried about a third of that; the terminal
carried a tenth; and there was no JSON report at all. Every reader without a
browser was handed a strictly worse report and never told what was missing.

These tests pin the fix: one payload in `dropoutt.report.payload`, four
renderings of it, and no format quietly dropping a section.
"""

from __future__ import annotations

import io
import json
import re

import pytest
from rich.console import Console

from dropoutt.fingerprint import build as build_fingerprint
from dropoutt.report import json_report
from dropoutt.report import markdown as md_report
from dropoutt.report import payload as payload_mod
from dropoutt.report import terminal as term_report
from dropoutt.runner import scan


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """A corpus with something to say in every section of the report."""
    root = tmp_path_factory.mktemp("corpus")
    for name, language in (("turkish", "tr"), ("english", "en")):
        folder = root / name
        folder.mkdir()
        rows = []
        for i in range(60):
            if language == "tr":
                body = (
                    f"Bu {i} numarali kayittir ve yeterince uzun bir metin icermektedir "
                    "cunku dil tespiti kisa metinlerde guvenilir degildir. "
                ) * 2
            else:
                body = (
                    f"This is record number {i} and it carries enough text for the "
                    "language identifier to be confident about what it is reading. "
                ) * 2
            if i % 9 == 0:
                body = "Repeated boilerplate paragraph appearing many times over. " * 6
            rows.append(json.dumps({"messages": [
                {"role": "user", "content": f"Explain subject {i % 7} in detail please."},
                {"role": "assistant", "content": body},
            ]}))
        (folder / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # The atlas is passed explicitly, as the CLI does. Without it the scan
    # produces no coverage and the parity tests below would silently pass by
    # having nothing to compare.
    from dropoutt.atlas import load_bundled
    from dropoutt.langid import LanguageDetector

    result = scan(str(root), detector=LanguageDetector(), atlas=load_bundled())
    fp = build_fingerprint(
        result.ctx, result.findings, total_chars=100_000, total_words=15_000
    )
    return result, fp


def _payload(scanned, **kwargs):
    result, fp = scanned
    return payload_mod.build(result, fp, None, **kwargs)


def _name_in_report(name: str, text: str) -> bool:
    """Match a subject-area label in prose or in a truncated terminal table."""
    if name in text:
        return True
    flat = re.sub(r"\s+", " ", text)
    if name in flat:
        return True
    # Rich truncates long labels mid-cell; the leading clause is still distinctive.
    head = name.split(",", 1)[0]
    return head in text or head in flat


def test_the_payload_carries_every_section_the_page_has(scanned):
    data = _payload(scanned)
    for key in ("verdict", "composition", "problems", "notes", "token_budget",
                "atlas", "not_checked", "degraded", "provenance", "capabilities"):
        assert key in data, key
    for key in ("records", "datasets", "files", "languages", "layouts",
                "dataset_table", "dataset_overlap", "chat_templates_in_text",
                "mean_characters_per_record", "total_characters"):
        assert key in data["composition"], key


def test_the_payload_is_json_serialisable_and_the_json_report_is_it(scanned):
    result, fp = scanned
    text = json_report.render(result, fp, None)
    parsed = json.loads(text)
    assert parsed["schema"] == "dropoutt.report/1"
    assert parsed == json.loads(json.dumps(_payload(scanned)))


def test_every_finding_reaches_every_format(scanned):
    """A finding the page shows and the log does not is a finding nobody acts on."""
    result, fp = scanned
    data = _payload(scanned)
    assert data["problems"], "the fixture is meant to produce findings"

    markdown = md_report.render(result, fp, None)
    buffer = io.StringIO()
    term_report.render(
        Console(file=buffer, width=200, no_color=True), result,
        summary=None, fingerprint=fp,
    )
    terminal = buffer.getvalue()
    text = json_report.render(result, fp, None)

    for problem in data["problems"][:md_report.DETAILED]:
        for rendering, name in ((markdown, "markdown"), (terminal, "terminal"),
                                (text, "json")):
            assert problem["check_id"] in rendering, f"{problem['check_id']} missing from {name}"


def test_the_composition_reaches_markdown_and_the_terminal(scanned):
    """The half of the report that is not a complaint used to be page-only."""
    result, fp = scanned
    data = _payload(scanned)
    markdown = md_report.render(result, fp, None)
    buffer = io.StringIO()
    term_report.render(
        Console(file=buffer, width=200, no_color=True), result, fingerprint=fp
    )
    terminal = buffer.getvalue()

    for rendering in (markdown, terminal):
        assert "What this corpus is" in rendering
        # Every dataset appears by name in the dataset table.
        for row in data["composition"]["dataset_table"]:
            assert row["name"] in rendering
        # And the language breakdown is there, not just the one-line summary.
        for language in data["composition"]["languages"][:3]:
            assert language["code"] in rendering


def test_the_atlas_detail_reaches_markdown_and_the_terminal(scanned):
    result, fp = scanned
    data = _payload(scanned)
    if data["atlas"] is None or not data["atlas"]["available"]:
        pytest.skip("no atlas coverage in this environment")

    markdown = md_report.render(result, fp, None)
    buffer = io.StringIO()
    term_report.render(
        Console(file=buffer, width=200, no_color=True), result, fingerprint=fp
    )
    terminal = buffer.getvalue()

    for rendering in (markdown, terminal):
        assert "Where your data sits" in rendering
        reached = [a for a in data["atlas"]["subject_areas"] if a["records"]]
        for area in reached[:5]:
            assert _name_in_report(area["name"], rendering)


def test_no_evidence_removes_quotes_from_all_four_formats(scanned, tmp_path):
    """One flag, honoured in one place, so it cannot be honoured in three."""
    result, fp = scanned
    quiet = _payload(scanned, include_evidence=False)
    assert all(not p["evidence"] for p in quiet["problems"])
    assert all(not p["evidence"] for p in quiet["notes"])
    if quiet["atlas"] and quiet["atlas"]["available"]:
        assert not quiet["atlas"]["off_map_examples"]
        assert all(not p["yours"] for p in quiet["atlas"]["most_of"])
        assert all(not i["yours"] for i in quiet["atlas"]["imbalances"])

    markdown = md_report.render(result, fp, None, include_evidence=False)
    assert "--no-evidence" in markdown
    text = json_report.render(result, fp, None, include_evidence=False)
    assert json.loads(text)["includes_evidence"] is False


def test_the_brief_terminal_view_is_still_available(scanned):
    """`--brief` is the old triage screen, kept for people who preferred it."""
    result, fp = scanned
    full = io.StringIO()
    term_report.render(Console(file=full, width=200, no_color=True), result,
                       fingerprint=fp)
    brief = io.StringIO()
    term_report.render(Console(file=brief, width=200, no_color=True), result,
                       fingerprint=fp, brief=True)

    assert len(brief.getvalue()) < len(full.getvalue())
    assert "What this corpus is" not in brief.getvalue()
    # The verdict survives, because that is the whole point of the brief view.
    assert "What would go wrong" in brief.getvalue()


def test_a_dataset_named_with_markup_cannot_style_the_terminal(scanned, tmp_path):
    """Everything derived from scanned data is escaped at the render site."""
    root = tmp_path / "hostile"
    (root / "[red]bold[/red]").mkdir(parents=True)
    (root / "[red]bold[/red]" / "train.jsonl").write_text(
        json.dumps({"text": "a record long enough to be read by the scanner"}) + "\n",
        encoding="utf-8",
    )
    result = scan(str(root))
    fp = build_fingerprint(result.ctx, result.findings, total_chars=50, total_words=10)
    buffer = io.StringIO()
    term_report.render(Console(file=buffer, width=200, no_color=True), result,
                       fingerprint=fp)
    assert "[red]bold[/red]" in buffer.getvalue()


# -- one story, four renderings ---------------------------------------------


def _flat(text: str) -> str:
    return " ".join(text.split())


def _terminal(result, fp, width: int = 400) -> str:
    buffer = io.StringIO()
    term_report.render(Console(file=buffer, width=width, no_color=True), result,
                       fingerprint=fp)
    return buffer.getvalue()


def _strings(value) -> list[str]:
    """Every string anywhere inside a parsed JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_a_hostile_record_excerpt_is_made_safe_before_any_rendering(scanned):
    """The story is sanitised once, above every renderer.

    The page rendered the record nearest each place straight from the story
    while the other three read it through the payload's `safe_snippet`, so
    `atlas.md` showed a control picture where `atlas.html` carried the raw
    byte. A bidi override in one record would have reversed the rest of the
    page.
    """
    from dropoutt.report import html as html_report

    result, fp = scanned
    data = _payload(scanned)
    if data["atlas"] is None or not data["atlas"]["available"] or not data["atlas"]["most_of"]:
        pytest.skip("no atlas coverage in this environment")

    hostile = "kodlama\x07 var \u202ereversed\u202c end"
    examples = result.ctx.stats.get("atlas_region_examples") or {}
    saved = {region: [dict(row) for row in rows] for region, rows in examples.items()}
    try:
        for rows in examples.values():
            for row in rows:
                row["excerpt"] = hostile
        renderings = {
            "report.md": md_report.render(result, fp, None),
            "atlas.md": md_report.render_atlas(result),
            "report.html": html_report.render(result, fp, None),
            "atlas.html": html_report.render_atlas(result),
            "terminal": _terminal(result, fp),
        }
        parsed = json.loads(json_report.render(result, fp, None))
    finally:
        for region, rows in saved.items():
            examples[region] = rows

    for name, out in renderings.items():
        assert "\x07" not in out, name
        assert "\u202e" not in out, name
    for text in _strings(parsed["atlas"]):
        assert "\x07" not in text and "\u202e" not in text
    # The excerpt did reach every rendering; it just arrived made visible.
    for name in ("report.md", "atlas.md", "report.html", "atlas.html"):
        assert "kodlama␇ var" in renderings[name], name


def test_the_rebalance_section_describes_and_does_not_prescribe(scanned):
    """Nothing tells the reader to cut or grow anything.

    The tool has not been told what the corpus is for, so a cell denser than
    the map is a description, not a fault. One caption says so, the same one
    in every rendering, and the JSON names a direction rather than an action.
    """
    from markupsafe import escape as html_escape

    from dropoutt.report import html as html_report
    from dropoutt.report.atlas_story import IMBALANCE_CAPTION

    result, fp = scanned
    data = _payload(scanned)
    atlas = data["atlas"]
    if atlas is None or not atlas["available"] or not atlas["imbalances"]:
        pytest.skip("no atlas coverage in this environment")

    assert atlas["imbalances_note"] == IMBALANCE_CAPTION
    for item in atlas["imbalances"]:
        assert "action" not in item
        assert item["direction"] in {"denser", "thinner", "unreached"}
        assert (item["direction"] == "unreached") == (item["records"] == 0)
        assert (item["direction"] == "denser") == (item["density"] > 1.0)

    markdown = md_report.render(result, fp, None)
    terminal = _terminal(result, fp)
    page = html_report.render(result, fp, None)
    atlas_page = html_report.render_atlas(result)

    assert IMBALANCE_CAPTION in _flat(markdown)
    assert IMBALANCE_CAPTION in _flat(terminal)
    assert str(html_escape(IMBALANCE_CAPTION)) in _flat(page)
    assert str(html_escape(IMBALANCE_CAPTION)) in _flat(atlas_page)
    for name, out in (("markdown", markdown), ("terminal", terminal),
                      ("page", page), ("atlas page", atlas_page)):
        low = out.lower()
        for phrase in ("cut volume", "add that kind of record", "use fewer records",
                       ">cut<", ">grow<", "grow starts", "deciding what to change"):
            assert phrase not in low, f"{name} still prescribes: {phrase!r}"


def test_every_rendering_prints_the_total_the_shares_are_over(scanned):
    """The Placed card and "N% of your placed records" share one denominator.

    A weighted scan writes an estimated histogram whose sum is not the raw
    placed count, and the shares used to divide by one while the card printed
    the other. The card now prints the total the shares sum to.
    """
    from dropoutt.report import html as html_report
    from dropoutt.report.summary import build

    result, fp = scanned
    story = build(result).atlas
    if story is None or not story.grid:
        pytest.skip("no atlas coverage in this environment")

    assert sum(area.records for area in story.grid) == story.placed
    assert sum(cell.records for area in story.grid for cell in area.cells) == story.placed
    assert sum(area.share for area in story.grid) == pytest.approx(1.0)
    for place in (*story.places, *story.thin_places):
        assert place.share == pytest.approx(place.records / story.placed)

    atlas = _payload(scanned)["atlas"]
    assert atlas["placed_records"] == story.placed
    assert sum(a["records"] for a in atlas["subject_areas"]) == atlas["placed_records"]
    stated = _flat(f"{atlas['placed_label']} {atlas['placed_note']}")
    assert stated in _flat(md_report.render(result, fp, None))
    assert stated in _flat(md_report.render_atlas(result))
    assert f"{atlas['placed_label']} placed {atlas['placed_note']}" in _flat(_terminal(result, fp))
    for page in (html_report.render(result, fp, None), html_report.render_atlas(result)):
        assert atlas["placed_label"] in page
        assert _flat(atlas["placed_note"]) in _flat(page)


def test_the_terminal_prints_every_insight_the_markdown_does(scanned):
    """Five of eight, silently, was the parity failure this file exists to end."""
    result, fp = scanned
    atlas = _payload(scanned)["atlas"]
    if atlas is None or not atlas["available"] or not atlas["insights"]:
        pytest.skip("no atlas coverage in this environment")
    terminal = _flat(_terminal(result, fp))
    markdown = _flat(md_report.render(result, fp, None))
    for insight in atlas["insights"]:
        assert _flat(insight["headline"]) in terminal
        assert _flat(insight["headline"]) in markdown


def test_the_page_uses_the_markdown_files_vocabulary(scanned):
    """Subject areas, subregions, the map — and 4,096 with its separator.

    The page said topics, subtopics and atlas where the text file and the docs
    say subject areas, subregions and map, and defined reach as a count of
    subregions at map density when it is a sum of min(1, density).
    """
    from markupsafe import escape as html_escape

    from dropoutt.report import html as html_report
    from dropoutt.report.atlas_story import DENSITY_DEFINITION, REACH_DEFINITION

    result, _fp = scanned
    if result.ctx.atlas is None:
        pytest.skip("no atlas in this environment")
    page = html_report.render_atlas(result)
    markdown = md_report.render_atlas(result)

    assert "subtopic" not in page.lower()
    assert ">Topics<" not in page and "Subject areas" in page
    assert "at atlas density" not in page and "the atlas to place" not in page
    assert str(html_escape(REACH_DEFINITION)) in _flat(page)
    assert str(html_escape(DENSITY_DEFINITION)) in _flat(page)
    assert REACH_DEFINITION in _flat(markdown)

    cells = int(result.ctx.atlas.n_regions)
    for out in (page, markdown):
        assert f"{cells:,}" in out
        if cells >= 1000:
            assert str(cells) not in out
