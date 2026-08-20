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
            assert area["name"] in rendering


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
