"""CLI usage errors must tell the user what input is expected."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from dropoutt.cli import _without_source_locations, app

runner = CliRunner()


def _encoder_is_cached() -> bool:
    """Whether `dropoutt atlas` can run without reaching the network.

    The atlas artifact ships in the package; its encoder does not, and no test
    in this suite is allowed to download one. The two tests below drive the real
    command end to end, so they run when a `dropoutt fetch` has already happened
    on this machine and skip when it has not, rather than turning a clean
    checkout's test run into an 81 MB download.
    """
    from dropoutt.atlas import DEFAULT_MODEL, embed
    from dropoutt.config import cache_dir

    return embed.local_model_dir(DEFAULT_MODEL, cache_dir()).exists()


needs_encoder = pytest.mark.skipif(
    not _encoder_is_cached(), reason="atlas encoder is not in the cache; run dropoutt fetch"
)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Output with styling removed.

    rich treats GitHub Actions as a colour-capable terminal and forces escape
    sequences on, which no ``NO_COLOR`` or ``TERM`` setting turns back off. The
    same help text therefore arrives styled in CI and bare on a developer's
    machine. Every assertion below is about the words, so the styling goes.
    """
    return _ANSI.sub("", text)


def words(text: str) -> str:
    """Styling removed and whitespace collapsed, for text rich may re-wrap."""
    return " ".join(plain(text).split())


def test_commands_with_required_arguments_show_full_help_when_empty():
    for command, example in (
        ("scan", "dropoutt scan ./data --offline"),
    ):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2
        assert "Usage:" in plain(result.output)
        assert example in words(result.output)
        assert "Traceback" not in result.output


def test_the_published_command_surface_is_exactly_what_is_supported():
    """The 1.0 surface is a promise, so it is asserted rather than assumed.

    `diff`, `index-eval` and `init` were dropped before the first stable
    release: each was useful and none was finished enough to freeze. `doctor`
    went in 1.1, with the extras it existed to help you choose between. `atlas`
    was dropped with them and came back in 1.3 — deliberately, because the map
    turned out to be a second question with a second cost rather than a section
    of the scan report. Adding one is that kind of decision, not something a
    stray decorator does.
    """
    result = runner.invoke(app, ["--help"])
    listed = {
        line.split()[1]
        for line in plain(result.output).splitlines()
        if line.startswith("\u2502 ") and len(line.split()) > 1
    }
    assert {"scan", "atlas", "checks", "benchmarks", "models", "fetch",
            "help", "version"} <= listed
    for gone in ("diff", "index-eval", "init", "doctor"):
        assert gone not in listed
        assert runner.invoke(app, [gone]).exit_code == 2


def test_help_and_version_work_as_words_as_well_as_flags():
    """Guessing wrong about a tool's conventions should not be a usage error."""
    from dropoutt import __version__

    assert plain(runner.invoke(app, ["version"]).output).strip() == __version__
    assert plain(runner.invoke(app, ["--version"]).output).strip() == __version__

    spelled_out = runner.invoke(app, ["help"])
    assert spelled_out.exit_code == 0
    assert "Usage:" in plain(spelled_out.output)

    for command in ("scan", "checks", "fetch"):
        one = runner.invoke(app, ["help", command])
        assert one.exit_code == 0
        assert f"Usage: dropoutt {command}" in words(one.output)

    unknown = runner.invoke(app, ["help", "wat"])
    assert unknown.exit_code == 2
    assert "No such command" in plain(unknown.output)
    assert "scan" in plain(unknown.output)


def test_a_scan_does_not_try_to_open_a_report_that_nobody_can_see(tmp_path, monkeypatch):
    """CliRunner output is a pipe, which is one of the reasons not to open."""
    from dropoutt import desktop

    (tmp_path / "data.jsonl").write_text('{"text": "a record"}\n', encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url: opened.append(url) or True)

    result = runner.invoke(app, ["scan", str(tmp_path), "--offline"])

    assert result.exit_code == 0
    assert (tmp_path / ".dropoutt" / "report.html").exists()
    assert opened == []
    assert "opened report.html" not in result.output


def test_no_evidence_removes_excerpts_from_written_outputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    planted = "proprietary text outside the PII pattern catalog"
    with open(data / "train.jsonl", "w", encoding="utf-8") as fh:
        for _ in range(8):
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Soru"},
                    {"role": "assistant", "content": planted},
                ],
            }) + "\n")
    out = tmp_path / "out"

    result = runner.invoke(app, [
        "scan", str(data),
        "--offline", "--no-evidence", "--quiet",
        "--out", str(out),
    ])

    assert result.exit_code == 0
    assert "record excerpts and source locations were omitted" in result.output
    findings = (out / "findings.jsonl").read_text(encoding="utf-8")
    report = (out / "report.html").read_text(encoding="utf-8")
    assert planted not in findings
    assert planted not in report
    assert all(not json.loads(line)["evidence"] for line in findings.splitlines())


def test_effective_cli_configuration_changes_the_fingerprint_id(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text(
        json.dumps({"text": "same corpus content"}) + "\n", encoding="utf-8"
    )

    ids = []
    for seq_len in ("64", "128"):
        out = tmp_path / f"out-{seq_len}"
        result = runner.invoke(app, [
            "scan", str(data),
            "--seq-len", seq_len,
            "--offline", "--no-html", "--quiet",
            "--out", str(out),
        ])
        assert result.exit_code == 0
        ids.append(json.loads(
            (out / "fingerprint.json").read_text(encoding="utf-8")
        )["fingerprint_id"])

    assert ids[0] != ids[1]


def test_malformed_config_is_a_usage_error_without_a_traceback(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"text": "record"}\n', encoding="utf-8")
    (tmp_path / "dropoutt.toml").write_text("[scan\nprofile = 3", encoding="utf-8")

    result = runner.invoke(app, [
        "scan", str(tmp_path), "--offline", "--no-html", "--quiet",
    ])

    assert result.exit_code == 2
    assert "Invalid dropoutt.toml" in result.output
    assert "Traceback" not in result.output


def test_unknown_configured_eval_set_is_a_usage_error(tmp_path):
    (tmp_path / "data.jsonl").write_text('{"text": "record"}\n', encoding="utf-8")
    (tmp_path / "dropoutt.toml").write_text(
        '[scan]\neval_sets = ["not-an-installed-index"]\n', encoding="utf-8"
    )

    result = runner.invoke(app, [
        "scan", str(tmp_path), "--offline", "--no-html", "--quiet",
    ])

    assert result.exit_code == 2
    assert "Unknown eval_sets" in result.output


def test_scan_rejects_a_directory_without_supported_data(tmp_path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF")

    result = runner.invoke(app, ["scan", str(tmp_path), "--offline"])

    assert result.exit_code == 2
    assert "No supported data files found" in result.output
    assert "JSONL/NDJSON" in result.output
    assert not (tmp_path / ".dropoutt").exists()


def test_scan_reports_active_phases_in_redirected_output(tmp_path):
    (tmp_path / "data.jsonl").write_text(
        '{"text": "a record long enough to scan normally"}\n', encoding="utf-8"
    )

    result = runner.invoke(app, [
        "scan", str(tmp_path), "--offline", "--no-html", "--limit", "1",
    ])

    assert result.exit_code == 0
    assert "Discovering supported data files..." in result.output
    assert "Inferring dataset layouts..." in result.output
    assert "Scanning records..." in result.output
    assert "Writing scan artifacts..." in result.output
    assert "done Scanned 1 records" in " ".join(result.output.split())


def test_evidence_free_structured_data_drops_nested_witness_paths():
    cleaned = _without_source_locations({
        "results": {
            "private-eval": {
                "witnesses": [
                    {"instance": 4, "record": ("doc-id", "/secret/data.jsonl", 12)}
                ],
            },
        },
    })

    assert cleaned == {
        "results": {"private-eval": {"witnesses": [{"instance": 4}]}}
    }


def test_declared_version_has_exactly_one_source():
    """`pyproject.toml` reads the version from the package, so they cannot drift.

    They did drift once: 0.1.4 was tagged in pyproject while `dropoutt doctor`
    still reported 0.1.3 from the constant, because both carried a literal.
    """
    import re
    from pathlib import Path

    import dropoutt

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in pyproject
    assert re.search(r'^\[tool\.hatch\.version\]\npath = "src/dropoutt/__init__\.py"',
                     pyproject, re.M)
    assert not re.search(r'^version = "', pyproject, re.M), (
        "a literal version in pyproject would shadow the package constant again"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", dropoutt.__version__)


def test_fetch_names_the_interpreter_and_the_machine_it_probed():
    """Every import status is an import against one specific Python.

    Without the path, installing with a `pip` from a different environment looks
    like the tool ignoring the install. uv venvs ship no pip, so this is the
    default way to hit it. This lived in `dropoutt doctor` until extras were
    removed and the capability table stopped being a decision; what was worth
    keeping moved to `fetch`, the other command whose job is an environment.

    The machine line is here for the same reason: a scan sizes its worker pool
    and its sample against this specific process's cores and memory, and inside
    a container those are not the host's.
    """
    import io
    import sys
    from pathlib import Path

    from rich.console import Console

    from dropoutt import __version__, cli

    buf = io.StringIO()
    # Written straight to a captured console rather than through `fetch`, which
    # would download a tokenizer to reach this block.
    cli.console = Console(file=buf, width=200, no_color=True)
    try:
        cli._print_environment()
    finally:
        cli.console = Console()
    flat = buf.getvalue().replace("\n", "")

    # rich wraps long paths, so compare on the basename rather than the whole path.
    assert Path(sys.executable).name in flat
    assert __version__ in flat
    assert "core" in flat and "machine" in flat


def test_scan_no_longer_draws_the_map_and_says_which_command_does(tmp_path):
    """The split is the point of 1.3, so both halves of it are asserted.

    A scan that quietly stopped reporting coverage would look identical to a
    scan whose atlas failed to load. The fingerprint keeps the facet either way
    — two fingerprints must have the same shape to be comparable — and the
    skipped-check line has to name the command that fills it rather than
    suggesting a reinstall.
    """
    (tmp_path / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"A record with enough prose in it to be worth "
                                f"placing on a topical map, number {i}."})
            for i in range(20)
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["scan", str(tmp_path), "--offline", "--no-html"])
    assert result.exit_code == 0

    fp = json.loads((tmp_path / ".dropoutt" / "fingerprint.json").read_text())
    assert fp["facets"]["coverage"]["values"] == {
        "status": "not computed by scan (run dropoutt atlas)"
    }
    assert "dropoutt atlas" in words(result.output)
    assert not (tmp_path / ".dropoutt" / "atlas.html").exists()


def test_scan_rejects_the_flag_that_used_to_turn_the_map_off(tmp_path):
    """`--no-atlas` is gone rather than accepted and ignored.

    A flag that still parses but no longer does anything is worse than one that
    errors: a CI job passing it would go on believing it had switched something
    off. There is nothing left to switch off.
    """
    (tmp_path / "data.jsonl").write_text('{"text": "hello there"}\n', encoding="utf-8")
    result = runner.invoke(app, ["scan", str(tmp_path), "--no-atlas"])
    assert result.exit_code == 2


@needs_encoder
def test_atlas_places_records_and_writes_the_map_in_every_shape(tmp_path):
    """`dropoutt atlas` is a complete command, not a flag in disguise."""
    (tmp_path / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"The garden strawberry is a widely grown hybrid "
                                f"plant cultivated worldwide for its fruit, {i}."})
            for i in range(20)
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["atlas", str(tmp_path), "--no-open", "--offline"])
    assert result.exit_code == 0, plain(result.output)

    out = tmp_path / ".dropoutt"
    assert {"atlas.html", "atlas.md", "atlas.json"} <= {p.name for p in out.iterdir()}
    # A scan's artifacts are not overwritten by a map, and vice versa.
    assert not (out / "report.html").exists()

    data = json.loads((out / "atlas.json").read_text())
    assert data["schema"] == "dropoutt.atlas/1"
    assert data["atlas"]["available"] is True
    assert data["atlas"]["placed_records"] > 0
    # No findings table: the map answers where, and `scan` answers what is wrong.
    assert all(f["check_id"].startswith("T1-ATLAS-") for f in data["findings"])
    assert "Where this corpus sits" in words(result.output)


@needs_encoder
def test_atlas_omits_evidence_everywhere_at_once(tmp_path):
    """`--no-evidence` is a promise about every file the command writes."""
    secret = "SPECIMEN-TOKEN-9d4f1c"
    (tmp_path / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"{secret} appears in a record with enough other "
                                f"prose around it to be placed, number {i}."})
            for i in range(20)
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["atlas", str(tmp_path), "--no-open", "--offline",
              "--no-evidence", "--quiet"]
    )
    assert result.exit_code == 0, plain(result.output)
    out = tmp_path / ".dropoutt"
    for name in ("atlas.html", "atlas.md", "atlas.json"):
        assert secret not in (out / name).read_text(), name
    assert secret not in plain(result.output)
