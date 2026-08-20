"""CLI usage errors must tell the user what input is expected."""

from __future__ import annotations

import json
import re

from typer.testing import CliRunner

from dropoutt.cli import _without_source_locations, app

runner = CliRunner()

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

    `diff`, `index-eval`, `init` and `atlas` were dropped before the first
    stable release: each was useful and none was finished enough to freeze.
    `doctor` went in 1.1, with the extras it existed to help you choose between.
    Re-adding one is a deliberate act, not something a stray decorator does.
    """
    result = runner.invoke(app, ["--help"])
    listed = {
        line.split()[1]
        for line in plain(result.output).splitlines()
        if line.startswith("\u2502 ") and len(line.split()) > 1
    }
    assert {"scan", "checks", "benchmarks", "models", "fetch",
            "help", "version"} <= listed
    for gone in ("diff", "index-eval", "init", "atlas", "doctor"):
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

    result = runner.invoke(app, ["scan", str(tmp_path), "--offline", "--no-atlas"])

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
        "--offline", "--no-atlas", "--no-evidence", "--quiet",
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
            "--offline", "--no-atlas", "--no-html", "--quiet",
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
        "scan", str(tmp_path), "--offline", "--no-atlas", "--no-html", "--quiet",
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
        "scan", str(tmp_path), "--offline", "--no-atlas", "--no-html", "--quiet",
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
        "scan", str(tmp_path), "--offline", "--no-atlas", "--no-html", "--limit", "1",
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
