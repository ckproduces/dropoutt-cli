"""Deciding whether there is a screen to put the report on.

The failure that matters is opening a browser on a machine nobody is looking
at: over SSH the window lands on the cluster, not the laptop. Every case below
is a way that happens, so each gets its own assertion rather than one combined
one.
"""

from __future__ import annotations

import sys

import pytest

from dropoutt import desktop


class _Terminal:
    """A stdout that claims to be a terminal. StringIO does not."""

    def isatty(self) -> bool:
        return True


class _Pipe:
    def isatty(self) -> bool:
        return False


DESKTOP = {"DISPLAY": ":0"}


def test_a_terminal_on_a_desktop_opens_the_report():
    assert desktop.blocked(DESKTOP, _Terminal()) is None


@pytest.mark.parametrize("var", desktop.SSH_VARS)
def test_an_ssh_session_does_not(var):
    env = {**DESKTOP, var: "10.0.0.2 51234 10.0.0.1 22"}
    assert desktop.blocked(env, _Terminal()) == "this looks like an SSH session"


@pytest.mark.parametrize("var", ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE"])
def test_ci_does_not(var):
    assert desktop.blocked({**DESKTOP, var: "true"}, _Terminal()) == "this looks like CI"


@pytest.mark.parametrize("var", desktop.BATCH_VARS)
def test_a_batch_job_does_not(var):
    env = {**DESKTOP, var: "8811"}
    assert desktop.blocked(env, _Terminal()) == "this looks like a batch job"


def test_redirected_output_does_not():
    assert desktop.blocked(DESKTOP, _Pipe()) == "output is not a terminal"


@pytest.mark.skipif(sys.platform in desktop.ALWAYS_GRAPHICAL,
                    reason="macOS and Windows always have a window server")
def test_a_headless_unix_box_does_not():
    assert desktop.blocked({}, _Terminal()) == "no display is available"


def test_the_environment_can_force_it_off_where_it_would_have_opened():
    env = {**DESKTOP, "DROPOUTT_OPEN": "0"}
    assert desktop.blocked(env, _Terminal()) == "DROPOUTT_OPEN is off"


def test_the_environment_can_force_it_on_over_ssh():
    """X11 forwarding is the case the SSH check gets wrong, so it has an override."""
    env = {**DESKTOP, "SSH_CONNECTION": "x", "DROPOUTT_OPEN": "1"}
    assert desktop.blocked(env, _Terminal()) is None


def test_open_report_returns_the_reason_rather_than_raising(tmp_path):
    report = tmp_path / "report.html"
    report.write_text("<p>hi</p>", encoding="utf-8")
    env = {"CI": "true"}

    assert desktop.open_report(report, env, _Terminal()) == "this looks like CI"


def test_open_report_passes_a_file_uri_to_the_browser(tmp_path, monkeypatch):
    """A path is not a URL. Windows drive letters and spaces both need the URI form."""
    report = tmp_path / "sub dir" / "report.html"
    report.parent.mkdir()
    report.write_text("<p>hi</p>", encoding="utf-8")
    seen: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url: seen.append(url) or True)

    assert desktop.open_report(report, {**DESKTOP, "DROPOUTT_OPEN": "1"}, _Terminal()) is None
    assert seen == [report.resolve().as_uri()]
    assert seen[0].startswith("file://") and "%20" in seen[0]


def test_a_browser_that_will_not_start_is_reported_not_raised(tmp_path, monkeypatch):
    report = tmp_path / "report.html"
    report.write_text("<p>hi</p>", encoding="utf-8")

    def explode(url):
        raise OSError("no browser")

    monkeypatch.setattr(desktop.webbrowser, "open", explode)
    env = {**DESKTOP, "DROPOUTT_OPEN": "1"}

    assert desktop.open_report(report, env, _Terminal()) == "no browser could be started"
