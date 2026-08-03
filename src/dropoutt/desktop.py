"""Is there a screen in front of this process, and can we put a file on it?

A scan ends with an HTML file and the next thing anyone does is open it, so
`dropoutt scan` opens it. That is right on a laptop and wrong nearly everywhere
else a scan runs. Over SSH the window would appear on the machine doing the
work rather than the one being looked at. In CI and in batch jobs there is no
session to open into. In a pipe the caller asked for text.

So every reason not to is checked before the default applies, and the caller
gets the reason back so it can say what happened instead of failing silently.
Set ``DROPOUTT_OPEN=0`` to never open, or ``DROPOUTT_OPEN=1`` to open anyway --
the second is for X11 forwarding, where the SSH check is wrong.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path

#: The shell is on the far end of a network connection. OpenSSH sets these for
#: interactive sessions; the browser would open on the wrong machine.
SSH_VARS = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")

#: Hosted runners. Most set CI on their own; the rest are named individually.
CI_VARS = (
    "CI", "CONTINUOUS_INTEGRATION", "BUILD_ID", "GITHUB_ACTIONS", "GITLAB_CI",
    "JENKINS_URL", "TEAMCITY_VERSION", "BUILDKITE", "CIRCLECI", "TF_BUILD",
)

#: Cluster schedulers. A compute node has no session even when the login node
#: it was submitted from did.
BATCH_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")

#: Platforms where a windowing system is always present if anyone is logged in.
ALWAYS_GRAPHICAL = ("darwin", "win32", "cygwin")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def blocked(env: Mapping[str, str] | None = None, stream=None) -> str | None:
    """Why the report should not be opened, or None if it should be.

    The string is written to be printed: it finishes the sentence "not opening
    the report: ...".
    """
    # Read-only, and os.environ is not a dict. Mapping is what both it and a
    # test's plain dict actually satisfy.
    environ: Mapping[str, str] = os.environ if env is None else env
    if _falsey(environ.get("DROPOUTT_OPEN")):
        return "DROPOUTT_OPEN is off"
    if _truthy(environ.get("DROPOUTT_OPEN")):
        return None

    if any(environ.get(name) for name in SSH_VARS):
        return "this looks like an SSH session"
    if any(environ.get(name) for name in CI_VARS):
        return "this looks like CI"
    if any(environ.get(name) for name in BATCH_VARS):
        return "this looks like a batch job"

    stream = sys.stdout if stream is None else stream
    try:
        interactive = bool(stream.isatty())
    except Exception:
        interactive = False
    if not interactive:
        return "output is not a terminal"

    if sys.platform not in ALWAYS_GRAPHICAL and not _has_display(environ):
        return "no display is available"
    return None


def _has_display(env: Mapping[str, str]) -> bool:
    """Whether X11, Wayland, or the WSL bridge can show a window."""
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return True
    return bool(env.get("WSL_DISTRO_NAME")) and shutil.which("wslview") is not None


def open_report(
    path: str | Path, env: Mapping[str, str] | None = None, stream=None
) -> str | None:
    """Show a written report to whoever ran the scan.

    Returns None once it has been handed to a browser, or the reason it was
    not. Never raises: this runs after the work is finished and after the
    findings have been printed, and a desktop that cannot open a file is not a
    reason to fail a scan that succeeded.
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    reason = blocked(environ, stream)
    if reason:
        return reason

    url = Path(path).resolve().as_uri()
    try:
        # WSL has no browser of its own. wslu's wslview hands the URL to
        # Windows, and Python's webbrowser does not know about it.
        if environ.get("WSL_DISTRO_NAME") and not environ.get("DISPLAY"):
            wslview = shutil.which("wslview")
            if wslview:
                subprocess.Popen(
                    [wslview, url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return None
        if webbrowser.open(url):
            return None
    except Exception:
        return "no browser could be started"
    return "no browser is registered"
