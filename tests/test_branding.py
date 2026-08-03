"""The banner must never be the reason a run fails or a log is unreadable."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dropoutt.branding import (
    LOGO_ASCII,
    LOGO_BLOCKS,
    MIN_WIDTH,
    TAGLINE,
    banner,
    logo,
    mark,
    supports_unicode,
    wants_decoration,
)
from dropoutt.cli import app

runner = CliRunner()


class _Stream:
    """A stand-in for stdout. StringIO's `encoding` is read-only, so this is a
    plain object with the two attributes the branding module consults."""

    def __init__(self, encoding: str, tty: bool = True) -> None:
        self.encoding = encoding
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_the_ascii_fallback_is_pure_seven_bit():
    """cmd.exe on a legacy code page cannot encode a block character, and
    Python raises rather than substituting — so a decorative banner would take
    the whole program down."""
    LOGO_ASCII.encode("ascii")
    assert all(ord(ch) < 128 for ch in LOGO_ASCII)


def test_block_art_is_used_only_when_the_stream_can_encode_it():
    assert supports_unicode(_Stream("utf-8"))
    assert not supports_unicode(_Stream("cp437"))
    assert not supports_unicode(_Stream("cp1252"))
    assert not supports_unicode(_Stream("ascii"))
    # A stream whose encoding is unknown is treated as one that cannot.
    assert not supports_unicode(_Stream(""))


def test_both_renderings_have_the_same_shape():
    """They are two pictures of one mark. If one is edited alone they drift."""
    blocks, ascii_art = LOGO_BLOCKS.split("\n"), LOGO_ASCII.split("\n")
    assert len(blocks) == len(ascii_art)
    assert (max(len(row) for row in blocks)
            == max(len(row) for row in ascii_art))


@pytest.mark.parametrize("width", [200, 120, 100, 80, 72, 64, 50, 40, 20, 1])
def test_the_banner_never_exceeds_the_terminal_width(width):
    """Wrapping the art breaks the picture, so it is dropped instead."""
    import re

    plain = re.sub(r"\[/?[a-z ]+\]", "", banner("1.0.0", width=width))
    for line in plain.split("\n"):
        assert len(line) <= width or width < MIN_WIDTH


def test_a_narrow_terminal_gets_words_rather_than_art():
    out = banner("1.0.0", width=MIN_WIDTH - 1)
    assert "█" not in out and "@" not in out
    assert "dropoutt" in out and TAGLINE in out


def test_redirected_output_is_not_decorated():
    """Piping the tool into a file must not put art in the file."""
    assert not wants_decoration(_Stream("utf-8", tty=False))


def test_no_color_is_honoured(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert not wants_decoration(_Stream("utf-8", tty=True))
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert not wants_decoration(_Stream("utf-8", tty=True))


def test_bare_invocation_prints_help_and_a_starting_point():
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "dropoutt scan ./data" in result.output


def test_the_mark_has_an_ascii_form_too():
    assert logo(unicode_ok=False) == LOGO_ASCII.split("\n")
    assert all(ord(ch) < 128 for ch in mark(unicode_ok=False))
