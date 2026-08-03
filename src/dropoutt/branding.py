"""The mark, and the rules for when a terminal can actually show it.

The art is the product's own icon, rasterised from ``icon-logo.svg`` at 24x24
and folded over its four-fold symmetry group so the character grid is as clean
as the vector. Two renderings of the same shape: half-block characters for
terminals that can draw them, and a 7-bit ramp for the ones that cannot.

Three ways a banner goes wrong on someone else's machine, all handled here.

**Encoding.** ``cmd.exe`` on a legacy code page cannot encode U+2588, and Python
raises rather than substituting, so a decorative banner would crash the program.
The block art is used only after encoding it against the real output stream.

**Width.** A 24-column mark in an 80-column terminal is fine and in a 40-column
one is wreckage. Narrow terminals get the wordmark alone.

**Consent.** ``NO_COLOR``, ``TERM=dumb``, and a redirected stdout all mean the
same thing: this output is being read by something that did not ask for
decoration. Piping ``dropoutt`` into a file should not put art in the file.
"""

from __future__ import annotations

import os
import sys

#: Half-block rendering, 24 columns by 12 rows.
LOGO_BLOCKS = """\
  ▄▄█████▄    ▄█████▄▄
 ███▀▀▀▀▀██▄▄██▀▀▀▀▀███
███       ████       ███
███     ▄██▀▀██▄     ███
▀██▄  ▄██▀    ▀██▄  ▄██▀
  ▀████▀  ▄██▄  ▀████▀
  ▄████▄  ▀██▀  ▄████▄
▄██▀  ▀██▄    ▄██▀  ▀██▄
███     ▀██▄▄██▀     ███
███       ████       ███
 ███▄▄▄▄▄██▀▀██▄▄▄▄▄███
  ▀▀█████▀    ▀█████▀▀"""

#: The same shape in ASCII, for terminals that cannot encode the blocks.
LOGO_ASCII = """\
  :%@@@@%#.  .#%@@@@%:
.@@%:..:#@%##%@#:..:%@@.
@@#      .@@@@.      #@@
@@#    .#@@##@@#.    #@@
:@@# .#@@#.  .#@@#. #@@:
 .#%@@@#. :%%: .#@@@%#.
 .#%@@@#. :%%: .#@@@%#.
:@@# .#@@#.  .#@@#. #@@:
@@#    .#@@##@@#.    #@@
@@#      .@@@@.      #@@
.@@%:..:#@%##%@#:..:%@@.
  :%@@@@%#.  .#%@@@@%:"""

#: A one-line mark for headers, where twelve rows would be an intrusion.
MARK_BLOCKS = "◧◨"
MARK_ASCII = "::"

TAGLINE = "Pre-flight checks for LLM training data."

#: Below this many columns the art is dropped rather than wrapped.
MIN_WIDTH = 46


def supports_unicode(stream=None) -> bool:
    """Whether the output stream can actually encode the block characters.

    Encoding a sample rather than trusting the encoding's name: ``cp1252`` and
    ``cp437`` both claim to be encodings and neither has U+2588, and the failure
    mode of guessing wrong is a traceback instead of a banner.
    """
    stream = stream or sys.stdout
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "█▀▄◧◨".encode(encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def wants_decoration(stream=None) -> bool:
    """Whether this output is being read by a person at a terminal."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def logo(*, unicode_ok: bool | None = None) -> list[str]:
    if unicode_ok is None:
        unicode_ok = supports_unicode()
    return (LOGO_BLOCKS if unicode_ok else LOGO_ASCII).split("\n")


def mark(*, unicode_ok: bool | None = None) -> str:
    if unicode_ok is None:
        unicode_ok = supports_unicode()
    return MARK_BLOCKS if unicode_ok else MARK_ASCII


def banner(version: str, *, width: int = 80, unicode_ok: bool | None = None,
           colour: bool = True) -> str:
    """The mark beside the wordmark, sized to the terminal.

    Returns rich markup. The caller decides whether to print it at all; this
    only decides what it looks like once that is settled.
    """
    accent = "[bold cyan]" if colour else "[bold]"
    close = "[/bold cyan]" if colour else "[/bold]"
    dim_open, dim_close = ("[dim]", "[/dim]") if colour else ("", "")

    title = f"{accent}dropoutt{close}  {dim_open}{version}{dim_close}"
    if width < MIN_WIDTH:
        return f"{title}\n{dim_open}{TAGLINE}{dim_close}"

    art = logo(unicode_ok=unicode_ok)
    art_width = max(len(line) for line in art)
    gap = "   "
    room = width - art_width - len(gap) - 1

    # The tagline is one line or the layout is wrong: it sits on a fixed row
    # beside the art, so wrapping it pushes every row below it out of register.
    if room < len(TAGLINE):
        rows = [f"{accent}{line}{close}" for line in art]
        rows += ["", title, f"{dim_open}{TAGLINE}{dim_close}"]
        return "\n".join(rows)

    body = ("Point it at a folder of training data. It tells you what is wrong "
            "before you spend a training run finding out.")
    side = ["", "", title, f"{dim_open}{TAGLINE}{dim_close}", ""]
    side += [f"{dim_open}{line}{dim_close}" for line in _wrap(body, room)]

    rows = []
    for index, line in enumerate(art):
        text = side[index] if index < len(side) else ""
        rows.append(f"{accent}{line.ljust(art_width)}{close}{gap}{text}".rstrip())
    return "\n".join(rows)


def _wrap(body: str, width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for word in body.split():
        probe = f"{line} {word}" if line else word
        if len(probe) <= width or not line:
            line = probe
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines
