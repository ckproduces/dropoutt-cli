"""How a number becomes a phrase.

Every report says the same things about the same scan, so they have to say them
the same way. These live apart from the modules that use them because both the
summary and the atlas story format shares, and a second copy of ``share()`` is
how "38%" and "0.38" end up on the same page.
"""

from __future__ import annotations


def share(value: float) -> str:
    """A fraction as a percentage, at a precision the reader can act on.

    Under a tenth of a percent is reported as a bound rather than a number:
    "0.03%" invites arithmetic that the sample size does not support.
    """
    if value <= 0:
        return "0%"
    if value < 0.001:
        return "under 0.1%"
    if value < 0.01:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.0f}%"


def plural(noun: str) -> str:
    """Pluralise a check's unit. Handles "record", "dataset", "pair of records"."""
    if noun.endswith(("s", "x", "ch")):
        return noun + "es"
    head, sep, tail = noun.partition(" of ")
    if sep:
        return f"{head}s of {tail}"
    return noun + "s"


def count(value: int) -> str:
    """A large count, short enough to sit in a headline."""
    if value >= 1_000_000_000:
        return f"{value / 1e9:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1e6:.1f}M"
    if value >= 1_000:
        return f"{value / 1e3:.0f}k"
    return f"{value:,}"


def in_words(value: float) -> str:
    """A share as a ratio a person reads without arithmetic.

    "One record in three" lands; "31.4%" has to be converted before it means
    anything, and the conversion is what the reader was going to do anyway.
    """
    if value >= 0.9:
        return "nearly every record"
    if value >= 0.66:
        return "two records in three"
    if value >= 0.45:
        return "about half your records"
    if value >= 0.28:
        return "one record in three"
    if value >= 0.18:
        return "one record in five"
    if value >= 0.08:
        return "one record in ten"
    if value >= 0.02:
        return "a few records in a hundred"
    return "a small number of records"
