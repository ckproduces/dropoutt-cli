"""Cheap necessary-condition gates derived from a regex.

Three checks run a list of patterns against every record, and that was the
single most expensive thing in the scan after language identification. The
reason is narrow and measurable: a case-sensitive literal scan runs at about
0.4 ns per character because CPython's engine can use a memchr-style skip, and
the same pattern under ``re.IGNORECASE`` runs at 4-6 ns per character because it
cannot. Sixteen such patterns over an 800-character record is 118 ns per
character of pure scanning.

The fix is a gate: a set of substrings at least one of which must be present for
the pattern to match at all. Testing one against a record costs about 0.36 ns
per character, and a record that fails the gate skips its pattern entirely.

The gates are **derived from the patterns**, never written by hand beside them,
because a hand-written gate that drifts from its pattern turns a check off
silently and nothing fails. The extractor below walks the parsed pattern and is
deliberately incomplete: any construct it does not understand yields no gate,
and no gate means the pattern always runs. Being conservative costs speed; being
wrong costs a check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

try:  # pragma: no cover - module moved in 3.11
    from re import _parser as _sre_parse  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python 3.10
    import sre_parse as _sre_parse  # type: ignore[no-redef]

#: Minimum selectivity a set must reach to be worth testing. Characters that
#: appear in ordinary prose count for one and everything else counts for three,
#: so a single ``@`` is a better gate than a two-letter run and a lone ``.`` is
#: no gate at all.
_MIN_SCORE = 3
_COMMON = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-'"
)
#: A second set only earns its own scan if it is clearly selective on its own.
#: The first set has already rejected most records by then.
_SECOND_SET_SCORE = 6

#: Any inline flag group that is not a leading global one. Scoped flags such as
#: ``(?i:...)`` change case sensitivity part-way through, and the walker below
#: tracks no per-node flag state, so patterns containing them get no gate.
_SCOPED_FLAGS = re.compile(r"\(\?[aiLmsux]*[-:]")


@dataclass(frozen=True, slots=True)
class PatternGate:
    """Necessary conditions for one pattern.

    ``sets`` are ANDed and their members ORed: every set must contribute at
    least one present substring. ``lowered`` says which copy of the record to
    test against, so a case-sensitive pattern is never gated on lowercase text.
    """

    sets: tuple[tuple[str, ...], ...] = ()
    lowered: bool = False

    def __bool__(self) -> bool:
        return bool(self.sets)

    def passes(self, text: str, lowered_text: str) -> bool:
        """Whether the record could possibly contain a match.

        Never False for text that does match: every set came from a literal the
        pattern requires. Often True for text that does not, which costs only
        the pattern being run.
        """
        haystack = lowered_text if self.lowered else text
        for alternatives in self.sets:
            for token in alternatives:
                if token in haystack:
                    break
            else:
                return False
        return True


_NO_GATE = PatternGate()


def _literal_runs(seq, ignorecase: bool) -> list[tuple[str, ...]]:
    """Alternative-sets that any match of ``seq`` must contain.

    A one-member set is a plain mandatory substring; a several-member set is a
    branch where every arm contributed its own mandatory substring.
    """
    out: list[tuple[str, ...]] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(("".join(run),))
        run.clear()

    for op, av in seq:
        name = str(op)
        if name == "LITERAL":
            ch = chr(av)
            run.append(ch.lower() if ignorecase else ch)
            continue
        flush()
        if name == "AT":
            continue  # zero-width assertion, contributes nothing
        if name == "SUBPATTERN":
            out.extend(_literal_runs(av[3], ignorecase))
        elif name in ("MAX_REPEAT", "MIN_REPEAT") and av[0] >= 1:
            # The body appears at least once, so its literals are mandatory too.
            out.extend(_literal_runs(av[2], ignorecase))
        elif name == "BRANCH":
            arms: list[str] = []
            for arm in av[1]:
                plains = [c[0] for c in _literal_runs(arm, ignorecase) if len(c) == 1]
                if not plains:
                    arms = []
                    break
                # The strongest run this arm offers, not merely the first: an
                # arm like ``G[uü]zel`` yields both "g" and "zel", and gating a
                # branch on its weakest arm makes the whole set worthless.
                arms.append(max(plains, key=lambda run: _selectivity((run,))))
            if arms:
                out.append(tuple(arms))
        # IN, ANY, GROUPREF, ASSERT, NEGATE, repeats that may match zero times,
        # and anything a future Python adds: no contribution. The run is already
        # flushed, so nothing spans the construct.
    flush()
    return out


def _selectivity(alternatives: tuple[str, ...]) -> int:
    """How much a set narrows the input, judged by its weakest member."""
    return min(
        sum(1 if ch in _COMMON else 3 for ch in alt) for alt in alternatives
    )


@lru_cache(maxsize=512)
def gate_for(pattern: str) -> PatternGate:
    """Derive a gate from a pattern source, or an empty gate if none is safe."""
    head = re.match(r"^\(\?([aiLmsux]+)\)", pattern)
    body = pattern[head.end():] if head else pattern
    if _SCOPED_FLAGS.search(body.replace("(?:", "")):
        return _NO_GATE
    ignorecase = bool(head and "i" in head.group(1))
    try:
        parsed = _sre_parse.parse(pattern, re.IGNORECASE if ignorecase else 0)
    except Exception:  # pragma: no cover - defensive
        return _NO_GATE
    sets = [
        alts for alts in _literal_runs(parsed, ignorecase)
        if _selectivity(alts) >= _MIN_SCORE
    ]
    if not sets:
        return _NO_GATE
    sets.sort(key=_selectivity, reverse=True)
    kept = sets[:1]
    if len(sets) > 1 and _selectivity(sets[1]) >= _SECOND_SET_SCORE:
        kept.append(sets[1])
    return PatternGate(tuple(kept), ignorecase)
