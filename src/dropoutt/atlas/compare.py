"""Reading a coverage report, and comparing two of them.

The atlas exists so that two datasets scanned on different machines land in the
same bins and can be put side by side. That property is worth nothing until
something actually puts them side by side, which is what this module is for.

Two rules are enforced here rather than left to callers.

**Suppressed coverage stays suppressed.** If either side of a comparison had too
many off-atlas records, its region histogram describes records that did not
really land anywhere. Comparing against it would manufacture a precise-looking
answer out of two unreliable ones, so the comparison refuses instead.

**Nothing here ranks datasets.** `added_mass` says what share of A sits in
regions B does not reach. Whether that is good depends on what you are training,
which this module does not know. It reports the geometry and stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..registry_data import taxonomy


def category_names() -> dict[int, str]:
    """Level-0 category id to its short key."""
    return {int(c["id"]): str(c["key"]) for c in taxonomy()["categories"]}


def category_labels() -> dict[int, str]:
    """Level-0 category id to its human-readable label."""
    return {int(c["id"]): str(c["label"]) for c in taxonomy()["categories"]}


def region_mass(coverage: dict[str, Any]) -> dict[int, float]:
    """Region id to share of placed records, from a coverage facet.

    Prefers the full sparse histogram written under `region_counts`. Falls back
    to `top_regions`, which is a display head capped at twelve, so that
    fingerprints written before 0.1.2 still compare — with shares computed over
    the head rather than the whole corpus. `head_coverage()` says which happened.
    """
    counts = coverage.get("region_counts")
    if counts:
        total = sum(int(v) for v in counts.values())
        if total > 0:
            return {int(k): int(v) / total for k, v in counts.items()}

    tops = coverage.get("top_regions") or []
    total = sum(int(r.get("records", 0)) for r in tops)
    if total <= 0:
        return {}
    return {int(r["region"]): int(r["records"]) / total for r in tops}


def region_terms(coverage: dict[str, Any]) -> dict[int, str]:
    """Region id to its label words.

    Reads the bundled atlas first, so regions outside the stored top-twelve head
    still get a name. Falls back to whatever the fingerprint carried, which is
    what happens when the fingerprint was made against an atlas this machine
    does not have.
    """
    terms: dict[int, str] = {}
    from .apply import load_bundled  # noqa: PLC0415

    atlas = load_bundled()
    if atlas is not None and atlas.meta.get("version") == coverage.get("atlas_version"):
        terms.update(dict(enumerate(atlas.region_terms)))

    for r in coverage.get("top_regions") or []:
        if r.get("terms"):
            terms[int(r["region"])] = str(r["terms"])
    return terms


def category_mass(coverage: dict[str, Any]) -> dict[int, float]:
    """Level-0 category id to share of records. Keys arrive as strings."""
    raw = coverage.get("by_category") or {}
    total = sum(int(v) for v in raw.values())
    if total <= 0:
        return {}
    return {int(k): int(v) / total for k, v in raw.items()}


def concentration(coverage: dict[str, Any]) -> float | None:
    """How concentrated the corpus is, as a share of maximum spread.

    Region entropy divided by the entropy of a corpus spread evenly over every
    region. 1.0 means perfectly even, near 0 means everything sits in one place.
    Reported rather than judged: a specialised corpus *should* be concentrated,
    and a pretraining mixture should not.
    """
    ent = coverage.get("region_entropy")
    top = coverage.get("max_region_entropy")
    if ent is None or not top:
        return None
    return float(ent) / float(top)


def is_usable(coverage: dict[str, Any] | None) -> bool:
    """Whether a coverage facet carries a histogram worth comparing."""
    return bool(coverage) and coverage.get("status") == "ok"


def unusable_reason(coverage: dict[str, Any] | None) -> str:
    if not coverage:
        return "no atlas coverage was computed"
    status = coverage.get("status")
    if status == "suppressed":
        return str(coverage.get("reason") or "coverage was suppressed")
    if status == "no records":
        return "no records were placed on the atlas"
    return f"coverage status is {status!r}"


@dataclass
class Comparison:
    """Two coverage reports placed side by side."""

    comparable: bool
    reason: str = ""
    #: Share of A's placed mass sitting in regions B also occupies.
    shared_mass: float = 0.0
    #: Share of A's placed mass in regions B does not reach at all.
    added_mass: float = 0.0
    #: Cosine similarity of the two region-mass vectors, over their union.
    similarity: float = 0.0
    a_only: list[tuple[int, float, str]] = field(default_factory=list)
    b_only: list[tuple[int, float, str]] = field(default_factory=list)
    shared: list[tuple[int, float, float, str]] = field(default_factory=list)
    #: Share of each side's placed records described by the stored head.
    a_head_coverage: float = 0.0
    b_head_coverage: float = 0.0
    category_shift: list[tuple[int, str, float, float]] = field(default_factory=list)


def compare(a: dict[str, Any] | None, b: dict[str, Any] | None) -> Comparison:
    """Compare coverage A against coverage B, from A's point of view.

    Directional on purpose, exactly like cross-dataset overlap. "How much of A is
    somewhere B never goes" and "how much of B is somewhere A never goes" are
    different questions, and the one worth asking when deciding whether to add A
    to a mixture is the first.
    """
    if not is_usable(a):
        return Comparison(False, f"left side: {unusable_reason(a)}")
    if not is_usable(b):
        return Comparison(False, f"right side: {unusable_reason(b)}")
    if a.get("atlas_version") != b.get("atlas_version"):
        return Comparison(
            False,
            f"different atlas versions ({a.get('atlas_version')} against "
            f"{b.get('atlas_version')}); region ids do not mean the same thing",
        )

    ma, mb = region_mass(a), region_mass(b)
    if not ma or not mb:
        return Comparison(False, "no region histogram was stored on one side")

    terms = {**region_terms(b), **region_terms(a)}
    shared_keys = set(ma) & set(mb)

    shared_mass = sum(ma[r] for r in shared_keys)
    added_mass = sum(v for r, v in ma.items() if r not in mb)

    union = sorted(set(ma) | set(mb))
    va = [ma.get(r, 0.0) for r in union]
    vb = [mb.get(r, 0.0) for r in union]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    similarity = dot / (na * nb) if na and nb else 0.0

    cats_a, cats_b = category_mass(a), category_mass(b)
    names = category_names()
    shift = [
        (c, names.get(c, f"category {c}"), cats_a.get(c, 0.0), cats_b.get(c, 0.0))
        for c in sorted(set(cats_a) | set(cats_b))
    ]
    shift.sort(key=lambda row: -abs(row[2] - row[3]))

    return Comparison(
        comparable=True,
        shared_mass=shared_mass,
        added_mass=added_mass,
        similarity=similarity,
        a_only=sorted(
            ((r, v, terms.get(r, "")) for r, v in ma.items() if r not in mb),
            key=lambda t: -t[1],
        ),
        b_only=sorted(
            ((r, v, terms.get(r, "")) for r, v in mb.items() if r not in ma),
            key=lambda t: -t[1],
        ),
        shared=sorted(
            ((r, ma[r], mb[r], terms.get(r, "")) for r in shared_keys),
            key=lambda t: -(t[1] + t[2]),
        ),
        a_head_coverage=_head_coverage(a),
        b_head_coverage=_head_coverage(b),
        category_shift=shift,
    )


def _head_coverage(coverage: dict[str, Any]) -> float:
    """Share of placed records the comparison actually saw.

    1.0 whenever the full histogram is present. Below that only for fingerprints
    written before `region_counts` existed, where the top-twelve head is all
    there is and the shares are computed over a partial view.
    """
    if coverage.get("region_counts"):
        return 1.0
    tops = coverage.get("top_regions") or []
    head = sum(int(r.get("records", 0)) for r in tops)
    placed = int(coverage.get("records", 0)) - int(coverage.get("off_atlas", 0))
    return head / placed if placed > 0 else 0.0
