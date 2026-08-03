"""Reading a coverage report, and comparing two of them.

The atlas exists so that two datasets scanned on different machines land in the
same bins and can be put side by side. That property is worth nothing until
something actually puts them side by side, which is what this module is for.

Two rules are enforced here rather than left to callers.

**Off-atlas mass is carried, not used as grounds for refusing.** Every number
here is computed over the records each side actually placed, and both placed
counts travel with the result so the reader always knows the denominator. A high
off-atlas rate on the *right* side biases novelty in one direction and only one:
regions the right side appears not to reach may in fact be reached by records it
could not place, so `added_mass` is an upper bound on what the left side adds.
That bound is reported. Refusing to compare, which is what this module did before
0.1.4, told the user less than the bound does.

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


def category_labels(atlas: Any | None = None) -> dict[int, str]:
    """Coarse-region id to its human-readable label.

    Prefers L1 labels from the given atlas (or the bundled one). Falls back to
    the hand-designed taxonomy.json used by legacy artifacts.
    """
    if atlas is None:
        from .apply import load_bundled

        atlas = load_bundled()
    if atlas is not None and atlas.l1_labels:
        return dict(enumerate(atlas.l1_labels))
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
    from .apply import load_bundled

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
    return max(0.0, min(1.0, float(ent) / float(top)))


def is_usable(coverage: dict[str, Any] | None) -> bool:
    """Whether a coverage facet carries a histogram worth comparing.

    A high off-atlas rate does not make a facet unusable; it makes it partial,
    which `placed_share` quantifies and the comparison reports. The only genuinely
    unusable cases are an absent facet, one from a scan where nothing was placed,
    and one written by a version that suppressed the histogram outright.
    """
    if not coverage or coverage.get("status") != "ok":
        return False
    return bool(coverage.get("region_counts") or coverage.get("top_regions"))


def placed_share(coverage: dict[str, Any] | None) -> float:
    """Share of the sampled records this facet's histogram is built from.

    1.0 when everything placed. The complement is the off-atlas rate, and it is
    the factor by which a comparison against this side is partial.
    """
    if not coverage:
        return 0.0
    records = int(coverage.get("records", 0))
    if records <= 0:
        return 0.0
    placed = coverage.get("placed")
    if placed is None:
        placed = records - int(coverage.get("off_atlas", 0))
    return max(0.0, min(1.0, int(placed) / records))


def unusable_reason(coverage: dict[str, Any] | None) -> str:
    if not coverage:
        return "no atlas coverage was computed"
    status = coverage.get("status")
    if status == "suppressed":
        # Written by 0.1.3 or earlier, which discarded the histogram above a 10%
        # off-atlas rate. The stored reason survived and carries the rate that
        # caused it, so it is repeated rather than replaced; the histogram did
        # not survive, so re-scanning is the only fix.
        stored = str(coverage.get("reason") or "coverage was suppressed")
        return (
            f"{stored.rstrip('. ')}. That was written by a dropoutt before 0.1.4, "
            f"which discarded the region histogram instead of describing the "
            f"off-atlas records. Re-run `dropoutt scan` to get a comparable fingerprint"
        )
    if status == "none placed":
        return "no records were placed on the atlas, so there is no histogram"
    if status == "no records":
        return "no records reached the atlas"
    if status == "ok":
        return "coverage carries no region histogram"
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
    #: Share of each side's sampled records that placed at all. Every mass above
    #: is a share of these, not of the whole corpus.
    a_placed_share: float = 1.0
    b_placed_share: float = 1.0
    #: The same shared/added split rescaled onto *all* of A's sampled records
    #: rather than only the placed ones. These two plus `a_unplaced` sum to 1 by
    #: construction, which `shared_mass` and `added_mass` do not.
    shared_of_all: float = 0.0
    added_of_all: float = 0.0
    a_unplaced: float = 0.0
    #: Set when either side placed little enough that the numbers need reading
    #: with a stated bias rather than at face value.
    caveats: list[str] = field(default_factory=list)


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

    placed_a, placed_b = placed_share(a), placed_share(b)

    terms = {**region_terms(b), **region_terms(a)}
    shared_keys = set(ma) & set(mb)

    shared_mass = sum(ma[r] for r in shared_keys)
    added_mass = sum(v for r, v in ma.items() if r not in mb)

    union = sorted(set(ma) | set(mb))
    va = [ma.get(r, 0.0) for r in union]
    vb = [mb.get(r, 0.0) for r in union]
    dot = sum(x * y for x, y in zip(va, vb, strict=True))
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
        a_placed_share=placed_a,
        b_placed_share=placed_b,
        # A three-way partition of everything the left side sampled. Off-atlas is
        # deliberately kept out of the *similarity* — it is the complement of the
        # atlas, an undifferentiated set, and two corpora that are off-atlas in
        # orthogonal directions would score near-identical if it were carried as
        # a shared coordinate. It is safe as a mass term and misleading as a
        # dimension.
        shared_of_all=shared_mass * placed_a,
        added_of_all=added_mass * placed_a,
        a_unplaced=1.0 - placed_a,
        caveats=_caveats(placed_a, placed_b, added_mass),
    )


def _caveats(placed_a: float, placed_b: float, added_mass: float) -> list[str]:
    """What an incomplete side does to the numbers, and in which direction.

    Stated as a bound rather than a warning, because "treat this with caution"
    tells the reader nothing they can act on. The two sides bias different
    quantities, so they get separate sentences.
    """
    from .apply import OFF_ATLAS_NOTABLE

    out: list[str] = []
    if placed_b < 1 - OFF_ATLAS_NOTABLE:
        out.append(
            f"the right side placed only {placed_b:.0%} of its records, so regions it "
            f"appears not to reach may be reached by records it could not place. "
            f"Read {added_mass:.0%} new as an upper bound"
        )
    if placed_a < 1 - OFF_ATLAS_NOTABLE:
        out.append(
            f"the left side placed only {placed_a:.0%} of its records, so these shares "
            f"describe that {placed_a:.0%} and say nothing about the rest"
        )
    return out


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
