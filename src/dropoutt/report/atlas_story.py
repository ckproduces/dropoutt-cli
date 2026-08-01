"""What the fixed map says about this corpus, in the order it is worth hearing.

The map is a coordinate system, not a quality reference. It says where a corpus
sits relative to a frozen sample of public training data, which is the one
question a list of findings cannot answer: a finding can say what is broken,
and only a fixed reference can say what is *missing*.

Two rules govern everything below.

**Atlas labels are captions, never claims.** A region's five words are the most
frequent terms among the first hundred and fifty reference records that landed
there, and about forty percent of that text is function words shared with every
other region. The level-0 category names were assigned per source dataset
rather than per record, and several describe their contents wrongly — the
category named for religion is Arabic Wikipedia. So a place is named by the
reader's own record nearest its centre, which is true by construction, and the
atlas caption sits beside it. Only in the gaps, where the corpus has no record
to show, is the atlas label used, and it is marked approximate there.

**Nothing appears for being true; it appears for being large and true.** Every
comparison clears an effect-size gate and a significance gate, and the two fail
in opposite directions. On two hundred thousand records every difference is
significant, so significance alone prints a page of three-percent deviations.
On four hundred a five-fold difference is what noise looks like, so effect size
alone prints confident nonsense. Below :data:`MIN_PLACED_FOR_INSIGHT` placed
records no comparison is made at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .phrasing import share as _share


@dataclass
class Place:
    """One neighbourhood of the atlas that this corpus occupies."""

    region: int
    share: float
    records: int
    #: The user's own record closest to the centre of this region.
    yours: str
    #: The atlas's own five words for it. A caption, never a claim.
    caption: str
    #: Mean pairwise cosine among this corpus's records here, when measured.
    cohesion: float | None = None
    #: The subject area the map files this region under.
    area: str = ""

    @property
    def repetitive(self) -> bool:
        return self.cohesion is not None and self.cohesion >= 0.85


@dataclass
class Insight:
    """One thing the map says that is worth a sentence.

    Every insight has to earn its place twice: by effect size, because a 3%
    difference is not worth a reader's attention however certain we are of it,
    and by significance, because on a two-hundred-record sample a 3x difference
    is what noise looks like. Both gates are applied where the insight is built,
    so a report that shows nothing here is saying something true — that the map
    found nothing about this corpus worth remarking on.
    """

    kind: str
    headline: str
    detail: str
    #: Ordering weight. Roughly "how much of the corpus this is about".
    magnitude: float
    tone: str = "neutral"
    #: The reader's own record, when the claim is about a place they occupy.
    evidence: str = ""


@dataclass
class AtlasStory:
    """What the map says, in the order it is worth hearing."""

    available: bool = False
    unavailable_reason: str = ""
    placed: int = 0
    sampled: int = 0
    too_short: int = 0
    regions_touched: int = 0
    regions_total: int = 0
    effective: float = 0.0
    concentration: float | None = None
    shape: str = ""
    shape_line: str = ""
    headline: str = ""
    off_rate: float = 0.0
    off_count: int = 0
    off_line: str = ""
    off_detail: dict[str, Any] = field(default_factory=dict)
    off_examples: list[dict] = field(default_factory=list)
    crowding: str = ""
    places: list[Place] = field(default_factory=list)
    #: Occupied regions holding almost nothing. Reaching a place is not the same
    #: as covering it, and the difference is invisible in an occupancy count.
    thin_places: list[Place] = field(default_factory=list)
    thin_share: float = 0.0
    insights: list[Insight] = field(default_factory=list)
    twins: list[dict] = field(default_factory=list)
    twins_line: str = ""
    gaps: list[dict] = field(default_factory=list)
    gaps_line: str = ""
    categories: list[dict] = field(default_factory=list)
    version: str = ""
    probe_accuracy: float | None = None


#: A corpus this concentrated is one thing rather than a mixture. Not a fault:
#: it is correct for a single-task set and wrong for a pretraining mixture, and
#: the tool has not been told which is being built.
SPECIALIST_CONCENTRATION = 0.45
BROAD_CONCENTRATION = 0.75


def build_story(result) -> AtlasStory | None:
    from ..atlas.compare import category_labels, concentration, unusable_reason

    coverage = result.ctx.stats.get("atlas_coverage")
    if not coverage:
        return None

    story = AtlasStory(version=str(coverage.get("atlas_version") or ""))
    status = coverage.get("status")
    if status not in ("ok", "none placed"):
        story.unavailable_reason = unusable_reason(coverage)
        return story

    story.available = status == "ok"
    story.sampled = int(coverage.get("records", 0))
    story.placed = int(coverage.get("placed", 0))
    story.too_short = int(coverage.get("excluded_too_short", 0))
    story.off_count = int(coverage.get("off_atlas", 0))
    story.off_rate = float(coverage.get("off_atlas_rate", 0.0))
    story.off_detail = coverage.get("off_atlas_detail") or {}
    story.probe_accuracy = coverage.get("l0_holdout_accuracy")
    story.off_examples = list(result.ctx.stats.get("atlas_off_examples") or [])

    if not story.available:
        story.unavailable_reason = (
            "None of the sampled records were close enough to any part of the map "
            "to be placed."
        )
        return story

    story.regions_touched = int(coverage.get("regions_occupied", 0))
    story.regions_total = int(coverage.get("regions_total", 0))
    story.effective = float(coverage.get("effective_regions", 0.0))
    story.concentration = concentration(coverage)
    labels = category_labels(_atlas_of(result))

    story.shape, story.shape_line, story.headline = _shape(story)
    story.places = _places(result, coverage)
    story.thin_places, story.thin_share = _thin_places(result, coverage)
    story.crowding = _crowding(story)
    story.twins, story.twins_line = _twins(coverage)
    story.gaps, story.gaps_line = _gaps(coverage, labels)
    story.categories, _palette = _categories(result, coverage)
    story.insights = _insights(result, coverage, story)
    story.off_line = _off_line(story)
    return story


def _shape(story: AtlasStory) -> tuple[str, str, str]:
    """How spread out the corpus is, said without a verdict attached."""
    touched, total = story.regions_touched, story.regions_total or 1
    effective = story.effective
    conc = story.concentration
    if conc is None:
        return "", "", ""
    if conc < SPECIALIST_CONCENTRATION:
        shape = "specialised"
        line = (
            "This is one specialist corpus, not a mixture. That is right for a "
            "single-task fine-tune and too narrow for general pretraining."
        )
    elif conc > BROAD_CONCENTRATION:
        shape = "broad"
        line = (
            "This is a broad mixture. That is right for a general corpus and "
            "unusually scattered for a single-task set."
        )
    else:
        shape = "mixed"
        line = (
            "This sits between a specialist set and a general mixture: several "
            "areas, with most of the weight in a few of them."
        )
    headline = (
        f"Your data reaches {touched} of {total} places on the map, and is as "
        f"spread out as {effective:.0f} evenly-used ones."
    )
    return shape, line, headline


def _place(result, region: int, records: int, share: float, caption: str = "") -> Place:
    """One region, named by the reader's own record before the atlas's caption."""
    from ..atlas.compare import category_labels

    atlas = _atlas_of(result)
    examples = result.ctx.stats.get("atlas_region_examples") or {}
    cohesion = result.ctx.stats.get("atlas_region_cohesion") or {}
    rows = examples.get(region) or examples.get(str(region)) or []
    coh = cohesion.get(region, cohesion.get(str(region)))
    area = ""
    if atlas is not None and region < len(atlas.region_category):
        area = category_labels(atlas).get(int(atlas.region_category[region]), "")
    if not caption and atlas is not None and region < len(atlas.region_terms):
        caption = atlas.region_terms[region]
    return Place(
        region=region,
        share=share,
        records=records,
        yours=str(rows[0].get("excerpt", "")).replace("\n", " ").strip() if rows else "",
        caption=caption,
        cohesion=float(coh) if coh is not None else None,
        area=area,
    )


#: How many crowded and how many thin places are worth listing. Five of each is
#: enough to see the shape of the distribution and few enough to read.
PLACES_SHOWN = 5


def _places(result, coverage: dict) -> list[Place]:
    return [
        _place(result, int(row["region"]), int(row.get("records", 0)),
               float(row.get("share", 0.0)), str(row.get("terms", "")))
        for row in (coverage.get("top_regions") or [])[:PLACES_SHOWN]
    ]


#: A region holding less than this share of the corpus is a toehold rather than
#: coverage. Set at one record in two hundred: below it, a region's presence in
#: the occupancy count is telling the reader something the data does not support.
THIN_SHARE = 0.005


def _thin_places(result, coverage: dict) -> tuple[list[Place], float]:
    """The sparsest occupied regions, and what they hold between them.

    Occupancy — "you reach 34 of 258 places" — counts a region holding one
    record the same as one holding a third of the corpus, which is exactly how a
    narrow corpus comes to look broad. This is the other half of that number.
    """
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    placed = sum(counts.values())
    if not placed or len(counts) < 4:
        return [], 0.0
    thin = [(r, c) for r, c in counts.items() if c / placed < THIN_SHARE]
    if len(thin) < 3:
        return [], 0.0
    share = sum(c for _r, c in thin) / placed
    thin.sort(key=lambda rc: (rc[1], rc[0]))
    return (
        [_place(result, r, c, c / placed) for r, c in thin[:PLACES_SHOWN]],
        share,
    )


def _crowding(story: AtlasStory) -> str:
    """The single most useful thing the map says about a lopsided corpus."""
    if not story.places:
        return ""
    top = story.places[0]
    if top.share < 0.2:
        return ""
    if top.repetitive:
        return (
            f"{_share(top.share)} of your data sits in one place, and those records "
            f"are {top.cohesion:.2f} alike. That is one thing written out many "
            f"times, not one subject covered many ways — near-duplicate detection "
            f"will not catch it, because they share almost no wording."
        )
    return (
        f"{_share(top.share)} of your data sits in one place on the map. The records "
        f"there vary in wording, so this is a subject you cover heavily rather than "
        f"a template."
    )


def _twins(coverage: dict) -> tuple[list[dict], str]:
    block = coverage.get("by_dataset_regions") or {}
    alike = list(block.get("most_alike") or [])
    if not alike:
        return [], ""
    rows = []
    for pair in alike[:5]:
        similarity = float(pair["similarity"])
        rows.append({
            "a": str(pair["a"]),
            "b": str(pair["b"]),
            "similarity": similarity,
            "verdict": (
                "same ground" if similarity >= 0.8
                else "partly overlapping" if similarity >= 0.5
                else "different ground"
            ),
        })
    top = rows[0]
    if top["similarity"] >= 0.8:
        line = (
            f"{top['a']} and {top['b']} occupy the same ground ({top['similarity']:.2f}). "
            f"Merging them adds volume, not coverage — and text-level duplicate "
            f"detection cannot see this, because they may share no wording at all."
        )
    else:
        line = (
            f"Your datasets cover different ground; the closest pair, {top['a']} and "
            f"{top['b']}, overlaps at {top['similarity']:.2f}."
        )
    return rows, line


def _gaps(coverage: dict, labels: dict[int, str]) -> tuple[list[dict], str]:
    raw = coverage.get("coverage_gaps") or []
    total = int(coverage.get("categories_total", 0)) or len(raw)
    rows = [
        {
            "name": labels.get(int(g["category"]), f"area {g['category']}"),
            "regions": int(g["regions"]),
            "records": int(g["records"]),
            "caption": str(g.get("terms") or ""),
        }
        for g in raw
    ]
    if not rows:
        return [], "Your data reaches every subject area the map covers."
    line = (
        f"{len(rows)} of the {total} subject areas the map covers are empty or "
        f"nearly empty here. Whether that matters depends on what you are "
        f"building — a specialist set is supposed to have gaps."
    )
    return rows, line


#: Distinct fills the subject bars cycle through, plus one for everything past
#: the named few. Chosen to stay apart at bar width, not to be pretty.
MAP_HUES = 6
OTHER_HUE = 6


def _atlas_of(result):
    atlas = result.ctx.atlas
    if atlas is not None:
        return atlas
    try:
        from ..atlas import load_bundled

        return load_bundled()
    except Exception:  # pragma: no cover - defensive
        return None


def _region_categories(result, coverage: dict) -> tuple[dict[int, int], dict[int, int]]:
    """Records per level-0 area, counted through the regions they landed in.

    Not from the ``by_category`` facet, which comes from the supervised probe
    applied to each record independently. The two disagree — a record can be
    nearest to a region filed under one area while the probe calls it another —
    and every other number on the page is counted through regions, so a bar
    chart driven by the probe would name areas that no place in the list beside
    it belongs to. One denominator, one path to it.
    """
    atlas = _atlas_of(result)
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    if atlas is None or not counts:
        return {}, counts
    per_area: dict[int, int] = {}
    for region, count in counts.items():
        if region < len(atlas.region_category):
            area = int(atlas.region_category[region])
            per_area[area] = per_area.get(area, 0) + count
    return per_area, counts


def _categories(result, coverage: dict) -> tuple[list[dict], dict[int, int]]:
    """Each subject area's share of this corpus, beside the map's own share.

    The second number is what turns a bar chart into a comparison. On its own
    "27% code generation" is a fact with nothing to weigh it against; next to
    "the map spends 4% of itself there" it is the finding.
    """
    from ..atlas.compare import category_labels

    labels = category_labels(_atlas_of(result))
    per_area, _ = _region_categories(result, coverage)
    if not per_area:
        return [], {}
    allocation, _total_regions = _map_allocation(result)
    placed = sum(per_area.values()) or 1
    ranked = sorted(per_area.items(), key=lambda kv: -kv[1])
    palette = {area: i for i, (area, _) in enumerate(ranked[:MAP_HUES])}
    rows = [
        {
            "name": labels.get(area, f"area {area}"),
            "share": count / placed,
            "map_share": allocation.get(area, 0.0),
            "hue": palette.get(area, OTHER_HUE),
        }
        for area, count in ranked[:8]
    ]
    return rows, palette


def _off_line(story: AtlasStory) -> str:
    if not story.off_count:
        return "Every sampled record was close enough to the map to be placed."
    diagnosis = str(story.off_detail.get("diagnosis") or "")
    lead = (
        f"{_share(story.off_rate)} of sampled records ({story.off_count:,}) look "
        f"like nothing in the reference corpus."
    )
    return f"{lead} {diagnosis[0].upper() + diagnosis[1:]}." if diagnosis else lead


# --------------------------------------------------------------------------
# What the map actually says
# --------------------------------------------------------------------------
#
# The page used to draw all 258 regions as a scatter plot. It was honest — the
# layout is fixed, so two reports could be held side by side — and it was
# useless: the positions are a projection, not distances, which had to be
# disclaimed in a caption directly under the picture, and having read the
# caption there was nothing left the reader could do with the dots. What people
# actually want from a map of their corpus is a handful of sentences, so this
# computes those instead.
#
# Nothing below is shown because it is true. It is shown because it is *large*
# and true. Both gates are needed and they fail in opposite directions: on a
# 200k-record corpus every difference is statistically significant, so
# significance alone would print a page of 3% deviations; on a 400-record sample
# a 5x difference is what noise looks like, so effect size alone would print
# confident nonsense.

#: A subject area has to be at least this dense relative to the map's own
#: allocation before over-representation is worth a sentence.
OVER_LIFT = 2.5
#: ...and hold at least this much of the corpus, so a rounding error in a tiny
#: area cannot produce a headline.
OVER_SHARE = 0.05
#: The map has to spend at least this much of itself on an area before its
#: absence is a finding rather than a triviality.
UNDER_MAP_SHARE = 0.04
#: An area is under-covered when the corpus holds under a quarter of what the
#: map's allocation would suggest.
UNDER_LIFT = 0.25
#: Standard errors between observed and expected before either is reported. Two
#: and a half is about a 1% two-sided false-positive rate per area tested, which
#: at twenty areas is the right trade for a page nobody should have to audit.
SIGMA = 2.5
#: One region holding this much of a corpus is the single most useful sentence
#: the map produces, so it leads.
DOMINANT_REGION = 0.20
#: Below this many placed records, no comparison against the map is made at all.
#: The binomial gate above would happily clear a 4x difference on thirty records
#: — and be right to, as arithmetic — but the placement itself is a nearest
#: centroid over an 86%-accurate probe, and thirty records is not a description
#: of a corpus. This is the floor the arithmetic cannot see.
MIN_PLACED_FOR_INSIGHT = 200
#: At most this many of each comparative kind. Past three, the fourth-densest
#: subject is not telling the reader anything the first three did not, and the
#: section stops being a set of findings and becomes a table with prose in it.
INSIGHTS_PER_KIND = 3

#: Reading order. Grouped by kind rather than sorted purely by size, because the
#: kinds are measured in different units — an over-representation is a share of
#: your corpus and an under-representation is a share of the map — and sorting
#: two different quantities against each other puts them in an order that means
#: nothing. Within a kind, largest first.
INSIGHT_ORDER = ("dominance", "over", "under", "thin", "twins")


def _significant(observed: float, expected: float, n: int) -> bool:
    """Whether a share differs from its expectation by more than sampling noise.

    A binomial standard error on the *expected* share, which is the null being
    tested. ``n`` is the number of records actually placed, not the corpus size:
    the atlas sees a sample, and pretending otherwise would narrow every
    interval on the page by the sampling ratio.
    """
    if n <= 0 or not 0.0 < expected < 1.0:
        return False
    se = math.sqrt(expected * (1.0 - expected) / n)
    return se > 0 and abs(observed - expected) >= SIGMA * se


def _map_allocation(result) -> tuple[dict[int, float], int]:
    """What share of its own regions the map spends on each subject area.

    This is the only reference distribution the shipped artifact carries. The
    atlas was clustered once over the whole reference corpus, so the number of
    regions an area drew is how much resolution that corpus made it worth — a
    proxy for reference mass, and reported as what it is. It is deliberately not
    called a share of the reference corpus, because it is not one.
    """
    atlas = _atlas_of(result)
    if atlas is None or not len(atlas.region_category):
        return {}, 0
    per_area: dict[int, int] = {}
    for area in atlas.region_category:
        per_area[int(area)] = per_area.get(int(area), 0) + 1
    total = sum(per_area.values()) or 1
    return {a: n / total for a, n in per_area.items()}, total


def _insights(result, coverage: dict, story: AtlasStory) -> list[Insight]:
    """Every claim the map supports, largest first."""
    from ..atlas.compare import category_labels

    out: list[Insight] = []
    placed = story.placed or 1
    labels = category_labels(_atlas_of(result))
    per_area, _counts = _region_categories(result, coverage)
    allocation, n_regions = _map_allocation(result)
    area_counts_by_id = dict(per_area)
    corpus_area = {a: c / max(sum(per_area.values()), 1) for a, c in per_area.items()}

    # -- one place holding most of the corpus ------------------------------
    if story.places and story.places[0].share >= DOMINANT_REGION:
        top = story.places[0]
        if top.repetitive:
            detail = (
                f"Records there are {top.cohesion:.2f} alike, which is one thing "
                f"written out many times rather than one subject covered many "
                f"ways. Near-duplicate detection will not catch it: they share "
                f"almost no wording."
            )
        else:
            detail = (
                "The records there vary in wording, so this is a subject covered "
                "heavily rather than a template repeated."
            )
        out.append(Insight(
            kind="dominance",
            headline=f"{_share(top.share)} of your data sits in a single place on the map",
            detail=detail,
            magnitude=top.share,
            tone="warn" if top.repetitive or top.share >= 0.5 else "neutral",
            evidence=top.yours,
        ))

    # -- subjects denser here than the map is built for --------------------
    #
    # The trailing "what this means" sentence goes on the first of each kind
    # only. Three cards ending in the same clause reads as boilerplate and
    # teaches the reader to stop at the headline.
    comparable = story.placed >= MIN_PLACED_FOR_INSIGHT
    over = 0
    for area, _count in sorted(area_counts_by_id.items(), key=lambda kv: -kv[1]):
        if not comparable or over >= INSIGHTS_PER_KIND:
            break
        observed = corpus_area.get(area, 0.0)
        expected = allocation.get(area, 0.0)
        if not expected or observed < OVER_SHARE:
            continue
        if observed / expected < OVER_LIFT:
            continue
        if not _significant(observed, expected, placed):
            continue
        regions = round(expected * n_regions)
        detail = (
            f"The map spends {regions} of its {n_regions} places on that "
            f"subject; {_share(observed)} of your placed records land there."
        )
        if not over:
            detail += (
                " That is what a specialist corpus looks like, and it is only a "
                "problem if you meant to build a general one."
            )
        out.append(Insight(
            kind="over",
            headline=(
                f"{labels.get(area, f'area {area}')} — {observed / expected:.1f}× "
                f"denser here than on the map"
            ),
            detail=detail,
            magnitude=observed,
        ))
        over += 1

    # -- subjects the map is built for and this corpus barely reaches ------
    under = 0
    for area, expected in sorted(allocation.items(), key=lambda kv: -kv[1]):
        if not comparable or under >= INSIGHTS_PER_KIND:
            break
        if expected < UNDER_MAP_SHARE:
            continue
        observed = corpus_area.get(area, 0.0)
        if observed > expected * UNDER_LIFT:
            continue
        if not _significant(observed, expected, placed):
            continue
        regions = round(expected * n_regions)
        detail = (
            f"{regions} of the map's {n_regions} places sit in that subject "
            f"because the reference corpus had enough of it to need them."
        )
        if not under:
            detail += " Whether the gap matters depends on what you are building."
        out.append(Insight(
            kind="under",
            headline=(
                f"{labels.get(area, f'area {area}')} — the map gives it "
                f"{_share(expected)} of itself, your data puts {_share(observed)} there"
            ),
            detail=detail,
            magnitude=expected,
        ))
        under += 1

    # -- reached but not covered -------------------------------------------
    if story.thin_places and story.thin_share:
        thin = len([1 for r, c in (coverage.get("region_counts") or {}).items()
                    if int(c) / placed < THIN_SHARE])
        solid = story.regions_touched - thin
        if thin >= 3 and solid >= 1:
            out.append(Insight(
                kind="thin",
                headline=(
                    f"Of the {story.regions_touched} places you reach, "
                    f"{thin} hold {_share(story.thin_share)} of your data between them"
                ),
                detail=(
                    f"Real presence in {solid} place"
                    f"{'' if solid == 1 else 's'}, a toehold in the rest. An "
                    f"occupancy count reads a place holding one record the same "
                    f"as one holding a third of the corpus, which is how a "
                    f"narrow corpus comes to look broad."
                ),
                magnitude=story.thin_share,
            ))

    # -- datasets standing on each other's ground --------------------------
    if story.twins and story.twins[0]["similarity"] >= 0.8:
        pair = story.twins[0]
        out.append(Insight(
            kind="twins",
            headline=(
                f"{pair['a']} and {pair['b']} occupy the same ground "
                f"({pair['similarity']:.2f} alike)"
            ),
            detail=(
                "Merging them adds volume, not coverage. Text-level duplicate "
                "detection cannot see this — the two may share no wording at all."
            ),
            magnitude=float(pair["similarity"]),
            tone="warn",
        ))

    order = {kind: n for n, kind in enumerate(INSIGHT_ORDER)}
    out.sort(key=lambda i: (order.get(i.kind, len(order)), -i.magnitude))
    return out
