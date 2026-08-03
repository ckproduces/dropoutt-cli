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
    #: This corpus's density here against the reference corpus's own, which is
    #: what both lists are ranked on. A share cannot rank a neighbourhood: the
    #: map is not evenly divided, so the biggest place in a corpus is usually
    #: just the biggest place on the map.
    ratio: float = 0.0
    #: Mean pairwise cosine among this corpus's records here, when measured.
    cohesion: float | None = None
    #: The subject area the map files this region under.
    area: str = ""

    @property
    def repetitive(self) -> bool:
        return self.cohesion is not None and self.cohesion >= 0.85


@dataclass
class Imbalance:
    """One cell far from map density, with an example and a cut/grow cue."""

    region: int
    ratio: float
    records: int
    share: float
    yours: str
    area: str
    #: ``cut`` when denser than the map, ``grow`` when thinner.
    action: str


@dataclass
class Cell:
    """One fine cell of the map, and how densely this corpus sits in it.

    ``ratio`` is the only number here that means anything on its own: it is this
    corpus's share of the cell divided by the reference corpus's share of the
    same cell, so 1.0 is "as dense as the map itself" and 0.0 is "never
    reached". ``level`` is that ratio placed on the 0–4 colour scale, which is
    anchored at parity rather than at the extremes — see :func:`_grid`.
    """

    region: int
    records: int
    ratio: float
    level: float
    caption: str

    @property
    def step(self) -> int:
        """The quantised colour step on the density ramp.

        Step 0 is white. Unreached cells use it with a zero; reached cells at
        the bottom of the scale use the same step so thin coverage reads as
        near-white, then steps climb smoothly toward green.
        """
        from dropoutt.report.theme import RAMP_DIVISOR

        top = int(MAX_LEVEL * RAMP_DIVISOR)
        step = round(self.level * RAMP_DIVISOR)
        return max(0, min(top, step)) if self.records else 0

    @property
    def label(self) -> str:
        """What the cell says, short enough to sit inside it.

        The ratio used to be a hover tooltip. A number that only exists while a
        mouse is over it does not exist on paper, in a screenshot pasted into a
        ticket, or for anyone reading with a keyboard — which is most of the
        ways this page is read. An unreached cell says ``0``.
        """
        if not self.records:
            return "0"
        if self.ratio >= 10:
            return f"{self.ratio:.0f}×"
        if self.ratio >= 1:
            return f"{self.ratio:.1f}×"
        return f"{self.ratio:.2f}×".lstrip("0")

    @property
    def described(self) -> str:
        """The same claim spelled out, for a screen reader."""
        if not self.records:
            return "not reached"
        return (
            f"{density_ratio(self.ratio)} the map's own density, "
            f"{self.records:,} record{'' if self.records == 1 else 's'}"
        )


@dataclass
class Area:
    """One subject area of the map: a row of the density grid."""

    area: int
    name: str
    records: int
    share: float
    ratio: float
    cells: list[Cell] = field(default_factory=list)

    @property
    def unreached(self) -> int:
        return sum(1 for cell in self.cells if not cell.records)

    @property
    def effective_reach(self) -> float:
        """How many of this area's cells the corpus covers, at map density.

        Each subregion contributes ``min(1, density_ratio)``: parity (1×) is a
        full score, thinner coverage is a fraction, and over-representation
        does not add more than one. Entropy used to shrink the number when a
        cell was heavy; that punished breadth for having a peak.
        """
        return sum(min(1.0, max(0.0, cell.ratio)) for cell in self.cells)

    @property
    def fully_reached(self) -> bool:
        """Every subregion is at least at map density."""
        return bool(self.cells) and self.effective_reach >= len(self.cells) - 1e-9


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

    @property
    def label(self) -> str:
        """What kind of claim this is, in two words.

        The headlines are each about a different quantity — a share of the
        corpus, a share of the map, a similarity — and a reader who cannot tell
        which one they are looking at reads five sentences and takes away none.
        """
        return {
            "dominance": "concentration",
            "over": "over-represented",
            "under": "under-represented",
            "thin": "reach without coverage",
            "twins": "overlapping datasets",
        }.get(self.kind, self.kind)


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
    #: Cells farthest from map density, cut and grow interleaved, so a reader
    #: can see which records to thin and which subjects to add.
    imbalances: list[Imbalance] = field(default_factory=list)
    #: The single place holding most of the corpus, by share rather than by
    #: density. `places` is ranked on density, and "42% of your data is here"
    #: is a different sentence from "you are 26x the map here".
    dominant: Place | None = None
    thin_share: float = 0.0
    insights: list[Insight] = field(default_factory=list)
    twins: list[dict] = field(default_factory=list)
    twins_line: str = ""
    gaps: list[dict] = field(default_factory=list)
    gaps_line: str = ""
    categories: list[dict] = field(default_factory=list)
    #: Every subject area of the map, occupied or not, each carrying its own
    #: fine cells. The whole map rather than the part this corpus reached,
    #: because the empty rows are half of what the picture says.
    grid: list[Area] = field(default_factory=list)
    #: The densest cell on the map relative to the reference corpus, which is
    #: what the top of the colour scale was normalised against.
    grid_peak: float = 0.0
    #: Independent observations behind the histogram, after weighting. Not
    #: `placed`: a record standing for five thousand others is still one
    #: record's worth of evidence, and every significance gate takes this.
    effective_sample: float = 0.0
    #: How hard each cell was pulled towards the map, and where an unreached
    #: cell lands. See :meth:`dropoutt.atlas.apply.Atlas._prior_strength`.
    prior_strength: float = 0.0
    unreached_density: float = 0.0
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

    model = coverage.get("density_model") or {}
    story.effective_sample = float(model.get("effective_sample", 0.0))
    story.prior_strength = float(model.get("prior_strength", 0.0))
    story.unreached_density = float(model.get("unreached_density", 0.0))
    story.regions_touched = int(coverage.get("regions_occupied", 0))
    story.regions_total = int(coverage.get("regions_total", 0))
    story.concentration = concentration(coverage)
    labels = category_labels(_atlas_of(result))

    story.places, story.thin_places = _ranked_places(result, coverage)
    story.imbalances = _imbalances(result, coverage)
    story.dominant = _dominant(result, coverage)
    story.thin_share = _thin_share(coverage)
    story.crowding = _crowding(story)
    story.twins, story.twins_line = _twins(coverage)
    story.gaps, story.gaps_line = _gaps(coverage, labels)
    story.categories, _palette = _categories(result, coverage)
    story.grid, story.grid_peak = _grid(result, coverage, labels)
    # Prefer the density-capped sum from the grid; fall back to the facet.
    if story.grid:
        story.effective = sum(area.effective_reach for area in story.grid)
    else:
        story.effective = float(coverage.get("effective_regions", 0.0))
    story.shape, story.shape_line, story.headline = _shape(story)
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
        f"Your data reaches {touched} of {total} places on the map, with "
        f"{format_reach(effective)} of {total} in effective coverage "
        f"(1× density counts as one)."
    )
    return shape, line, headline


def _place(result, region: int, records: int, share: float,
           ratio: float = 0.0, caption: str = "") -> Place:
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
        ratio=ratio,
        cohesion=float(coh) if coh is not None else None,
        area=area,
    )


#: How many over- and how many under-represented places are worth listing. Five
#: of each is enough to see the shape and few enough to read.
PLACES_SHOWN = 5

def _ranked_places(result, coverage: dict) -> tuple[list[Place], list[Place]]:
    """The map's own extremes for this corpus: densest and thinnest, by ratio.

    Both lists used to be ranked on raw record count — which asks "where is
    most of my data", a question the section above already answers, and which
    is mostly a fact about the map rather than about the corpus. A region the
    reference corpus spent forty cells on will hold more of anybody's data than
    one it spent four on.

    Ranking by density against the map asks the question the fixed reference
    exists for: where does this corpus stand *out*, and where does it show up
    only nominally.

    There is no minimum share here any more. There used to be, because the raw
    quotient made a one-record cell the most extreme thing on the map in either
    direction, and a hand-picked floor was the cheapest way to keep it off the
    list. The estimate is shrunk now — see
    :meth:`dropoutt.atlas.apply.Atlas._density` — so a cell with no evidence
    behind it sits near parity and sorts itself out of both ends.
    """
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    ratios = {
        int(region): float(value)
        for region, value in (coverage.get("region_density") or {}).items()
    }
    placed = sum(counts.values())
    if not placed or not ratios:
        return [], []

    candidates = [
        (region, counts[region], counts[region] / placed, ratios[region])
        for region in counts
        if ratios.get(region, 0.0) > 0
    ]
    if not candidates:
        return [], []

    by_density = sorted(candidates, key=lambda row: (-row[3], row[0]))
    dense = [_place(result, r, n, share, ratio)
             for r, n, share, ratio in by_density[:PLACES_SHOWN]]
    # The thin list is the other end of the same ordering, and must not repeat
    # the dense one when the corpus occupies fewer places than both want.
    shown = {p.region for p in dense}
    thin = [_place(result, r, n, share, ratio)
            for r, n, share, ratio in reversed(by_density)
            if r not in shown][:PLACES_SHOWN]
    return dense, thin


#: How many cut/grow cells to show in the rebalance section.
IMBALANCE_SHOWN = 8


def _imbalances(result, coverage: dict) -> list[Imbalance]:
    """Cells farthest from map density, so a reader knows what to cut or grow.

    Ranked by distance from parity on a log scale: 6× and 0.17× are the same
    distance from 1×, and both are more useful than a 1.1× cell.
    """
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    ratios = {
        int(region): float(value)
        for region, value in (coverage.get("region_density") or {}).items()
    }
    placed = sum(counts.values())
    if not placed or not ratios:
        return []

    ranked = sorted(
        (
            (region, counts[region], counts[region] / placed, ratios[region])
            for region in counts
            if ratios.get(region, 0.0) > 0
        ),
        key=lambda row: (-abs(math.log(row[3])), row[0]),
    )
    out: list[Imbalance] = []
    for region, records, share, ratio in ranked[:IMBALANCE_SHOWN]:
        place = _place(result, region, records, share, ratio)
        out.append(Imbalance(
            region=region,
            ratio=ratio,
            records=records,
            share=share,
            yours=place.yours,
            area=place.area,
            action="cut" if ratio > 1.0 else "grow",
        ))
    return out


#: A region holding less than this share of the corpus is a toehold rather than
#: coverage. Set at one record in two hundred: below it, a region's presence in
#: the occupancy count is telling the reader something the data does not support.
THIN_SHARE = 0.005


def _thin_share(coverage: dict) -> float:
    """What the toehold regions hold between them.

    Occupancy — "you reach 34 of 212 places" — counts a region holding one
    record the same as one holding a third of the corpus, which is exactly how
    a narrow corpus comes to look broad. This is the other half of that number,
    and it feeds the insight rather than a list.
    """
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    placed = sum(counts.values())
    if not placed or len(counts) < 4:
        return 0.0
    thin = [count for count in counts.values() if count / placed < THIN_SHARE]
    return sum(thin) / placed if len(thin) >= 3 else 0.0


def _dominant(result, coverage: dict) -> Place | None:
    """The place holding most of the corpus, by share."""
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    placed = sum(counts.values())
    if not placed:
        return None
    ratios = {
        int(region): float(value)
        for region, value in (coverage.get("region_density") or {}).items()
    }
    region = max(counts, key=lambda r: (counts[r], -r))
    return _place(result, region, counts[region], counts[region] / placed,
                  ratios.get(region, 0.0))


def _crowding(story: AtlasStory) -> str:
    """The single most useful thing the map says about a lopsided corpus."""
    if story.dominant is None:
        return ""
    top = story.dominant
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
    rows: list[dict[str, Any]] = []
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


#: How many of a cell's caption terms are worth showing. The caption is a
#: frequency list, and past the fourth term it is function words.
CAPTION_TERMS = 4


def density_ratio(value: float) -> str:
    """A density ratio as the reader would say it out loud."""
    if value <= 0:
        return "0×"
    if value < 0.1:
        return "under 0.1×"
    if value < 10:
        return f"{value:.1f}×"
    return f"{value:.0f}×"


def format_reach(value: float) -> str:
    """Effective reach for the grid: ``5`` not ``5.0``, else one decimal."""
    if abs(value - round(value)) < 1e-9:
        return str(round(value))
    return f"{value:.1f}"


def _grid(result, coverage: dict,
          labels: dict[int, str]) -> tuple[list[Area], float]:
    """The whole map as a grid: one row per subject area, one square per cell.

    Every other number in this section is a share of the corpus, which answers
    "what is my data" and not "where is it, against everything else". This
    answers the second question the only way a fixed map can, and the division
    that does it is :meth:`Atlas._density` rather than anything here — a number
    a reader can see has to be a number a script can read, and it is written
    into the coverage facet before it is drawn.

    The scale is anchored at parity, which is the one value that means
    something without a second number beside it. Above parity it is normalised
    to the densest cell this corpus actually has, so the picture uses its whole
    range on a flat corpus and on a lopsided one alike. Below parity it is not:
    0.4 has to look like 0.4 of the way to parity in every report, or two
    reports cannot be held side by side.
    """
    atlas = _atlas_of(result)
    if atlas is None or not len(getattr(atlas, "region_category", ())):
        return [], 0.0
    counts = {
        int(region): int(count)
        for region, count in (coverage.get("region_counts") or {}).items()
    }
    ratios = {
        int(region): float(value)
        for region, value in (coverage.get("region_density") or {}).items()
    }
    placed = sum(counts.values())
    if not placed or not ratios:
        return [], 0.0
    peak = max(ratios.values(), default=0.0)

    terms = atlas.region_terms
    sizes = (
        [float(x) for x in atlas.region_size]
        if atlas.region_size is not None else []
    )
    reference = sum(sizes)

    rows: dict[int, Area] = {}
    for region in range(len(atlas.region_category)):
        parent = int(atlas.region_category[region])
        area = rows.get(parent)
        if area is None:
            area = rows[parent] = Area(
                area=parent,
                name=labels.get(parent, f"area {parent}"),
                records=0, share=0.0, ratio=0.0,
            )
        records = counts.get(region, 0)
        ratio = ratios.get(region, 0.0)
        area.records += records
        area.cells.append(Cell(
            region=region,
            records=records,
            ratio=ratio,
            level=_level(ratio, peak),
            caption=", ".join(
                term.strip()
                for term in (terms[region] if region < len(terms) else "").split(",")
                [:CAPTION_TERMS]
            ),
        ))

    # The row's own density is shrunk by the same rule as its cells, and with
    # the same prior. A row printed as a raw quotient beside cells that were
    # not is a row that disagrees with the squares underneath it.
    model = coverage.get("density_model") or {}
    n = float(model.get("effective_sample") or placed) or 1.0
    alpha = float(model.get("prior_strength") or 0.0)

    grid = list(rows.values())
    for area in grid:
        area.cells.sort(key=lambda cell: (-cell.ratio, cell.region))
        area.share = area.records / placed
        if reference > 0:
            expected = sum(sizes[cell.region] for cell in area.cells) / reference
            if expected > 0:
                area.ratio = (n * area.share + alpha) / (n * expected + alpha)
    return sorted(grid, key=lambda a: (-a.share, a.name.lower())), peak


#: Top of the colour scale. Four stops — white, green, yellow, red — so a level
#: is also the index of the stop it sits on or past.
MAX_LEVEL = 3.0

#: How hard the sub-parity half of the ramp is pulled towards white. Linear was
#: wrong in a way that only shows on real data: a cell at 0.02x — one record
#: where the map has fifty — came out a soft green two percent of the way along,
#: and read as "fine". Squaring over-compressed the mid-low band so 0.34x
#: collapsed to white; 1.5 keeps the bottom near white while still tinting
#: through the 0.1–0.5 band.
SUB_PARITY_CURVE = 1.5


def _level(ratio: float, peak: float) -> float:
    """A density ratio on the 0–3 colour scale, anchored at parity.

    Below parity the curve is fixed rather than normalised, so 0.4x is the same
    colour in every report ever produced and two of them can be held side by
    side. Above parity it is logarithmic to the densest cell this corpus has,
    because density ratios are multiplicative — twice the map and half the map
    are the same distance from it — and because one runaway cell is common. A
    corpus with a 149x neighbourhood in it would, on a linear scale, print every
    other cell as parity green: the outlier would be all the picture said.
    """
    if ratio <= 0:
        return 0.0
    if ratio <= 1:
        return ratio ** SUB_PARITY_CURVE
    if peak <= 1:
        return 1.0
    return 1.0 + (MAX_LEVEL - 1.0) * min(math.log(ratio) / math.log(peak), 1.0)


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


def _significant(observed: float, expected: float, n: float) -> bool:
    """Whether a share differs from its expectation by more than sampling noise.

    A binomial standard error on the *expected* share, which is the null being
    tested. ``n`` is how many independent observations are behind the histogram
    — the effective sample size, not the corpus size and not the weighted total.
    Pretending otherwise would narrow every interval on the page by the sampling
    ratio, and after weighting a single record can stand for thousands.
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
    # Effective, not placed: every gate below asks "could this be noise", and
    # the answer depends on how many independent records were seen rather than
    # on how many they were scaled up to represent.
    placed = story.effective_sample or story.placed or 1
    labels = category_labels(_atlas_of(result))
    per_area, _counts = _region_categories(result, coverage)
    allocation, n_regions = _map_allocation(result)
    area_counts_by_id = dict(per_area)
    corpus_area = {a: c / max(sum(per_area.values()), 1) for a, c in per_area.items()}

    # -- one place holding most of the corpus ------------------------------
    if story.dominant is not None and story.dominant.share >= DOMINANT_REGION:
        top = story.dominant
        if top.repetitive:
            detail = (
                f"Records there are {top.cohesion:.2f} alike, which is one thing "
                f"written out many times rather than one subject covered many "
                f"ways. Near-duplicate detection will not catch it: they share "
                f"almost no wording."
            )
        elif top.cohesion is not None:
            detail = (
                f"{top.records:,} records land there, {top.cohesion:.2f} alike on "
                f"average. Below 0.85 that is a subject, not a template: they say "
                f"the same kind of thing in enough different ways to be worth "
                f"keeping."
            )
        else:
            detail = (
                f"{top.records:,} of your placed records land there — a quarter "
                f"of the corpus or more in one neighbourhood of the map."
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
    comparable = placed >= MIN_PLACED_FOR_INSIGHT
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
