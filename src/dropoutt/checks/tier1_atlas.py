"""Tier 1 checks that read the atlas.

The atlas was, until now, a panel: it printed where the corpus sat and stopped.
Nothing it measured could ever appear in the findings table, which meant the one
part of the tool holding a fixed external reference frame produced no
actionable output at all.

These two checks close that. Both are about *shape* rather than content, and
both are things no per-record check can see, because a record is never
individually wrong for being in a crowded region or for failing to be in an
empty one.

Neither of them recommends deleting anything. Concentration is correct for a
specialised dataset and wrong for a pretraining mixture, and the tool does not
know which one is being built — so both findings describe and neither
prescribes. That is the same rule that governs the corpus quality filters.
"""

from __future__ import annotations

from typing import Any

from ..context import ScanContext
from ..models import CostClass, Finding, Profile, Requirement, Severity
from .base import Check, make_finding, register

ALL_PROFILES = (Profile.SFT, Profile.CORPUS, Profile.PREFERENCE, Profile.UNKNOWN)

#: A single cell holding at least this many times its even share of the
#: placed records is crowded, floored at :data:`CROWDED_REGION_FLOOR`. Set
#: against the map's own geometry rather than as one fixed share: on the
#: 215-cell atlas-v1-lite even mass is 0.47% per cell and forty times that is
#: the 20% the threshold used to be written as; on the 4,096-cell atlas-v3 a
#: fixed 20% was eight hundred times even mass, a bar no template cluster
#: reaches, and the check could not fire on the product it ships with.
CROWDED_REGION_MULTIPLE = 40
CROWDED_REGION_FLOOR = 0.02


def crowded_region_share(regions_total: int) -> float:
    """Share of placed records above which one cell counts as crowded."""
    if regions_total <= 0:
        return CROWDED_REGION_FLOOR
    return max(CROWDED_REGION_FLOOR, CROWDED_REGION_MULTIPLE / regions_total)


#: Mean pairwise cosine inside a crowded region, above which its contents are
#: better described as one thing repeated than as a topic. Calibrated on the
#: same measurements as the off-atlas coherence bands: real prose on a shared
#: subject runs near 0.28, and templated or near-identical text runs 0.87 and up.
CROWDED_REGION_COHESION = 0.75

#: Below this many placed records the histogram is too thin to describe shape.
#: Scaled with the map: three hundred records over 4,096 cells leaves every
#: cell's expected count under the floor the density prior needs, so each
#: occupied cell rounds to parity and reach collapses to occupancy.
MIN_PLACED = 300
MIN_PLACED_PER_CELLS = 8


def min_placed(regions_total: int) -> int:
    return max(MIN_PLACED, regions_total // MIN_PLACED_PER_CELLS)


def _coverage(ctx: ScanContext) -> dict[str, Any] | None:
    cov = ctx.stats.get("atlas_coverage")
    if not cov or cov.get("status") != "ok":
        return None
    if int(cov.get("placed", 0)) < min_placed(int(cov.get("regions_total", 0))):
        return None
    return cov


@register
class TopicalConcentration(Check):
    check_id = "T1-ATLAS-001"
    title = "The corpus covers very little of the map"
    tier = 1
    unit = "map area"
    profiles = ALL_PROFILES
    requires = (Requirement.ATLAS,)
    cost = CostClass.EMBEDDING
    severity = Severity.INFO
    fix = (
        "If this was meant to be a broad mixture, the missing regions are listed "
        "under coverage gaps."
    )
    rationale = (
        "Occupancy counts a region the same whether it holds one record or a third of the "
        "corpus, so 'a few dozen regions occupied' can describe a broad corpus or a corpus "
        "that is really two regions with noise around them. Effective coverage sums "
        "min(1, density_ratio) over subregions — parity is a full score, thinner coverage a "
        "fraction — and the gap between occupied and effective is the finding. This is "
        "reported, never judged: a Turkish legal-QA set should be concentrated, and a "
        "pretraining mixture should not, and the tool has not been told which one this is."
    )

    #: Narrowness is tested against the map's size, not against the occupied
    #: count. Effective coverage at or under this share of the map's cells is
    #: small in topical extent whatever the map: ten cells' worth of the
    #: 215-cell atlas-v1-lite, which is where the bound was first drawn, is
    #: 4.7%, and the same share of atlas-v3 is 205 of its 4,096 cells. A lead
    #: cell holding a quarter of the corpus is dominance on any map.
    MAX_EFFECTIVE_SHARE = 0.05
    MAX_TOP_SHARE = 0.25

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        cov = _coverage(ctx)
        if cov is None:
            return []
        occupied = int(cov.get("regions_occupied", 0))
        effective = float(cov.get("effective_regions", 0.0))
        total = int(cov.get("regions_total", 0))
        if occupied < 2 or effective <= 0 or total <= 0:
            return []
        tops = cov.get("top_regions") or []
        lead = tops[0] if tops else {}
        lead_share = float(lead.get("share", 0.0)) if lead else 0.0
        if effective > self.MAX_EFFECTIVE_SHARE * total and lead_share < self.MAX_TOP_SHARE:
            return []
        detail = (
            f"placed records touch {occupied} of {int(cov.get('regions_total', 0))} "
            f"subregions with {effective:.1f} effective coverage "
            f"(1× density counts as one)"
        )
        if lead:
            detail += f"; the largest single area holds {lead_share:.0%} of them"
        # The atlas's own five words for a region are deliberately not quoted
        # here. They are the most frequent words among the first hundred and
        # fifty reference records that landed there, about forty percent of that
        # text is function words shared with other regions, and a finding that
        # ends in "(such, used, other, also, some)" reads as noise. The report's
        # map names the same place with the reader's own nearest record.
        return [
            make_finding(
                self,
                count=occupied,
                total=int(cov.get("regions_total", 0)),
                detail=detail,
                data={
                    "regions_occupied": occupied,
                    "effective_regions": effective,
                    "top_region_share": float(lead.get("share", 0)) if lead else None,
                },
            )
        ]


@register
class RedundantRegion(Check):
    check_id = "T1-ATLAS-002"
    title = "One crowded area holds near-identical records"
    tier = 1
    unit = "map area"
    profiles = ALL_PROFILES
    requires = (Requirement.ATLAS,)
    cost = CostClass.EMBEDDING
    severity = Severity.WARNING
    fix = (
        "Inspect that region's examples; if they are one template, sample it down "
        "rather than deduplicating."
    )
    rationale = (
        "Near-duplicate detection works on shingle overlap, so it finds records that share "
        "wording. It does not find a thousand records generated from one template with the "
        "nouns swapped, because those share almost no shingles. In embedding space they are "
        "obvious: they pile into a single region and sit very close to each other. This check "
        "fires only when both are true, so a genuinely large topic with varied writing does "
        "not trip it."
    )

    def finalize(self, ctx: ScanContext) -> list[Finding]:
        cov = _coverage(ctx)
        if cov is None:
            return []
        cohesion = ctx.stats.get("atlas_region_cohesion") or {}
        if not cohesion:
            return []
        offenders = []
        crowded = crowded_region_share(int(cov.get("regions_total", 0)))
        for entry in cov.get("top_regions") or []:
            region = int(entry.get("region", -1))
            share = float(entry.get("share", 0.0))
            coh = cohesion.get(region)
            if coh is None or share < crowded:
                continue
            if coh >= CROWDED_REGION_COHESION:
                offenders.append((region, share, float(coh), entry.get("terms", "")))
        if not offenders:
            return []
        region, share, coh, _terms = offenders[0]
        return [
            make_finding(
                self,
                count=len(offenders),
                total=int(cov.get("regions_occupied", 0)),
                detail=(
                    f"one area of the map holds {share:.0%} of placed records, and those "
                    f"records average {coh:.2f} cosine similarity to each other — that is "
                    f"one thing written out many times rather than one subject covered "
                    f"many ways, and shingle-based duplicate detection cannot see it"
                ),
                data={
                    "regions": [
                        {"region": r, "share": s, "cohesion": c, "terms": t}
                        for r, s, c, t in offenders
                    ]
                },
            )
        ]
