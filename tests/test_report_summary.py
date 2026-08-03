"""The reading of a scan that both reports render.

What is tested here is editorial, not arithmetic: the order things are said in,
and the two ways the previous report misled a reader who was not the author.

A finding that counts datasets and a finding that counts records both produce a
percentage, and the two mean entirely different things. Sorting them together
put "1 of 1 datasets has no licence" above "49,872 of 50,000 records are
duplicates", because both read as a hundred percent.

And the atlas's own words for a region are not a description of the data. They
are the most frequent words among the first hundred and fifty reference records
that landed there, roughly forty percent of that text is function words shared
with other regions, and one level-0 area named for religion contains Arabic
Wikipedia. A finding ending in "(such, used, other, also, some)" is noise
presented as insight.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from dropoutt.models import Severity
from dropoutt.report.phrasing import in_words
from dropoutt.report.summary import Problem


def _problem(**kwargs) -> Problem:
    base = {
        "check_id": "T0-X-001", "title": "Something", "severity": Severity.WARNING,
        "is_blocking": False, "would_block": (), "affected": 1, "considered": 1,
        "share": 1.0, "tokens": None, "detail": "", "fix": "",
    }
    base.update(kwargs)
    return Problem(**base)


def test_dataset_counts_do_not_outrank_record_counts():
    licence = _problem(
        check_id="T1-LIC-001", title="Datasets have no recorded licence",
        affected=1, considered=1, share=1.0, unit="dataset", total_unit="dataset",
    )
    duplicates = _problem(
        check_id="T0-DUP-001", title="The same record appears more than once",
        affected=49_872, considered=50_000, share=0.997,
    )
    assert sorted([licence, duplicates], key=lambda p: p.rank)[0] is duplicates
    assert not licence.about_records
    assert duplicates.about_records


def test_scale_reads_in_the_unit_that_was_counted():
    assert _problem(
        affected=4, considered=4, share=1.0, unit="dataset", total_unit="dataset"
    ).scale == "4 of 4 datasets"
    assert _problem(
        affected=15_523, considered=50_000, share=0.31, unit="response",
        total_unit="response",
    ).scale == "15,523 of 50,000 responses · 31%"
    assert _problem(
        check_id="T1-DUP-002", affected=25, considered=48_527, share=0.0005,
        unit="prompt", total_unit="record",
    ).scale == "25 prompts across 48,527 records"


def test_consequence_orders_the_list_and_size_orders_within_a_tier():
    """What would stop a run leads, however few records it touches.

    This inverts the earlier rule, under which a large warning was promoted past
    a small would-block finding. The arithmetic behind that was defensible and
    the page it produced was not: a badge reading "warning" above five badged
    "would block" reads as an ordering bug, and a reader who thinks the order is
    broken stops using it to decide what to read.
    """
    rare = _problem(severity=Severity.BLOCKING, would_block=("sft",), affected=3,
                    considered=50_000, share=0.00006)
    common = _problem(severity=Severity.WARNING, affected=16_000, considered=50_000,
                      share=0.32)
    assert sorted([rare, common], key=lambda p: p.rank)[0] is rare

    # Within one tier, size still decides.
    small = _problem(severity=Severity.WARNING, affected=40, considered=50_000,
                     share=0.0008)
    assert sorted([small, common], key=lambda p: p.rank)[0] is common


def test_what_actually_blocks_the_run_leads_regardless_of_size():
    blocking = _problem(severity=Severity.BLOCKING, is_blocking=True,
                        would_block=("sft",), affected=59, considered=2_000,
                        share=0.03)
    common = _problem(severity=Severity.WARNING, affected=1_894, considered=1_930,
                      share=0.98)
    assert sorted([common, blocking], key=lambda p: p.rank)[0] is blocking


def test_shares_are_said_as_ratios():
    assert in_words(0.97) == "nearly every record"
    assert in_words(0.31) == "one record in three"
    assert in_words(0.001) == "a small number of records"


def _corpus(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    lines = []
    for i in range(400):
        lines.append(json.dumps({"messages": [
            {"role": "user", "content": f"Explain subject {i % 17} carefully."},
            {"role": "assistant", "content":
                "A long enough answer to be placed on the atlas and to carry some "
                "shape of its own, repeated a little so the checks have work. " * 3},
        ]}))
    (root / "train.jsonl").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture
def scanned(tmp_path):
    from dropoutt.atlas import load_bundled
    from dropoutt.langid import LanguageDetector
    from dropoutt.runner import scan

    return scan(str(_corpus(tmp_path)), detector=LanguageDetector(),
                atlas=load_bundled(), offline=True)


def test_atlas_places_are_named_by_the_readers_own_records(scanned):
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    if story is None or not story.available:
        pytest.skip("atlas coverage unavailable in this environment")
    assert story.places
    # Every place carries the record of the reader's that sits closest to it.
    # The atlas caption may be shown beside that, never instead of it.
    assert any(place.yours for place in story.places)


def test_atlas_findings_do_not_quote_region_captions(scanned):
    """The captions are known to be about forty percent function words."""
    from dropoutt.atlas import load_bundled

    atlas = load_bundled()
    if atlas is None:
        pytest.skip("no bundled atlas")
    captions = {
        term.strip()
        for caption in atlas.region_terms
        for term in caption.split(",")
        if len(term.strip()) > 3
    }
    for finding in scanned.findings:
        if not finding.check_id.startswith("T1-ATLAS-"):
            continue
        quoted = [c for c in captions if f"({c}," in finding.detail]
        assert not quoted, f"{finding.check_id} quotes atlas captions: {quoted}"


def test_every_subject_bar_is_weighed_against_the_maps_own_share(scanned):
    """A share of your corpus means nothing without the map's share beside it.

    "27% code generation" is a fact the reader cannot act on. "27%, where the
    map spends 4% of itself" is the finding.
    """
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    if story is None or not story.categories:
        pytest.skip("atlas coverage unavailable in this environment")
    assert all("map_share" in row for row in story.categories)
    assert sum(row["map_share"] for row in story.categories) > 0


def test_insights_are_gated_on_size_as_well_as_certainty(scanned):
    """No insight may fire on a difference too small to act on.

    The gates exist because the two failure modes pull opposite ways: on a large
    corpus everything is significant, and on a small sample everything looks
    dramatic. An insight has to clear both.
    """
    from dropoutt.report.atlas_story import (
        OVER_LIFT,
        OVER_SHARE,
        UNDER_LIFT,
        UNDER_MAP_SHARE,
    )
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    if story is None or not story.available:
        pytest.skip("atlas coverage unavailable in this environment")
    by_area = {row["name"]: row for row in story.categories}
    for insight in story.insights:
        assert insight.headline and insight.detail
        if insight.kind == "over":
            row = next(r for name, r in by_area.items() if name in insight.headline)
            assert row["share"] >= OVER_SHARE
            assert row["share"] / row["map_share"] >= OVER_LIFT
        if insight.kind == "under":
            assert UNDER_MAP_SHARE <= 1.0 and UNDER_LIFT < 1.0

    # Grouped by kind, largest first within a group. Sorting the whole list by
    # magnitude would rank a share of the corpus against a share of the map.
    from dropoutt.report.atlas_story import INSIGHT_ORDER, INSIGHTS_PER_KIND

    seen = [INSIGHT_ORDER.index(i.kind) for i in story.insights]
    assert seen == sorted(seen)
    for kind in ("over", "under"):
        group = [i.magnitude for i in story.insights if i.kind == kind]
        assert len(group) <= INSIGHTS_PER_KIND
        assert group == sorted(group, reverse=True)


def test_the_density_grid_is_anchored_at_parity_not_stretched_to_fit():
    """One is one, in every report ever produced.

    The scale exists so that two reports can be held side by side, which they
    cannot be if the middle of the ramp means "the middle of this corpus".
    Empty is 0 and the corpus's own densest cell is 4 — but a cell as dense as
    the map is 1 whatever else the corpus contains.
    """
    from dropoutt.report.atlas_story import MAX_LEVEL, _level

    assert _level(0.0, 50.0) == 0.0
    assert _level(1.0, 50.0) == 1.0
    assert _level(50.0, 50.0) == pytest.approx(MAX_LEVEL)
    # Above parity the scale is logarithmic, so one runaway cell cannot flatten
    # everything else onto the parity colour.
    assert _level(2.0, 4.0) > _level(2.0, 400.0) > 1.0

    # Below parity the curve is fixed rather than normalised, and it is pulled
    # towards white: a cell at 0.02x is one record where the map has fifty, and
    # a linear ramp painted it a comfortable-looking green.
    assert _level(0.4, 50.0) == _level(0.4, 3.0), "sub-parity must not normalise"
    assert _level(0.4, 50.0) < 0.4
    assert _level(0.02, 50.0) < 0.01, "a near-empty cell must read as near-empty"
    assert _level(0.9, 0.9) == pytest.approx(0.9 ** 1.5)


def test_a_reached_cell_at_the_bottom_of_the_scale_stays_on_step_zero():
    """Thin coverage uses step 0 with a fill, not a hard green band.

    Step 0 without a fill is an absence — a hatched box with an ``N`` — but
    with one it is near-white, which is what 0.01× should look like. Mid-low
    ratios climb the ramp rather than staying white.
    """
    from dropoutt.report.atlas_story import MAX_LEVEL, Cell, _level
    from dropoutt.report.theme import RAMP_DIVISOR

    invisible = Cell(region=0, records=1, ratio=0.0001, level=0.0001, caption="")
    assert invisible.step == 0
    assert Cell(region=0, records=0, ratio=0.0, level=0.0, caption="").step == 0
    mid = Cell(
        region=0, records=1, ratio=0.34, level=_level(0.34, 9.3), caption=""
    )
    assert mid.step > 0, "0.34× must tint, not stay pure white"
    top = Cell(region=0, records=9, ratio=99.0, level=MAX_LEVEL, caption="")
    assert top.step == int(MAX_LEVEL * RAMP_DIVISOR)


def test_cell_ink_follows_luminance_contrast():
    """Ink is black or white by which contrast wins against the fill."""
    from dropoutt.report.theme import (
        INK_DARK,
        INK_LIGHT,
        RAMP_DIVISOR,
        RAMP_STEPS,
        _fill,
        _ink,
        _luminance,
    )

    for step in range(RAMP_STEPS + 1):
        fill = _fill(step / RAMP_DIVISOR)
        ink, _ = _ink(fill)
        luminance = _luminance(fill)
        black = (luminance + 0.05) / 0.05
        white = 1.05 / (luminance + 0.05)
        expect = INK_DARK if black >= white else INK_LIGHT
        assert ink == expect, f"step d{step} chose {ink}, expected {expect}"


def test_the_grid_draws_the_whole_map_densest_first(scanned):
    """The areas a corpus never reaches are half of what the picture says.

    A grid of only the occupied areas answers "what is my data made of", which
    the section above it already answered. Drawing the empty rows is the whole
    reason it is a map rather than a second bar chart.
    """
    from dropoutt.atlas import load_bundled
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    atlas = load_bundled()
    if story is None or not story.available or atlas is None or not story.grid:
        pytest.skip("atlas coverage unavailable in this environment")

    assert len(story.grid) == atlas.n_l1
    assert sum(len(area.cells) for area in story.grid) == atlas.n_regions
    assert any(not area.records for area in story.grid), "empty areas must be drawn"

    shares = [area.share for area in story.grid]
    assert shares == sorted(shares, reverse=True)


def test_the_place_lists_are_ranked_by_density_not_by_record_count(scanned):
    """Where a corpus stands out, not where the map happens to be large.

    Ranking these by record count asked "where is most of my data", which the
    section above already answers — and answers mostly with a fact about the
    map, since a region the reference corpus spent forty cells on will hold
    more of anybody's data than one it spent four on.
    """
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    if story is None or not story.places:
        pytest.skip("atlas coverage unavailable in this environment")

    dense = [place.ratio for place in story.places]
    assert dense == sorted(dense, reverse=True)
    assert all(place.ratio > 0 for place in story.places)
    if story.thin_places:
        thin = [place.ratio for place in story.thin_places]
        assert thin == sorted(thin)
        # The two ends of one ordering, and they may not overlap.
        assert not ({p.region for p in story.places}
                    & {p.region for p in story.thin_places})
        assert min(dense) >= max(thin)

    # The share-based sentences keep their own handle on the biggest place,
    # which is a different region whenever the map is unevenly divided.
    if story.dominant is not None:
        assert story.dominant.share >= max(p.share for p in story.thin_places or
                                           story.places)


def test_every_square_carries_its_own_ratio(scanned):
    """The number has to survive paper, a screenshot, and a keyboard.

    It was a hover tooltip, which meant it existed only while a mouse was over
    it — and the page is read printed, pasted into tickets, and by people not
    using a pointer at all.
    """
    from dropoutt.report.summary import build

    story = build(scanned).atlas
    if story is None or not story.grid:
        pytest.skip("atlas coverage unavailable in this environment")

    for area in story.grid:
        for cell in area.cells:
            if cell.records:
                assert cell.label.endswith("×")
                assert "the map" in cell.described
            else:
                # An empty box is ambiguous between "nothing here" and
                # "nothing rendered", so an unreached cell says so.
                assert cell.label == "N"
                assert cell.described == "not reached"


def test_the_atlas_samples_far_deeper_than_the_tokenizer_panel(tmp_path):
    """The two samples are one bottom-k cut at two depths.

    They used to share a 20,000 record ceiling, which put a 272,000 record
    corpus on the map with under seven thousand of its records — few enough
    that a subject area holding two percent of the corpus was decided by a
    hundred of them. Placement is one encoder pass per record; pricing is five
    real tokenizers over the same text, and the quantity it estimates converges
    long before a histogram over 212 cells does. So only the atlas went deep.
    """
    from dropoutt import runner
    from dropoutt.runner import (
        ATLAS_SAMPLE_TARGET,
        BUDGET_SAMPLE_TARGET,
        _per_dataset,
        scan,
    )

    assert ATLAS_SAMPLE_TARGET >= 200_000
    assert BUDGET_SAMPLE_TARGET < ATLAS_SAMPLE_TARGET
    assert _per_dataset(ATLAS_SAMPLE_TARGET, 1) == ATLAS_SAMPLE_TARGET
    assert _per_dataset(ATLAS_SAMPLE_TARGET, 40) == 5_000
    # The floor, so a corpus of very many small datasets is not sampled a
    # handful of records deep in each and called coverage.
    assert _per_dataset(ATLAS_SAMPLE_TARGET, 100_000) == 200

    root = tmp_path / "data"
    root.mkdir()
    (root / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"record {i} written out at a readable length"})
            for i in range(300)
        ) + "\n",
        encoding="utf-8",
    )

    original = runner._per_dataset
    runner._per_dataset = lambda target, datasets: (
        40 if target == BUDGET_SAMPLE_TARGET else 150
    )
    try:
        result = scan(str(tmp_path), offline=True)
    finally:
        runner._per_dataset = original

    # The budget stops at its own depth; the sample behind it went further.
    assert sum(len(v) for v in result.ctx.stats["budget_sample"].values()) == 40


def test_significance_is_measured_against_records_placed():
    """The gate is a binomial standard error on the expected share.

    Small *differences* are what it exists to reject. It is deliberately not the
    thing that rejects small *samples* — see the test below for that, because a
    binomial test is happy to call a 4x difference on thirty records real, and
    on this data it should not be believed.
    """
    from dropoutt.report.atlas_story import _significant

    assert not _significant(0.11, 0.10, 300)
    assert _significant(0.40, 0.10, 300)
    assert not _significant(0.10, 0.10, 30_000)


def test_no_comparison_against_the_map_below_the_sample_floor():
    """Arithmetic cannot see that thirty placed records is not a corpus.

    Placement is a nearest centroid over a probe that is 86% accurate on
    held-out reference data. A handful of records clears the binomial gate and
    still says nothing, so the floor is enforced separately.
    """
    from dropoutt.report.atlas_story import (
        MIN_PLACED_FOR_INSIGHT,
        AtlasStory,
        _insights,
    )

    class _Result:
        class ctx:
            atlas = None
            stats: ClassVar[dict] = {}

    story = AtlasStory(available=True, placed=MIN_PLACED_FOR_INSIGHT - 1)
    kinds = {i.kind for i in _insights(_Result(), {}, story)}
    assert not (kinds & {"over", "under"})


# --------------------------------------------------------------------------
# Token budget
# --------------------------------------------------------------------------


def _budget_corpus():
    """Two datasets with genuinely different tokens-per-character.

    ``wide`` records are ordinary words; ``dense`` records are unbroken strings,
    which every BPE tokenizer splits far more finely. The sample cap gives both
    the same number of records while one of them is most of the corpus.
    """
    wide = ["the quick brown fox jumps over the lazy dog " * 6 for _ in range(400)]
    dense = ["zxqvkwjhgfdmbptrslncy" * 12 for _ in range(400)]
    return wide, dense


def test_budget_prices_each_dataset_at_its_own_rate():
    """A pooled ratio prices the corpus at the sample's blend, not the corpus's.

    The regression this guards is worth 12-38% on a real mixed corpus: the
    per-dataset sample cap makes a small dataset as loud as a huge one, so a
    single pooled tokens-per-character ratio charges the whole corpus at a blend
    it never had.
    """
    from dropoutt.tokenizer_panel import HAVE_TOKENIZERS, estimate_budget

    wide, dense = _budget_corpus()
    # The corpus is overwhelmingly `wide`; the sample is half and half.
    chars = {"wide": sum(len(t) for t in wide) * 100, "dense": sum(len(t) for t in dense)}
    total = sum(chars.values())
    report = estimate_budget(
        {"wide": wide, "dense": dense}, total, 0,
        chars_by_dataset=chars,
        records_by_dataset={"wide": len(wide) * 100, "dense": len(dense)},
    )
    assert report.estimates
    if not HAVE_TOKENIZERS:
        return

    est = report.cheapest
    assert est is not None
    # The blended rate over the pooled sample is what the old code used. The
    # stratified estimate has to sit near `wide`'s own rate instead, because
    # `wide` is 99% of the corpus.
    pooled_rate = _pooled_rate(wide + dense)
    wide_rate = _pooled_rate(wide)
    assert abs(est.tokens_per_char - wide_rate) < abs(est.tokens_per_char - pooled_rate)


def _pooled_rate(texts):
    from dropoutt.tokenizer_panel import PANEL, load_tokenizer

    handle = load_tokenizer(PANEL[0][1])
    counts = handle.count_batch(texts)
    return sum(counts) / sum(len(t) for t in texts)


def test_a_dataset_sampled_whole_contributes_no_uncertainty():
    """The finite-population correction is not decoration.

    When every record of a dataset is in the sample, its token count has been
    counted rather than estimated, and an interval that still widened for it
    would be reporting sampling error that does not exist.
    """
    from dropoutt.tokenizer_panel import HAVE_TOKENIZERS, estimate_budget

    if not HAVE_TOKENIZERS:
        return
    wide, _ = _budget_corpus()
    chars = {"wide": sum(len(t) for t in wide)}
    report = estimate_budget(
        {"wide": wide}, chars["wide"], 0,
        chars_by_dataset=chars, records_by_dataset={"wide": len(wide)},
    )
    assert report.cheapest.margin == 0


def test_budget_accepts_a_bare_list_as_one_dataset():
    from dropoutt.tokenizer_panel import estimate_budget

    wide, _ = _budget_corpus()
    report = estimate_budget(wide, sum(len(t) for t in wide), 0)
    assert report.sample_size > 0
    assert report.covered_share == 1.0
