"""Atlas behaviour and report safety.

The atlas tests build a tiny synthetic artifact in-memory and load it through
the real loader, so they exercise the actual code path without a network call.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from dropoutt.atlas.apply import Atlas
from dropoutt.report.escaping import json_script_payload, safe_snippet, visible_controls
from dropoutt.report.terminal import _m


@pytest.fixture
def tiny_atlas(tmp_path):
    """Three categories, four regions each, eight dimensions, known geometry."""
    rng = np.random.default_rng(0)
    dim, n_regions = 8, 12
    centroids = rng.normal(size=(n_regions, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    path = tmp_path / "tiny.npz"
    np.savez_compressed(
        path,
        centroids=centroids,
        region_category=np.repeat([0, 1, 2], 4).astype(np.int32),
        region_size=np.full(n_regions, 100.0, dtype=np.float32),
        coords=rng.normal(size=(n_regions, 2)).astype(np.float32),
        probe_coef=rng.normal(size=(3, dim)).astype(np.float32),
        probe_intercept=np.zeros(3, dtype=np.float32),
        probe_classes=np.array([0, 1, 2], dtype=np.int32),
        meta=np.array([json.dumps({
            "version": "tiny-test", "embed_model": "fake", "off_atlas_threshold": 0.90,
            "n_reference_records": 100, "l0_holdout_accuracy": 0.9,
            "region_purity_by_taxonomy": 0.8,
            "region_terms": [f"region {i}" for i in range(n_regions)],
        })], dtype=object),
        allow_pickle=True,
    )
    return Atlas.load(path)


def test_records_near_a_centroid_are_assigned_to_it(tiny_atlas):
    target = 5
    emb = np.tile(tiny_atlas.centroids[target], (10, 1))
    regions, scores = tiny_atlas.assign(emb)
    assert (regions == target).all()
    assert (scores > 0.99).all()


def test_distant_records_are_off_atlas(tiny_atlas):
    """Off-atlas is a real state, not a nearest-neighbour fallback."""
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(50, tiny_atlas.dim)).astype(np.float32)
    regions, _ = tiny_atlas.assign(emb)
    assert (regions == -1).sum() > 0


def test_a_high_off_atlas_rate_does_not_withhold_the_histogram(tiny_atlas):
    """The placed records stay measured however many others failed to place.

    Off-atlas records are filtered out of the histogram before it is counted, so
    the histogram was never contaminated by them and withholding it discarded a
    correct measurement. This pins the replacement of that behaviour.
    """
    # 60 placed, 140 off-atlas: a 70% rate, far above anything the old
    # suppression rule tolerated.
    regions = np.concatenate([
        np.array([1, 2, 3] * 20, dtype=np.int32),
        np.full(140, -1, dtype=np.int32),
    ])
    categories = np.zeros(200, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, categories)

    assert cov["status"] == "ok"
    assert cov["off_atlas_rate"] == 0.7
    assert cov["fit"] == "poor"
    assert cov["placed"] == 60
    assert cov["regions_occupied"] == 3
    assert sum(cov["region_counts"].values()) == 60
    # Category counts share the region histogram's denominator. Counting them
    # over all 200 records put two denominators in one panel.
    assert sum(cov["by_category"].values()) == 60


def test_coverage_describes_the_corpus_not_the_average_of_its_datasets(tiny_atlas):
    """The sample is equal per dataset; the histogram must not be.

    A dataset of a million records and one of a thousand are sampled to the
    same size, so an unweighted histogram gives them equal say. Weighting each
    sampled record by how many corpus records it stands for is what turns the
    sample back into a description of the corpus.
    """
    # Fifty records land in region 1 and fifty in region 2. The first fifty are
    # a 1000x sample of a huge dataset; the second fifty are all of a small one.
    regions = np.array([1] * 50 + [2] * 50, dtype=np.int32)
    categories = np.zeros(100, dtype=np.int32)

    flat = tiny_atlas.coverage(regions, categories)
    assert flat["region_counts"]["1"] == flat["region_counts"]["2"]

    weighted = tiny_atlas.coverage(
        regions, categories, weights=[1000.0] * 50 + [1.0] * 50
    )
    assert weighted["region_counts"]["1"] == 50_000
    assert weighted["region_counts"]["2"] == 50
    assert weighted["placed_estimated"] == 50_050
    # The sample counts stay sample counts: they are the honest denominator for
    # "how much did we actually look at".
    assert weighted["placed"] == 100
    # And the shape of the corpus follows the weighted histogram, so the corpus
    # now reads as concentrated rather than as evenly split.
    assert weighted["effective_regions"] < flat["effective_regions"]


def test_a_weighted_cell_never_rounds_away_a_record_that_is_there(tiny_atlas):
    """A weight is at least one — a record stands for itself at minimum."""
    regions = np.array([3] * 4, dtype=np.int32)
    cov = tiny_atlas.coverage(
        regions, np.zeros(4, dtype=np.int32), weights=[1.0, 1.0, 1.0, 1.0]
    )
    assert cov["region_counts"]["3"] == 4


def test_the_density_of_each_cell_travels_with_the_coverage_facet():
    """A number a reader can see has to be a number a script can read.

    The report draws each cell against the reference corpus's own density
    there. Deriving that only at render time would mean the one comparison the
    atlas exists to make could not be asserted on in CI without loading the
    artifact and dividing by `region_size` by hand.
    """
    from dropoutt.atlas import load_bundled

    atlas = load_bundled()
    if atlas is None or atlas.region_size is None:
        pytest.skip("bundled atlas carries no reference sizes")

    regions = np.array([1] * 80 + [2] * 20, dtype=np.int32)
    cov = atlas.coverage(regions, np.zeros(100, dtype=np.int32))

    assert set(cov["region_density"]) == set(cov["region_counts"])
    # Four times the records in cell 1, so its density has to come out higher.
    # The two are not in exactly a 4:1 ratio, because each is shrunk towards
    # the map by its own expected count — see the tests below.
    assert cov["region_density"]["1"] > cov["region_density"]["2"]
    assert cov["density_model"]["effective_sample"] == 100


def test_a_cells_density_is_shrunk_by_how_much_evidence_is_behind_it():
    """The same observation means different things at different sample sizes.

    One record where forty were due is strong evidence of under-coverage. The
    same one record where half of one was due is nothing at all. A raw quotient
    reports both as a confident number, and on a small scan the whole grid is
    the second case.
    """
    from dropoutt.atlas import load_bundled

    atlas = load_bundled()
    if atlas is None or atlas.region_size is None:
        pytest.skip("bundled atlas carries no reference sizes")
    share = np.asarray(atlas.region_size, dtype=float)
    share = share / share.sum()
    rng = np.random.default_rng(0)

    def scan(n, p):
        regions = rng.choice(len(share), size=n, p=p).astype(np.int32)
        return atlas.coverage(regions, np.zeros(n, dtype=np.int32))

    # A corpus drawn from the map itself has nothing to report, and every cell
    # comes back at parity rather than at whatever the sampling noise was.
    same = scan(9_000, share)
    assert all(abs(v - 1.0) < 0.05 for v in same["region_density"].values())
    assert same["density_model"]["prior_strength"] >= 1e6

    # Lopsided and well sampled: the signal survives essentially untouched.
    lopsided = (share ** 4) / (share ** 4).sum()
    big = scan(9_000, lopsided)
    counts = {int(k): int(v) for k, v in big["region_counts"].items()}
    thin = min(counts, key=lambda r: counts[r])
    raw = (counts[thin] / 9_000) / share[thin]
    assert big["region_density"][str(thin)] < 0.5
    assert raw < big["region_density"][str(thin)], "shrinkage only ever pulls inward"

    # The same corpus at a hundredth of the sample: now a single record is not
    # evidence of anything, and the cell reads as parity instead of as a claim.
    small = scan(95, lopsided)
    counts = {int(k): int(v) for k, v in small["region_counts"].items()}
    singles = [r for r, c in counts.items() if c == 1]
    assert singles, "fixture must produce at least one single-record cell"
    for cell in singles:
        assert abs(small["region_density"][str(cell)] - 1.0) < 0.45


def test_the_effective_sample_is_what_the_weights_leave_behind():
    """A record standing for five thousand others is still one observation.

    Every gate that asks "could this be noise" takes this number. Handing it
    the weighted total would claim certainty proportional to the corpus rather
    than to what was actually read.
    """
    from dropoutt.atlas import load_bundled

    atlas = load_bundled()
    if atlas is None:
        pytest.skip("no bundled atlas")
    regions = np.array([1] * 50 + [2] * 50, dtype=np.int32)
    cats = np.zeros(100, dtype=np.int32)

    flat = atlas.coverage(regions, cats, weights=[1.0] * 100)
    assert flat["density_model"]["effective_sample"] == 100

    # Fifty records carrying a thousand each, beside fifty carrying one: a
    # hundred records read, nowhere near a hundred records' worth of evidence.
    skewed = atlas.coverage(regions, cats, weights=[1000.0] * 50 + [1.0] * 50)
    assert skewed["density_model"]["effective_sample"] < 55
    assert skewed["placed"] == 100


def test_a_legacy_artifact_without_reference_sizes_says_nothing_rather_than_guessing(
    tiny_atlas,
):
    """The divisor is the reference distribution, and older artifacts lack it."""
    tiny_atlas.region_size = None
    cov = tiny_atlas.coverage(
        np.array([1] * 10, dtype=np.int32), np.zeros(10, dtype=np.int32)
    )
    assert cov["region_density"] == {}


def test_coverage_reports_none_placed_only_when_nothing_placed(tiny_atlas):
    """The one case where the numbers genuinely do not exist."""
    regions = np.full(200, -1, dtype=np.int32)
    categories = np.zeros(200, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, categories)
    assert cov["status"] == "none placed"
    assert cov["off_atlas_rate"] == 1.0
    assert "region_counts" not in cov


def test_off_atlas_records_are_described_not_merely_counted(tiny_atlas):
    """A count says how much failed to place; the description says what and why."""
    rng = np.random.default_rng(3)
    placed = np.tile(tiny_atlas.centroids[4], (60, 1)).astype(np.float32)
    stray = rng.normal(size=(40, tiny_atlas.dim)).astype(np.float32)
    emb = np.vstack([placed, stray])
    regions, scores, nearest = tiny_atlas.assign_full(emb)
    assert (regions < 0).sum() == 40, "fixture must produce exactly the stray rows"

    cov = tiny_atlas.coverage(
        regions, np.zeros(100, dtype=np.int32),
        ["en"] * 60 + ["tr"] * 40,
        scores=scores, nearest=nearest, embeddings=emb,
        lengths=[900] * 60 + [90] * 40,
        datasets=["big"] * 60 + ["small"] * 40,
    )
    detail = cov["off_atlas_detail"]

    assert detail["score"]["off_median"] < detail["score"]["placed_median"]
    assert detail["length"]["off_median_chars"] == 90
    assert detail["length"]["placed_median_chars"] == 900
    assert "short records" in detail["diagnosis"]
    # Every off-atlas record still has a nearest region. `assign` replaces it
    # with -1; the description exists to give it back.
    # `nearest_regions` is a top-six head, so it accounts for some but not
    # necessarily all of the off-atlas records; `nearest_region_spread` carries
    # the rest of the story.
    assert detail["nearest_regions"]
    assert 0 < sum(r["records"] for r in detail["nearest_regions"]) <= 40
    assert detail["nearest_region_spread"] >= len(detail["nearest_regions"])
    assert detail["by_dataset"]["small"]["rate"] == 1.0
    assert "big" not in detail["by_dataset"]


def _coherent_off_atlas_fixture(atlas):
    """60 placed rows, plus 40 that are far from every centroid and near each other."""
    rng = np.random.default_rng(11)
    placed = np.repeat(atlas.centroids[:6], 10, axis=0).astype(np.float32)
    axis = -atlas.centroids[0].astype(np.float32)
    stray = (np.tile(axis, (40, 1))
             + rng.normal(scale=0.01, size=(40, atlas.dim))).astype(np.float32)
    emb = np.vstack([placed, stray])
    regions, scores, nearest = atlas.assign_full(emb)
    assert (regions < 0).sum() == 40, "fixture must produce exactly the stray rows"
    return emb, regions, scores, nearest


def test_a_coherent_off_atlas_set_that_reads_as_prose_is_reported_as_one_thing(
    tiny_atlas,
):
    """Alike plus prose-like surface is as far as the measurement goes.

    The claim stops at "one kind of thing". It deliberately does not assert a
    missing subject area, because coherence cannot support that — see the
    template test below.
    """
    emb, regions, scores, nearest = _coherent_off_atlas_fixture(tiny_atlas)
    prose = "the committee approved the proposal after a lengthy debate on funding "

    cov = tiny_atlas.coverage(
        regions, np.zeros(len(emb), dtype=np.int32), None,
        scores=scores, nearest=nearest, embeddings=emb,
        # Equal lengths and identical surface on both sides, so only the
        # coherence branch can fire.
        lengths=[500] * len(emb), texts=[prose] * len(emb),
    )
    detail = cov["off_atlas_detail"]
    assert detail["coherence"]["off"] > detail["coherence"]["placed"]
    assert "one kind of thing" in detail["diagnosis"]
    assert "missing" not in detail["diagnosis"], (
        "coherence does not identify a missing subject area; templates score higher "
        "than prose on it"
    )


def test_a_coherent_off_atlas_set_of_machine_format_is_not_called_a_missing_topic(
    tiny_atlas,
):
    """The regression this test exists for.

    Minified JS scores 0.969 mean pairwise cosine against 0.277 for real prose,
    so a coherence-only rule labels every template a missing subject area. The
    surface test has to win over the coherence test.
    """
    emb, regions, scores, nearest = _coherent_off_atlas_fixture(tiny_atlas)
    prose = "the committee approved the proposal after a lengthy debate on funding "
    minified = "a.b.c=function(g){return h(g)};for(i=0;i<n;i++){e[i]=f}if(!c){d()}"

    cov = tiny_atlas.coverage(
        regions, np.zeros(len(emb), dtype=np.int32), None,
        scores=scores, nearest=nearest, embeddings=emb,
        lengths=[500] * len(emb),
        texts=[prose] * 60 + [minified] * 40,
    )
    detail = cov["off_atlas_detail"]
    # Still highly coherent, so the coherence branch would have fired.
    assert detail["coherence"]["off"] > detail["coherence"]["placed"]
    assert detail["surface"]["off_whitespace"] < detail["surface"]["placed_whitespace"]
    assert "not written like prose" in detail["diagnosis"]
    assert "one kind of thing" not in detail["diagnosis"]


def test_mean_pairwise_matches_the_naive_matrix(tiny_atlas):
    """The linear-time identity must equal the quadratic definition exactly."""
    from dropoutt.atlas.apply import _mean_pairwise

    rng = np.random.default_rng(5)
    v = rng.normal(size=(50, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    naive = (sim.sum() - np.trace(sim)) / (len(v) * (len(v) - 1))
    assert _mean_pairwise(v) == pytest.approx(float(naive), abs=1e-4)


def test_coverage_is_reported_when_records_land_on_the_map(tiny_atlas):
    regions = np.array([1, 1, 2, 3, 3, 3, 5] * 20, dtype=np.int32)
    categories = np.array([0, 0, 0, 0, 1, 1, 1] * 20, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, categories)
    assert cov["status"] == "ok"
    assert cov["regions_occupied"] == 4
    assert cov["regions_total"] == tiny_atlas.n_regions
    # Keys are strings: numpy int keys are refused by orjson.
    assert cov["by_category"] == {"0": 80, "1": 60}


def test_off_atlas_rate_is_reported_per_language(tiny_atlas):
    """A global average would hide a language the embedder represents poorly."""
    regions = np.array([1] * 100 + [-1] * 100, dtype=np.int32)
    categories = np.zeros(200, dtype=np.int32)
    langs = ["en"] * 100 + ["ota"] * 100
    cov = tiny_atlas.coverage(regions, categories, langs)
    per = cov["per_language"]
    assert per["en"]["off_atlas_rate"] == 0.0 and per["en"]["reliable"]
    assert per["ota"]["off_atlas_rate"] == 1.0 and not per["ota"]["reliable"]


def test_atlas_quality_numbers_travel_with_the_artifact(tiny_atlas):
    """They must be printable next to any coverage figure."""
    cov = tiny_atlas.coverage(np.array([1] * 50), np.zeros(50, dtype=np.int32))
    assert cov["l0_holdout_accuracy"] == 0.9
    assert cov["region_purity_by_taxonomy"] == 0.8


# -- terminal rendering --------------------------------------------------


def test_install_hints_name_no_extras_and_survive_rich_markup():
    """No hint may name an extra, and every hint must reach the screen intact.

    Two regressions in one assertion, because the first caused the second.
    dropoutt used to ship extras, so hints read `pip install 'dropoutt[lid]'` —
    and rich reads `[lid]` as a style tag and silently deletes it, so the user
    was told to run `pip install 'dropoutt'`, which installs the package they
    already have and not the piece they were missing.

    There are no extras now: one install brings everything. So the hint must not
    invent one, and it must still render unchanged.
    """
    import io

    from rich.console import Console

    from dropoutt.compat import capability_report

    for name, info in capability_report().items():
        hint = str(info.get("install") or "")
        if not hint:
            continue
        assert "dropoutt[" not in hint, (
            f"{name}: install hint names an extra, and extras were removed in 1.1"
        )
        buf = io.StringIO()
        Console(file=buf, width=120, no_color=True).print(f"[dim]{_m(hint)}[/dim]")
        assert hint in buf.getvalue(), (
            f"{name}: the hint was altered by rich markup; "
            f"{hint!r} rendered as {buf.getvalue().strip()!r}"
        )


def test_the_terminal_is_triage_and_says_where_the_rest_went(tmp_path):
    """Short enough to read, and honest about what it left out.

    Everything it used to print in full is in a file written milliseconds
    earlier, so the screen answers "do I need to look" and then says where
    looking happens. A screen that omits the detail without naming the file
    that holds it is not brief, it is lossy.
    """
    import io

    from rich.console import Console

    from dropoutt.report import terminal as term_report
    from dropoutt.runner import scan

    data = tmp_path / "d"
    data.mkdir()
    (data / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"messages": [
                {"role": "human", "content": f"question {i} spelled out at length"},
                {"role": "gpt", "content": f"answer {i} spelled out at some length"},
            ]})
            for i in range(60)
        ) + "\n",
        encoding="utf-8",
    )
    result = scan(str(tmp_path))
    written = ["report.html", "report.md", "findings.jsonl", "fingerprint.json"]

    buf = io.StringIO()
    term_report.render(
        Console(file=buf, width=100, no_color=True), result,
        out_dir=str(tmp_path / "out"), written=written,
    )
    out = buf.getvalue()

    assert len(out.splitlines()) <= 40, "the terminal report has grown a wall again"
    for name in written:
        assert name in out, f"{name} was written and never mentioned"
    # The fix text and the excerpts belong to the files, not to the scroll-back.
    for problem in term_report.build(result).problems:
        assert problem.fix not in out


def test_dataset_names_cannot_inject_terminal_styling():
    """A folder named `[red]` must not restyle our own output."""
    import io

    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=120, no_color=True).print(f"    {_m('[red]evil')} plain")
    assert "[red]evil" in buf.getvalue()


# -- report safety -------------------------------------------------------


def test_control_characters_are_made_visible_not_stripped():
    """A tool that reports control characters must let you see them."""
    out = visible_controls("before\x00after\x07end")
    assert "\x00" not in out and "\x07" not in out
    assert "␀" in out and "␇" in out


def test_bidi_overrides_are_neutralised():
    """One unbalanced override would otherwise reverse the whole report."""
    out = safe_snippet("safe\u202etext\u202dmore")
    assert "\u202e" not in out and "\u202d" not in out


def test_script_payload_cannot_close_its_own_element():
    payload = json_script_payload({"text": "</script><img src=x onerror=alert(1)>"})
    assert "</script>" not in payload
    assert "\\u003c" in payload
    # Still valid JSON after escaping.
    assert json.loads(payload)["text"].startswith("</script>")


def test_snippets_are_length_bounded():
    assert len(safe_snippet("x" * 5000, limit=200)) <= 200


def test_html_report_does_not_leak_planted_secrets(tmp_path):
    """Values matched by the PII catalog must be masked before rendering."""
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.langid import LanguageDetector
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    secret = "sk-proj-SECRETSECRETSECRETSECRET1234"
    email = "planted.address@example-domain.com"
    data = tmp_path / "d"
    data.mkdir()
    with open(data / "train.jsonl", "w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": f"Iletisim bilgisi nedir {i}"},
                {"role": "assistant", "content": f"Bana {email} ya da {secret} ile ulasin"},
            ]}) + "\n")

    result = scan(str(tmp_path), detector=LanguageDetector())
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)
    page = html_report.render(result, fp, None)

    assert secret not in page, "an API key reached the shareable report"
    assert email not in page, "an email address reached the shareable report"
    assert "T1-PII-001" in page, "but the finding itself must still be reported"


def test_html_report_can_omit_all_record_evidence(tmp_path):
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.langid import LanguageDetector
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    planted = "private surrounding words that are not themselves PII"
    data = tmp_path / "d"
    data.mkdir()
    with open(data / "train.jsonl", "w", encoding="utf-8") as fh:
        for _ in range(8):
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": "Soru"},
                {"role": "assistant", "content": planted},
            ]}) + "\n")

    result = scan(str(tmp_path), detector=LanguageDetector())
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)
    evidence_before = sum(len(f.evidence) for f in result.findings)
    assert evidence_before
    page = html_report.render(result, fp, None, include_evidence=False)

    assert planted not in page
    assert "Record excerpts and source locations were omitted" in page
    assert sum(len(f.evidence) for f in result.findings) == evidence_before, (
        "rendering a redacted report must not mutate the scan result"
    )


def test_fingerprint_omits_contamination_witness_paths():
    from dropoutt.context import ScanContext
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.models import Confidence, Finding, Profile, Severity

    finding = Finding(
        check_id="T1-CONTAM-001",
        title="overlap",
        severity=Severity.WARNING,
        confidence=Confidence.UNVERIFIED,
        count=1,
        total_considered=1,
        detail="overlap",
        fix="remove it",
        data={
            "results": {
                "private-eval": {
                    "n_instances": 1,
                    "n_contaminated": 1,
                    "witnesses": [
                        {"instance": 0, "record": ("doc", "/secret/eval.jsonl", 2)}
                    ],
                },
            },
            "rule": "test",
        },
    )
    fp = build_fingerprint(
        ScanContext(root="/secret", profile=Profile.SFT),
        [finding],
        total_chars=0,
        total_words=0,
    )

    values = fp.facets["contamination"].values
    assert "witnesses" not in values["private-eval"]
    assert "/secret/eval.jsonl" not in json.dumps(fp.to_dict())


def test_off_atlas_excerpts_never_reach_the_fingerprint(tiny_atlas):
    """The fingerprint is the shareable artifact. Record text is not shareable.

    Off-atlas excerpts are the one part of the new coverage output that is raw
    corpus content, so they are held in scan stats and rendered from there rather
    than being written into the coverage facet.
    """
    from dropoutt.context import ScanContext
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.models import Profile

    ctx = ScanContext(root="/secret", profile=Profile.SFT)
    ctx.atlas = tiny_atlas
    ctx.stats["atlas_coverage"] = {
        "status": "ok", "records": 10, "placed": 8, "off_atlas": 2,
        "off_atlas_rate": 0.2, "fit": "partial",
        "region_counts": {"1": 8},
        "off_atlas_detail": {"diagnosis": "scattered", "score": {"off_median": 0.2}},
    }
    ctx.stats["atlas_off_examples"] = [
        {"score": 0.11, "excerpt": "PROPRIETARY-CUSTOMER-TEXT", "chars": 40,
         "dataset": "d", "language": "en", "nearest_region": 3},
    ]

    fp = build_fingerprint(ctx, [], total_chars=0, total_words=0)
    dumped = json.dumps(fp.to_dict())

    assert "PROPRIETARY-CUSTOMER-TEXT" not in dumped
    assert "atlas_off_examples" not in dumped
    # The statistics that describe those records must survive, though.
    assert fp.facets["coverage"].values["off_atlas_detail"]["diagnosis"] == "scattered"


def test_html_report_escapes_markup_from_the_corpus(tmp_path):
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.langid import LanguageDetector
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    data = tmp_path / "d"
    data.mkdir()
    payload = "<script>alert('xss')</script>"
    with open(data / "train.jsonl", "w", encoding="utf-8") as fh:
        for i in range(8):
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": f"Soru {i} {payload}"},
                {"role": "assistant", "content": ""},
            ]}) + "\n")

    result = scan(str(tmp_path), detector=LanguageDetector())
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)
    page = html_report.render(result, fp, None)
    assert "<script>alert" not in page


def test_html_report_describes_the_corpus_before_it_faults_it(tmp_path):
    from dropoutt.atlas import load_bundled
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.langid import LanguageDetector
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    data = tmp_path / "d"
    data.mkdir()
    (data / "train.jsonl").write_text(
        json.dumps({"text": "a sufficiently long atlas record for coverage"}) + "\n",
        encoding="utf-8",
    )
    result = scan(str(tmp_path), detector=LanguageDetector())
    atlas = load_bundled()
    assert atlas is not None
    result.ctx.stats["atlas_coverage"] = atlas.coverage(
        np.array([0, 0, 2, 4], dtype=np.int32),
        atlas.region_category[np.array([0, 0, 2, 4], dtype=np.int32)],
    )
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)

    page = html_report.render(result, fp, None)

    assert "ui-sans-serif" in page, "no web font may be required to read this"
    assert "&#34;" not in page

    # Composition leads. A reader handed a folder should learn what it is before
    # learning what is wrong with it, and a findings list only ever mentions a
    # property of the corpus when that property is broken.
    assert page.index("What this corpus is") < page.index("What would go wrong")
    assert page.index("What would go wrong") < page.index("Where your data sits")

    # The 258-dot scatter is gone. It was honest and useless: the positions are
    # a projection rather than distances, which had to be disclaimed directly
    # under the picture, and having read the disclaimer there was nothing left
    # to do with the dots.
    assert "<circle" not in page
    assert "not distances" not in page
    assert "What the map says" in page

    # The verdict is a caption for the findings list, so it stands at the head
    # of that list rather than at the head of the page.
    assert page.index("What would go wrong") < page.index('class="verdict')
    assert page.index('class="verdict') < page.index('class="finding')


def test_the_page_works_with_scripting_off_and_in_one_theme(tmp_path):
    """Two constraints that are one constraint: the file has to survive travel.

    It is opened from a mail attachment, from a shared drive, from a CI artifact
    browser — under its own CSP, which grants no ``script-src`` at all — and it
    is printed. So the sort control on the density grid is radio buttons and a
    CSS ``order``, its tooltips are ``::after`` on ``:hover``, and there is one
    palette rather than one per reader's laptop.
    """
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    data = tmp_path / "d"
    data.mkdir()
    (data / "train.jsonl").write_text(
        json.dumps({"text": "a sufficiently long atlas record for coverage"}) + "\n",
        encoding="utf-8",
    )
    result = scan(str(tmp_path))
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)

    page = html_report.render(result, fp, None)

    assert "<script" not in page.lower()
    assert "onclick" not in page.lower()
    assert "prefers-color-scheme" not in page
    assert "data-theme" not in page
    assert "color-scheme:light" in page


def test_the_density_grid_reaches_the_page_with_its_scale_explained(tmp_path):
    """A colour ramp nobody can read is decoration.

    Every square is a claim about a ratio, so the ratio is printed inside the
    square rather than revealed by hovering it, and a legend names the anchor
    that makes the colours mean anything. It is a real table so that its header
    repeats when the 48 rows cross a page boundary on paper.
    """
    from dropoutt.atlas import load_bundled
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.report import html as html_report
    from dropoutt.runner import scan

    data = tmp_path / "d"
    data.mkdir()
    (data / "train.jsonl").write_text(
        json.dumps({"text": "a sufficiently long atlas record for coverage"}) + "\n",
        encoding="utf-8",
    )
    result = scan(str(tmp_path))
    atlas = load_bundled()
    if atlas is None or atlas.region_size is None:
        pytest.skip("bundled atlas carries no reference sizes")
    landed = np.array([0, 0, 2, 4], dtype=np.int32)
    result.ctx.stats["atlas_coverage"] = atlas.coverage(
        landed, atlas.region_category[landed]
    )
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)

    page = html_report.render(result, fp, None)

    assert page.count('class="cell d') >= atlas.n_regions
    assert "own density" in page
    assert "no reach" in page
    assert "as common as on the map" in page
    assert 'aria-label="' in page
    # The scale is continuous, so the legend is one gradient rather than four
    # swatches the reader has to interpolate between by eye.
    assert "linear-gradient(90deg" in page

    # No hover, no sort control, and a table header the print stylesheet can
    # repeat. All three are the same decision.
    assert "data-t=" not in page
    assert "Hover" not in page
    assert 'type="radio"' not in page
    assert "<thead>" in page
    assert "display:table-header-group" in page


def _table_blocks(page: str) -> list[list[str]]:
    """Every Markdown table in the page, as its contiguous run of rows.

    Grouped by adjacency rather than by counting header rules. A rule sits
    *between* a table's header and its body, so counting rules before a line
    puts the header in one bucket and the body in the next — which then collides
    with the following table's header and compares rows that were never meant to
    line up.
    """
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in page.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def test_the_markdown_report_is_pasteable_and_honours_no_evidence(tmp_path):
    """The third rendering, for the reader who has no browser.

    A scan runs in CI and what a reviewer sees is a comment or a job log, so
    this one has to survive being pasted: no HTML, no images, and table cells
    that cannot be broken by a dataset called ``a|b``.
    """
    from dropoutt.fingerprint import build as build_fingerprint
    from dropoutt.report import markdown as md_report
    from dropoutt.runner import scan

    # The dataset is named for the one character that breaks a markdown table.
    # Windows forbids `|` anywhere in a path, so there the rest of this test
    # still runs and only that hostile name is dropped — the escaping it
    # exercises is in the renderer, which does not vary by platform.
    data = tmp_path / ("ab" if os.name == "nt" else "a|b")
    data.mkdir()
    planted = "sk-live-DEADBEEFdeadbeef01234567"
    (data / "train.jsonl").write_text(
        "\n".join(
            json.dumps({"text": f"a sufficiently long record number {i} to be read"})
            for i in range(20)
        ) + "\n" + json.dumps({"text": f"my key is {planted} keep it safe"}) + "\n",
        encoding="utf-8",
    )
    result = scan(str(tmp_path))
    fp = build_fingerprint(result.ctx, result.findings, total_chars=100, total_words=20)

    page = md_report.render(result, fp, None)
    assert "<div" not in page and "<span" not in page
    assert page.startswith("##")
    # A pipe inside a cell would add a column to the row it sits in and
    # misalign every cell after it, so every table row has the same shape.
    for block in _table_blocks(page):
        widths = {len(line.replace("\\|", "").split("|")) for line in block}
        assert len(widths) == 1, (widths, block[:3])

    quiet = md_report.render(result, fp, None, include_evidence=False)
    assert planted not in quiet
    assert "--no-evidence" in quiet


def test_category_counts_are_json_serialisable(tiny_atlas):
    """Regression: numpy int keys are refused by orjson and broke the writer."""
    from dropoutt.compat import json_dumps

    cov = tiny_atlas.coverage(
        np.array([1] * 50, dtype=np.int32), np.zeros(50, dtype=np.int32)
    )
    assert all(isinstance(k, str) for k in cov["by_category"])
    json_dumps(cov)  # must not raise


def test_single_region_coverage_does_not_render_negative_zero():
    from dropoutt.atlas.compare import concentration

    value = concentration({"region_entropy": -0.0, "max_region_entropy": 5.5})

    assert value == 0.0
    assert f"{value:.0%}" == "0%"


def test_short_records_are_excluded_from_the_atlas_and_the_count_reported(tmp_path):
    """A 20-character record cannot be placed on a topical map.

    Including it would inflate the off-atlas rate with records that were never
    placeable, so they are excluded. The exclusion is reported, not hidden.
    """
    import json as _json

    from dropoutt.runner import ATLAS_MIN_CHARS

    assert ATLAS_MIN_CHARS >= 40, "the gate must be at least as strict as language ID"

    data = tmp_path / "d"
    data.mkdir()
    with open(data / "train.jsonl", "w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(_json.dumps({"messages": [
                {"role": "user", "content": f"kisa {i}"},
                {"role": "assistant", "content": "evet"},
            ]}) + "\n")
    # Every record here is far below the gate, so none may be placed.
    from dropoutt.langid import LanguageDetector
    from dropoutt.runner import scan

    result = scan(str(tmp_path), detector=LanguageDetector())
    assert result.ctx.stats.get("atlas_coverage") is None


# -- atlas comparison ----------------------------------------------------


def _coverage(counts: dict[int, int], *, cats: dict[str, int] | None = None,
              version: str = "tiny-test", status: str = "ok") -> dict:
    """A coverage facet shaped like the one the fingerprint carries."""
    total = sum(counts.values())
    return {
        "status": status,
        "records": total,
        "off_atlas": 0,
        "atlas_version": version,
        "region_counts": {str(k): v for k, v in counts.items()},
        "by_category": cats or {},
        "top_regions": [
            {"region": r, "records": n, "terms": f"region {r}"}
            for r, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]
        ],
        "region_entropy": 1.0,
        "max_region_entropy": 2.0,
    }


def test_comparison_is_directional():
    """A small corpus inside a large one is contained; the reverse is not.

    A symmetric similarity score hides exactly the case worth acting on, which
    is the same reason cross-dataset overlap is directional.
    """
    from dropoutt.atlas.compare import compare

    small = _coverage({1: 100})
    large = _coverage({1: 100, 2: 400, 3: 500})

    assert compare(small, large).added_mass == 0.0, "small is wholly inside large"
    assert compare(large, small).added_mass > 0.85, "large is mostly outside small"


def test_disjoint_corpora_report_all_of_it_as_new():
    from dropoutt.atlas.compare import compare

    result = compare(_coverage({1: 50, 2: 50}), _coverage({8: 50, 9: 50}))
    assert result.comparable
    assert result.added_mass == 1.0
    assert result.similarity == 0.0
    assert {r for r, _, _ in result.a_only} == {1, 2}


def test_a_corpus_compared_with_itself_is_identical():
    from dropoutt.atlas.compare import compare

    cov = _coverage({1: 10, 4: 30, 7: 60})
    result = compare(cov, cov)
    assert result.added_mass == 0.0
    assert result.shared_mass == pytest.approx(1.0)
    assert result.similarity == pytest.approx(1.0)
    assert not result.a_only


def test_comparison_refuses_a_pre_0_1_4_suppressed_fingerprint():
    """A fingerprint whose histogram was discarded has nothing left to compare.

    This is not the old "too much off-atlas, refuse" rule, which is gone. It is
    the narrower fact that a fingerprint written before 0.1.4 does not carry the
    histogram at all, and the only fix is re-scanning.
    """
    from dropoutt.atlas.compare import compare

    good = _coverage({1: 100})
    bad = _coverage({1: 100}, status="suppressed")
    bad["reason"] = "78% of records are off-atlas"

    assert not compare(good, bad).comparable
    assert not compare(bad, good).comparable
    reason = compare(good, bad).reason
    assert "78%" in reason, "the stored rate is the useful part; do not drop it"
    assert "dropoutt scan" in reason, "say what would fix it"


def test_comparison_proceeds_with_a_high_off_atlas_rate_and_states_the_bound():
    """A partial right side biases novelty upward. Report the bound, do not refuse."""
    from dropoutt.atlas.compare import compare

    left = _coverage({1: 60, 2: 40})
    left.update(records=100, placed=100, off_atlas=0, off_atlas_rate=0.0)
    # Right placed only half its records, so regions it appears to miss may be
    # covered by the half that never placed.
    right = _coverage({1: 50})
    right.update(records=100, placed=50, off_atlas=50, off_atlas_rate=0.5)

    result = compare(left, right)
    assert result.comparable
    assert result.added_mass == pytest.approx(0.4)
    assert result.b_placed_share == pytest.approx(0.5)
    assert result.caveats, "an incomplete side must state its effect"
    assert "upper bound" in " ".join(result.caveats)


def test_the_three_way_partition_of_the_left_side_always_sums_to_one():
    """shared + new + unplaceable accounts for every record the left side sampled.

    `shared_mass` and `added_mass` are shares of the placed part, which is the
    right denominator for a distribution and the wrong one for "how much of this
    dataset are we talking about". These three are the second question.
    """
    from dropoutt.atlas.compare import compare

    left = _coverage({1: 60, 2: 40})
    left.update(records=200, placed=100, off_atlas=100, off_atlas_rate=0.5)
    right = _coverage({1: 50})
    right.update(records=100, placed=100, off_atlas=0, off_atlas_rate=0.0)

    r = compare(left, right)
    assert r.shared_of_all + r.added_of_all + r.a_unplaced == pytest.approx(1.0)
    assert r.a_unplaced == pytest.approx(0.5)
    assert r.shared_of_all == pytest.approx(0.3)  # 60% of the placed half
    assert r.added_of_all == pytest.approx(0.2)


def test_off_atlas_mass_never_enters_the_similarity():
    """Two corpora unplaceable in different directions are not therefore alike.

    Off-atlas is the complement of the atlas, an undifferentiated set. Carried as
    a shared coordinate it would push any two badly-fitting corpora toward 1.0
    however unrelated their placed halves are.
    """
    from dropoutt.atlas.compare import compare

    left = _coverage({1: 10})
    left.update(records=100, placed=10, off_atlas=90, off_atlas_rate=0.9)
    right = _coverage({2: 10})
    right.update(records=100, placed=10, off_atlas=90, off_atlas_rate=0.9)

    r = compare(left, right)
    assert r.comparable
    assert r.similarity == pytest.approx(0.0), "disjoint regions, whatever the off-atlas rate"
    assert r.added_mass == pytest.approx(1.0)


def test_comparison_refuses_across_atlas_versions():
    """Region ids are only meaningful within one atlas."""
    from dropoutt.atlas.compare import compare

    result = compare(_coverage({1: 10}), _coverage({1: 10}, version="atlas-lite-v9"))
    assert not result.comparable
    assert "atlas version" in result.reason


def test_full_histogram_is_used_rather_than_the_display_head():
    """Comparing on the top-twelve head computes shares over a fraction of the data.

    On real corpora the head covered 88% of one side and 36% of the other, which
    overstated novelty at 100% where the true figure was 62%.
    """
    from dropoutt.atlas.compare import compare, region_mass

    counts = dict.fromkeys(range(30), 10)
    cov = _coverage(counts)
    assert len(region_mass(cov)) == 30, "all regions must be read, not just the head"
    assert compare(cov, cov).a_head_coverage == 1.0


def test_region_counts_reach_the_coverage_report(tiny_atlas):
    """Regression: only the top twelve regions were stored, so diff was partial."""
    regions = np.array([1, 1, 2, 3, 3, 3, 5] * 20, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, np.zeros(140, dtype=np.int32))
    assert cov["region_counts"] == {"1": 40, "2": 20, "3": 60, "5": 20}
    from dropoutt.compat import json_dumps

    json_dumps(cov)  # keys must stay JSON-serialisable


# -- reading the coverage, not merely reporting it ------------------------
#
# The atlas used to emit an occupancy count, a spread percentage and five
# frequency terms per region, none of which a reader could act on. These cover
# the numbers added to make it legible.


def test_effective_regions_separates_breadth_from_occupancy(tiny_atlas):
    """Occupancy counts a region holding one record the same as one holding half.

    Two corpora that occupy the same four regions get the same occupancy number
    and should not get the same breadth number. Effective coverage caps each
    cell at 1× map density, so a peak does not erase the thin neighbours.
    """
    even = np.array([0, 1, 2, 3] * 25, dtype=np.int32)
    lopsided = np.array([0] * 97 + [1, 2, 3], dtype=np.int32)
    cats = tiny_atlas.region_category

    a = tiny_atlas.coverage(even, cats[even])
    b = tiny_atlas.coverage(lopsided, cats[lopsided])

    assert a["regions_occupied"] == b["regions_occupied"] == 4
    assert a["effective_regions"] == pytest.approx(4.0, abs=0.05)
    # Lopsided still scores the three thin cells as fractions, so below full.
    assert b["effective_regions"] < a["effective_regions"]
    assert b["effective_regions"] > 1.0


def test_coverage_names_the_subject_areas_the_corpus_never_reaches(tiny_atlas):
    """The question a histogram of your own data cannot answer."""
    regions = np.array([0, 1, 2, 3] * 25, dtype=np.int32)  # category 0 only
    cov = tiny_atlas.coverage(regions, tiny_atlas.region_category[regions])

    gaps = {int(g["category"]): g for g in cov["coverage_gaps"]}
    assert set(gaps) == {1, 2}
    assert gaps[1]["records"] == 0
    assert gaps[1]["regions"] == 4
    assert gaps[1]["regions_empty"] == 4
    assert cov["categories_total"] == 3


def test_a_category_the_corpus_covers_is_not_listed_as_a_gap(tiny_atlas):
    regions = np.array(list(range(12)) * 30, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, tiny_atlas.region_category[regions])
    assert cov["coverage_gaps"] == []


def test_datasets_occupying_the_same_regions_are_reported_as_alike(tiny_atlas):
    """Two datasets can share no wording and still cover identical ground."""
    regions = np.array([0, 1] * 40 + [0, 1] * 40 + [7, 8] * 40, dtype=np.int32)
    names = ["a"] * 80 + ["b"] * 80 + ["c"] * 80
    cov = tiny_atlas.coverage(regions, tiny_atlas.region_category[regions],
                              datasets=names)

    block = cov["by_dataset_regions"]
    assert set(block["datasets"]) == {"a", "b", "c"}
    pairs = {frozenset((p["a"], p["b"])): p["similarity"] for p in block["most_alike"]}
    assert pairs[frozenset(("a", "b"))] == pytest.approx(1.0, abs=1e-6)
    assert pairs[frozenset(("a", "c"))] == pytest.approx(0.0, abs=1e-6)


def test_off_atlas_records_are_left_out_of_the_per_dataset_signature(tiny_atlas):
    """The signature describes where a dataset sits, so unplaced rows cannot count."""
    regions = np.array([0] * 40 + [-1] * 40, dtype=np.int32)
    names = ["a"] * 80
    cov = tiny_atlas.coverage(regions, np.zeros(80, dtype=np.int32), datasets=names)
    assert cov["by_dataset_regions"]["datasets"]["a"]["placed"] == 40


def test_the_coverage_facet_carries_no_record_text(tiny_atlas):
    """fingerprint.json is the shareable artifact; excerpts live in ctx.stats."""
    regions = np.array([0, 1, 2, 3] * 25, dtype=np.int32)
    cov = tiny_atlas.coverage(
        regions, tiny_atlas.region_category[regions],
        datasets=["a"] * 50 + ["b"] * 50,
        texts=["gizli kayit metni burada duruyor ve paylasilmamali"] * 100,
    )
    assert "gizli kayit" not in json.dumps(cov, default=str)


def test_a_gap_is_absolute_when_the_atlas_records_no_reference_mass(tiny_atlas):
    """atlas-lite-v0 stores no reference distribution, so no share is invented."""
    tiny_atlas.region_size = None
    regions = np.array([0, 1, 2, 3] * 25, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, tiny_atlas.region_category[regions])
    assert all("reference_share" not in g for g in cov["coverage_gaps"])


def test_a_gap_becomes_relative_when_the_atlas_records_reference_mass(tiny_atlas):
    """With a baseline, 'nothing of yours is here' gains 'and it holds 60% of the reference'."""
    # Category 1 is four regions carrying most of the reference corpus;
    # category 2 is four regions carrying almost none of it.
    tiny_atlas.region_size = np.array(
        [10] * 4 + [150] * 4 + [1] * 4, dtype=np.int32
    )
    regions = np.array([0, 1, 2, 3] * 25, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, tiny_atlas.region_category[regions])

    gaps = {int(g["category"]): g for g in cov["coverage_gaps"]}
    assert gaps[1]["reference_share"] == pytest.approx(600 / 644, abs=1e-3)
    assert gaps[2]["reference_share"] == pytest.approx(4 / 644, abs=1e-3)
    # The gap that matters most is the one the reference corpus fills, not the
    # one with the most regions, so ordering follows the baseline when there is
    # one.
    assert cov["coverage_gaps"][0]["category"] == 1
