"""Atlas behaviour and report safety.

The atlas tests build a tiny synthetic artifact in-memory and load it through
the real loader, so they exercise the actual code path without a network call.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dropoutt.atlas.apply import OFF_ATLAS_SUPPRESS, Atlas
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


def test_coverage_is_suppressed_when_off_atlas_rate_is_high(tiny_atlas):
    """Better to withhold a number than to publish a misleading one."""
    regions = np.full(200, -1, dtype=np.int32)
    categories = np.zeros(200, dtype=np.int32)
    cov = tiny_atlas.coverage(regions, categories)
    assert cov["status"] == "suppressed"
    assert cov["off_atlas_rate"] == 1.0
    assert "does not fit this corpus" in cov["reason"]
    assert "by_category" not in cov, "no coverage detail may leak when suppressed"


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


def test_install_hints_survive_rich_markup():
    """Regression: rich reads `[tokenizer]` as a style tag and deletes it.

    The user was told to run `pip install 'dropoutt'`, which installs the core
    package they already have and not the extra they were missing. Every install
    hint reaches the terminal through markup, so every one of them must be
    escaped at the render site.
    """
    import io

    from rich.console import Console

    from dropoutt.compat import capability_report

    for name, info in capability_report().items():
        hint = str(info.get("install") or "")
        if not hint:
            continue
        buf = io.StringIO()
        Console(file=buf, width=120, no_color=True).print(f"[dim]{_m(hint)}[/dim]")
        rendered = buf.getvalue()
        extra = hint.split("[", 1)[1].split("]", 1)[0]
        assert f"[{extra}]" in rendered, (
            f"{name}: the extras name was eaten by rich markup; "
            f"hint {hint!r} rendered as {rendered.strip()!r}"
        )


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
    out = safe_snippet("safe‮text‭more")
    assert "‮" not in out and "‭" not in out


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
        for i in range(8):
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


def test_html_report_uses_sans_serif_and_plots_atlas_coverage(tmp_path):
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

    assert "Inter,ui-sans-serif,system-ui" in page
    assert "ui-monospace" not in page
    assert "&#34;" not in page
    assert '<svg class="atlas-map"' in page
    assert page.count("<circle class=\"atlas-point") == atlas.n_regions
    assert "Larger highlighted circles contain more sampled records" in page


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


def test_comparison_refuses_when_either_side_was_suppressed():
    """Two unreliable histograms must not produce a confident-looking answer."""
    from dropoutt.atlas.compare import compare

    good = _coverage({1: 100})
    bad = _coverage({1: 100}, status="suppressed")
    bad["reason"] = "78% of records are off-atlas"

    assert not compare(good, bad).comparable
    assert not compare(bad, good).comparable
    assert "78%" in compare(good, bad).reason


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

    counts = {r: 10 for r in range(30)}
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
