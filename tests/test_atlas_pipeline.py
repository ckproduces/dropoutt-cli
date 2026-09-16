"""Shared atlas pipeline: extraction, chunking, normalization, soft assign."""

from __future__ import annotations

import json

import numpy as np

from dropoutt.atlas.apply import Atlas
from dropoutt.atlas.chunk import chunk_text
from dropoutt.atlas.embed import Embedder, QuantizedTable
from dropoutt.atlas.extract import detect_format, extract_text
from dropoutt.atlas.normalize import fit_norm
from dropoutt.atlas.pipeline import pipeline_hash, population_crosswalk


def test_json_extraction_keeps_content_drops_keys():
    raw = json.dumps({
        "id": "abc12345",
        "count": 3,
        "active": True,
        "note": (
            "The patient presented with acute chest pain and dyspnea lasting "
            "more than two hours before arrival at the clinic."
        ),
    })
    text, fmt = extract_text(raw)
    assert fmt == "json"
    assert "chest pain" in text
    assert "abc12345" not in text
    assert "active" not in text


def test_html_extraction_strips_tags():
    raw = (
        "<html><script>x=1</script><body><p>"
        "Hello from the body text about clinical medicine and patient care in "
        "the emergency department this morning."
        "</p></body></html>"
    )
    text, fmt = extract_text(raw, filename="page.html")
    assert fmt == "html"
    assert "Hello from the body" in text
    assert "<script>" not in text
    assert "x=1" not in text


def test_code_extraction_keeps_comments_and_strings():
    raw = '''
def add(a, b):
    """Add two numbers and return the sum for the caller."""
    # Important: handle overflow in production later
    return a + b
'''
    text, fmt = extract_text(raw, filename="add.py")
    assert fmt == "code"
    assert "Add two numbers" in text
    assert "handle overflow" in text


def test_detect_format_returns_unknown_rather_than_guessing():
    assert detect_format("@@@\x00\x01\x02") == "unknown"


def test_chunker_splits_long_documents():
    para = "Word " * 80
    doc = "\n\n".join([para] * 8)
    chunks = chunk_text(doc, target_words=200, max_words=300, min_words=20)
    assert len(chunks) >= 2
    assert all(len(c.split()) <= 350 for c in chunks)


def test_normalization_is_frozen_and_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 128)).astype(np.float32)
    norm = fit_norm(x, pca_k=2, dim=128)
    a = norm.apply(x[:10])
    b = norm.apply(x[:10])
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_soft_assign_puts_mass_on_multiple_regions(tmp_path):
    rng = np.random.default_rng(1)
    dim, n = 16, 20
    centroids = rng.normal(size=(n, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    mid = centroids[0] + centroids[1]
    mid /= np.linalg.norm(mid)
    path = tmp_path / "a.npz"
    np.savez_compressed(
        path,
        centroids=centroids,
        region_category=np.zeros(n, dtype=np.int32),
        coords=np.zeros((n, 2), dtype=np.float32),
        probe_coef=np.zeros((0, dim), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        meta=np.array([json.dumps({
            "version": "tiny", "embed_model": "fake",
            "off_atlas_threshold": 0.1, "soft_k": 5, "soft_temperature": 0.2,
            "region_terms": [f"r{i}" for i in range(n)],
        })], dtype=object),
        allow_pickle=True,
    )
    atlas = Atlas.load(path)
    cells, weights, _scores = atlas.soft_assign(mid.reshape(1, -1))
    assert cells.shape == (1, 5)
    assert abs(float(weights[0].sum()) - 1.0) < 1e-5
    assert int((weights[0] > 0.05).sum()) >= 2


def test_pipeline_hash_is_stable():
    assert pipeline_hash() == pipeline_hash()
    assert len(pipeline_hash()) == 32


def test_bundled_atlas_products_match_their_profiles():
    from dropoutt.atlas.apply import load_bundled
    from dropoutt.atlas.profiles import (
        ATLAS_V2,
        ATLAS_V2_LITE,
        ATLAS_V3,
        DEFAULT_ATLAS_VERSION,
    )

    default = load_bundled()
    assert default is not None, "the default product must be bundled"
    assert default.meta.get("version") == DEFAULT_ATLAS_VERSION == ATLAS_V3.version
    assert default.dim == ATLAS_V3.dim == 128
    assert default.n_l1 == ATLAS_V3.n_l1 == 256
    assert default.n_regions == ATLAS_V3.l2_budget == 4096
    assert len(default.region_terms) == default.n_regions
    # The map ships with the evidence that it is a map of subjects: how much
    # language a linear probe still recovers after normalization, and that
    # each record's language was detected rather than inherited from its shard.
    probe = default.meta.get("language_probe") or {}
    assert probe.get("language_normalized", 1.0) < probe.get("language_raw", 0.0)
    assert default.meta.get("language_labels_source", "").startswith("detected-per-record")
    assert len(default.meta["normalization"]["lang_labels"]) >= 45

    lite = load_bundled("atlas-v2-lite")
    assert lite is not None, "atlas-v2-lite must be bundled"
    assert lite.meta.get("version") == ATLAS_V2_LITE.version
    assert lite.dim == ATLAS_V2_LITE.dim == 64
    assert lite.n_l1 == ATLAS_V2_LITE.n_l1 == 32
    assert 32 <= lite.n_regions <= 320
    assert len(lite.region_terms) == lite.n_regions
    assert lite.embed_model == "minishlab/potion-multilingual-128M"

    full = load_bundled("atlas-v2")
    assert full is not None, "atlas-v2 must be bundled"
    assert full.meta.get("version") == ATLAS_V2.version
    assert full.dim == ATLAS_V2.dim == 128
    assert full.n_l1 == ATLAS_V2.n_l1 == 128
    assert 128 <= full.n_regions <= 1_280
    assert len(full.region_terms) == full.n_regions


def test_bundled_v3_carries_an_explicit_off_atlas_cutoff():
    """The cutoff ships in the artifact; the loader's 0.35 is a fallback only.

    docs/atlas.md promises the 2nd percentile of the atlas's own reference
    records. A map that leaves the key out silently takes 0.35 instead, and on
    atlas-v2 that number sits above the map's own 2nd percentile (0.309), which
    is what put 12-18% of ordinary held-out prose off-atlas there. atlas-v3 has
    to say its cutoff itself, and say how it was calibrated.
    """
    from dropoutt.atlas.apply import load_bundled

    atlas = load_bundled("atlas-v3")
    assert atlas is not None, "atlas-v3 must be bundled"
    assert "off_atlas_threshold" in atlas.meta, "cutoff must be stamped, not defaulted"
    cutoff = atlas.meta["off_atlas_threshold"]
    assert isinstance(cutoff, float)
    assert atlas.off_threshold == cutoff
    calibration = atlas.meta["off_atlas_calibration"]
    assert calibration["percentile"] == 2.0
    assert calibration["draw"] == "language-axis-balanced"
    assert abs(calibration["reference_percentiles"]["p2"] - cutoff) < 1e-4
    assert calibration["rows_scored"] >= 1_000_000
    # Held-out in-family prose scores p2 0.425 on this map (benchmark b4); a
    # cutoff above that would reject ordinary prose, below 0.2 nothing at all.
    assert 0.2 <= cutoff <= 0.45


def test_bundled_maps_carry_no_reference_text():
    """Centroids, sizes, constants and names ship; reference records do not.

    The builder writes `exemplar_texts`, a few hundred characters of the
    records nearest each cell's centre, as a labelling aid. Nothing at runtime
    reads it, and in the wheel it was 16,384 verbatim excerpts of web, forum
    and encyclopedia text with no licence manifest. `tools/strip_atlas_exemplars.py`
    removes it before release; this keeps it removed.
    """
    import numpy as np

    from dropoutt.atlas.apply import atlas_path_for

    for version in ("atlas-v3", "atlas-v2", "atlas-v2-lite", "atlas-v1-lite"):
        path = atlas_path_for(version)
        assert path is not None, f"{version} must be bundled"
        data = np.load(path, allow_pickle=True)
        assert "exemplar_texts" not in data.files, f"{version} ships reference excerpts"
        for name in data.files:
            if name in ("meta", "prototype_record_ids"):
                continue
            assert data[name].dtype.kind not in "OSU", (
                f"{version}: array {name!r} holds text ({data[name].dtype})"
            )


def test_v2_products_carry_distinct_l1_subject_labels():
    from dropoutt.atlas.apply import load_bundled
    from dropoutt.atlas.compare import category_labels
    from dropoutt.atlas.profiles import ATLAS_V2, ATLAS_V2_LITE

    for version, profile in (("atlas-v2-lite", ATLAS_V2_LITE), ("atlas-v2", ATLAS_V2)):
        atlas = load_bundled(version)
        assert atlas is not None
        labels = category_labels(atlas)
        assert len(labels) == profile.n_l1
        assert len(set(labels.values())) == profile.n_l1
        assert atlas.meta.get("l1_labels_source") == f"curated:l1_labels_{version}.json"


def test_v1_lite_still_carries_curated_subject_labels():
    from dropoutt.atlas.apply import load_bundled
    from dropoutt.atlas.compare import category_labels

    atlas = load_bundled("atlas-v1-lite")
    assert atlas is not None
    labels = category_labels(atlas)
    assert len(labels) == atlas.n_l1 == 48
    assert len(set(labels.values())) == 48, "every subject area needs its own name"
    assert labels[45] == "Database schemas and query construction"
    assert labels[21] == "Website boilerplate and page furniture"
    assert "C and C++ source code" in labels.values()
    assert atlas.meta.get("l1_labels_source", "").startswith("curated:")


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def __init__(self):
        self.calls = 0

    def encode_batch(self, texts, add_special_tokens=False):
        assert not add_special_tokens
        self.calls += 1
        return [_Encoding([int(part) for part in text.split()]) for text in texts]


def _fake_encoder(out_dim=3):
    table = QuantizedTable.from_float(
        np.arange(40, dtype=np.float32).reshape(10, 4), width=4
    )
    tokenizer = _Tokenizer()
    return Embedder(table, tokenizer, "fake", out_dim=out_dim), table, tokenizer


def test_sparse_sif_pool_matches_weighted_token_average():
    base, table, _ = _fake_encoder()
    tokens = base.tokenize(["1 1 2", "", "2 3"], batch_size=2, max_length=8)
    probs, ids, log_probs = base.token_log_prob(tokens)
    embedder = base.bind_idf(probs)

    actual = embedder.encode_tokenized(tokens)

    expected = np.zeros((3, 3), dtype=np.float32)
    for row, token_ids in enumerate(([1, 1, 2], [], [2, 3])):
        if not token_ids:
            continue
        p = np.array([np.exp(probs[token_id]) for token_id in token_ids])
        weights = 1e-3 / (1e-3 + p)
        weights /= weights.sum()
        expected[row] = weights @ table.rows(np.array(token_ids), 3)

    assert np.allclose(actual, expected, atol=1e-6)
    assert tokens.indptr.tolist() == [0, 3, 3, 5]
    assert ids.tolist() == [1, 2, 3]
    assert np.all(np.diff(log_probs) <= 0)


def test_weighted_encode_batch_tokenizes_each_text_once():
    base, _, tokenizer = _fake_encoder()
    tokens = base.tokenize(["1 2", "2 3", "3 4"])
    probs, _, _ = base.token_log_prob(tokens)
    tokenizer.calls = 0

    result = base.bind_idf(probs).encode(["1 2", "2 3", "3 4"], batch_size=64)

    assert result.shape == (3, 3)
    assert tokenizer.calls == 1


def test_quantised_rows_stay_within_a_step_of_the_weights_they_replace():
    """int8 with a per-row scale, so the error is bounded by half a step."""
    rng = np.random.default_rng(0)
    table = rng.normal(size=(64, 128)).astype(np.float32)
    quantized = QuantizedTable.from_float(table)

    restored = quantized.rows(np.arange(64))
    step = np.abs(table).max(axis=1) / 127.0

    assert restored.shape == (64, 128)
    assert np.all(np.abs(restored - table) <= step[:, None] / 2 + 1e-6)


def test_atlas_v2_profile_windows_and_columns_are_declared():
    from dropoutt.atlas.embed import select_token_windows
    from dropoutt.atlas.profiles import get_profile

    full = get_profile("atlas-v2")
    lite = get_profile("atlas-v2-lite")  # the default product is atlas-v3 now
    assert (full.dim, full.pooling, full.max_chars, full.max_tokens, full.default_sample) == (
        128, "sif", 2_000, 512, 200_000,
    )
    assert (lite.version, lite.dim, lite.pooling, lite.max_chars, lite.max_tokens, lite.default_sample) == (
        "atlas-v2-lite", 64, "sif", 2_000, 512, 50_000,
    )
    assert select_token_windows(list(range(100)), 10) == [0, 1, 2, 3, 49, 50, 96, 97, 98, 99]


def test_v2_coverage_is_flat_l2_only(tmp_path):
    rng = np.random.default_rng(7)
    centroids = rng.normal(size=(3, 16)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    path = tmp_path / "v2.npz"
    np.savez_compressed(
        path,
        centroids=centroids,
        region_category=np.array([0, 0, 1], dtype=np.int32),
        coords=np.zeros((3, 2), dtype=np.float32),
        probe_coef=np.zeros((0, 16), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        meta=np.array([json.dumps({"version": "atlas-v2-lite", "region_terms": ["a", "b", "c"]})], dtype=object),
        allow_pickle=True,
    )
    atlas = Atlas.load(path)
    coverage = atlas.coverage(np.array([0, 1, 2]), np.array([0, 0, 1]))
    assert atlas.flat_cells
    assert coverage["by_category"] == {}
    assert coverage["coverage_gaps"] == []
    assert coverage["categories_total"] == 0
    assert "category" not in coverage["top_regions"][0]


def test_population_crosswalk_uses_shared_members_not_coordinates():
    previous = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    current = np.array([0, 0, 1, 2, 2, 2], dtype=np.int32)

    result = population_crosswalk(
        current,
        previous,
        n_current=3,
        n_previous=2,
        previous_version="old",
    )

    assert result["previous_version"] == "old"
    assert result["cells"][2]["previous_cell_id"] == 1
    assert result["cells"][2]["relationship"] == "unchanged"
    assert result["cells"][2]["population_jaccard"] == 1.0
def test_v1_loader_reads_norm_and_idf(tmp_path):
    dim, n_l2, n_l1 = 8, 12, 3
    rng = np.random.default_rng(2)
    centroids = rng.normal(size=(n_l2, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    l1 = rng.normal(size=(n_l1, dim)).astype(np.float32)
    l1 /= np.linalg.norm(l1, axis=1, keepdims=True)
    path = tmp_path / "v1.npz"
    np.savez_compressed(
        path,
        centroids=centroids,
        region_category=np.repeat(np.arange(n_l1), 4).astype(np.int32),
        coords=np.zeros((n_l2, 2), dtype=np.float32),
        l1_centroids=l1,
        l1_size=np.array([4, 4, 4], dtype=np.int32),
        region_size=np.ones(n_l2, dtype=np.int32),
        norm_mean=np.zeros(dim, dtype=np.float32),
        norm_pca=np.zeros((2, dim), dtype=np.float32),
        idf_token_ids=np.array([1, 2, 3], dtype=np.int32),
        idf_log_probs=np.array([-1.0, -2.0, -3.0], dtype=np.float32),
        distance_refs=np.zeros((n_l2, 5), dtype=np.float32),
        distance_refs_support=np.arange(n_l2, dtype=np.int32),
        distance_refs_reliable=np.arange(n_l2) >= 4,
        cell_source_counts=np.ones((n_l2, 2), dtype=np.int32),
        cell_topic_counts=np.ones((n_l2, 3), dtype=np.int32),
        cell_language_counts=np.ones((n_l2, 2), dtype=np.int32),
        cooccurrence_ids=np.zeros((n_l2, 2), dtype=np.int32),
        cooccurrence_scores=np.ones((n_l2, 2), dtype=np.float32),
        prototype_vectors=np.ones((n_l2, 2, dim), dtype=np.float16),
        prototype_record_ids=np.full((n_l2, 2), b"abc", dtype="S16"),
        prototype_distances=np.ones((n_l2, 2), dtype=np.float32),
        probe_coef=np.zeros((0, dim), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
        meta=np.array([json.dumps({
            "version": "atlas-lite-v1", "embed_model": "fake",
            "off_atlas_threshold": 0.2, "pipeline_hash": "abc",
            "region_terms": [f"r{i}" for i in range(n_l2)],
            "l1_labels": ["a", "b", "c"],
        })], dtype=object),
        allow_pickle=True,
    )
    atlas = Atlas.load(path)
    assert atlas.norm is not None
    assert atlas.token_log_prob[2] == -2.0
    assert atlas.n_l1 == 3
    assert atlas.pipeline_hash == "abc"
    assert atlas.distance_refs_support.tolist() == list(range(n_l2))
    assert int(atlas.distance_refs_reliable.sum()) == n_l2 - 4
    assert atlas.cell_source_counts.shape == (n_l2, 2)
    assert atlas.prototype_vectors.shape == (n_l2, 2, dim)
    cats = atlas.categorize(centroids[:3])
    assert cats.shape == (3,)


def _tiny_atlas(tmp_path, *, lang_means=None, lang_labels=()):
    """A two-parent, six-child atlas with a deliberately misleading L1 geometry."""
    rng = np.random.default_rng(11)
    dim, n_l2, n_l1 = 8, 6, 2
    centroids = rng.normal(size=(n_l2, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    parents = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    # L1 centroids that do NOT agree with the children they claim to summarise.
    # This is the v2 situation in miniature; categorize must ignore them.
    l1 = -np.stack([centroids[:3].mean(0), centroids[3:].mean(0)]).astype(np.float32)
    l1 /= np.linalg.norm(l1, axis=1, keepdims=True)

    arrays = {
        "centroids": centroids,
        "region_category": parents,
        "l1_centroids": l1,
        "coords": np.zeros((n_l2, 2), dtype=np.float32),
        "probe_coef": np.zeros((0, dim), dtype=np.float32),
        "probe_intercept": np.zeros(0, dtype=np.float32),
        "probe_classes": np.zeros(0, dtype=np.int32),
        "norm_mean": np.zeros(dim, dtype=np.float32),
        "norm_pca": np.zeros((0, dim), dtype=np.float32),
        "coarse_knots": np.tile(np.linspace(0.0, 1.0, 5), (n_l1, 1)).astype(np.float32),
        "coarse_expected": np.tile(np.linspace(0.0, 0.5, 5), (n_l1, 1)).astype(np.float32),
        "family_distinguishable": np.array([True, False]),
        "family_sibling_overlap": np.array([0.1, 0.9], dtype=np.float32),
        "meta": np.array([json.dumps({
            "version": "tiny-v3", "embed_model": "fake",
            "encoder_weight_hash": "deadbeef",
            "off_atlas_threshold": -1.0, "soft_k": 3, "soft_temperature": 0.2,
            "region_terms": [f"r{i}" for i in range(n_l2)],
            "normalization": {"variant": "per_language" if lang_means is not None
                              else "global", "lang_labels": list(lang_labels)},
        })], dtype=object),
        "allow_pickle": True,
    }
    if lang_means is not None:
        arrays["norm_lang_means"] = np.asarray(lang_means, dtype=np.float32)
    path = tmp_path / "tiny.npz"
    np.savez_compressed(path, **arrays)
    return Atlas.load(path)


def test_coarse_label_is_the_parent_of_the_fine_cell(tmp_path):
    """Drilling down must never flip the coarse answer.

    v2 took a separate arg-max over L1 centroids, and on its own shipped
    reference records that disagreed with the parent of the nearest fine cell
    for 24.9% of them. Here the L1 centroids are deliberately pointed the wrong
    way: if categorize still consults them, every row disagrees.
    """
    atlas = _tiny_atlas(tmp_path)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 8)).astype(np.float32)

    coarse = atlas.categorize(x)
    _, _, nearest = atlas.assign_full(x)

    assert np.array_equal(coarse, atlas.region_category[nearest])


def test_per_language_centering_changes_placement_only_for_known_languages(tmp_path):
    means = np.stack([np.full(8, 0.5, dtype=np.float32), np.zeros(8, dtype=np.float32)])
    atlas = _tiny_atlas(tmp_path, lang_means=means, lang_labels=("tr", "en"))
    assert atlas.uses_language_centering

    x = np.tile(np.linspace(-1, 1, 8).astype(np.float32), (4, 1))
    tr_first = atlas.project(x, ["tr", "en", "unknown", "de"])

    # "en" mean is zero and unknown/de fall back to the global mean, which is
    # also zero here, so only the Turkish row moves.
    assert not np.allclose(tr_first[0], tr_first[1])
    assert np.allclose(tr_first[1], tr_first[2])
    assert np.allclose(tr_first[2], tr_first[3])


def test_a_record_that_pooled_to_nothing_is_never_placed(tmp_path):
    """Forty spaces must not land, confidently, in one particular cell.

    Mean removal turns an all-zero embedding into the fixed direction ``-mean``,
    which on atlas-v3 scores 0.76 against one cell — above the cutoff, so a
    blank record was placed as if it were about something. The build's own
    calibration dropped such rows; the runtime has to as well.
    """
    means = np.stack([np.full(8, 0.5, dtype=np.float32), np.zeros(8, dtype=np.float32)])
    atlas = _tiny_atlas(tmp_path, lang_means=means, lang_labels=("tr", "en"))
    atlas.meta["off_atlas_threshold"] = 0.1
    rng = np.random.default_rng(5)
    x = rng.normal(size=(4, 8)).astype(np.float32)
    x[1] = 0.0
    x[3] = 0.0

    projected = atlas.project(x, ["tr", "tr", "en", "unknown"])
    assert not projected[1].any() and not projected[3].any()
    assert projected[0].any() and projected[2].any()

    placed = atlas.assign_all(x, ["tr", "tr", "en", "unknown"])
    assert placed.best[1] == -1 and placed.best[3] == -1
    assert placed.score[1] == 0.0 and placed.score[3] == 0.0
    assert placed.best[0] >= 0 and placed.best[2] >= 0
    best, score, _ = atlas.assign_full(x, ["tr", "tr", "en", "unknown"])
    assert best[1] == -1 and score[1] == 0.0
    # Without language centering the same rule holds through the global path.
    plain = _tiny_atlas(tmp_path)
    plain.meta["off_atlas_threshold"] = 0.1
    assert plain.assign(x)[0][1] == -1


def test_assignment_chunks_are_sized_by_the_map(tmp_path):
    """A chunk is a byte budget, not a row count tuned for a 212-cell map."""
    atlas = _tiny_atlas(tmp_path)
    for cells, rows in ((215, 32_768), (4_096, 2_048), (65, 32_768), (16_384, 1_024)):
        atlas.centroids = np.zeros((cells, 8), dtype=np.float32)
        assert atlas.assign_chunk_rows() == rows
        assert rows * cells * 4 <= max(atlas.ASSIGN_CHUNK_BYTES, 1_024 * cells * 4)


def test_coarse_distance_correction_pulls_novelty_down(tmp_path):
    atlas = _tiny_atlas(tmp_path)
    raw = np.array([0.25, 0.5, 0.75], dtype=np.float32)

    corrected = atlas.correct_coarse_distance(np.array([0, 0, 0]), raw)

    # The table maps [0,1] onto [0,0.5]: a coarse distance overstates novelty,
    # and without this every corpus is told it is unusual.
    assert np.allclose(corrected, raw / 2, atol=1e-6)


def test_family_flag_gates_naming_a_child_cell(tmp_path):
    atlas = _tiny_atlas(tmp_path)
    assert atlas.can_name_children(0) is True
    assert atlas.can_name_children(1) is False
    # Unknown families and older artifacts stay nameable rather than silent.
    assert atlas.can_name_children(99) is True


def test_containment_crosswalk_scores_a_clean_split_as_continuity():
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
    from build_atlas import containment_crosswalk

    # One previous cell split cleanly into two; the other survives intact.
    previous = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    current = np.array([0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)

    result = containment_crosswalk(current, previous, 3, 2, "old")

    assert result["summary"]["clean_split"] == 1
    assert result["summary"]["unchanged"] == 1
    assert result["summary"]["continuity_rate"] == 1.0


def test_region_labels_rescore_artifacts_that_predate_them():
    """A map without stored labels still captions itself contrastively."""
    from dropoutt.atlas.apply import _contrastive_labels

    terms = [
        "this, with, firefox, chrome",
        "this, with, taxonomy, described",
        "this, with, directed, films",
    ]

    labels = _contrastive_labels(terms, width=2)

    assert labels[0] == "firefox, chrome"
    assert all("this" not in label for label in labels)
