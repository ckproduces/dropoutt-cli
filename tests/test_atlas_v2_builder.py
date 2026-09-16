"""Focused, dependency-light checks for the dual Atlas v2 build entry point."""

from __future__ import annotations

import gzip
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

BUILD = Path(__file__).resolve().parents[1] / "tools" / "build_atlas_v2.py"
SPEC = importlib.util.spec_from_file_location("build_atlas_v2", BUILD)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_three_window_selection_is_exact_and_deterministic():
    ids = np.arange(20, dtype=np.int32)

    selected = builder.selected_token_ids(ids, 10)

    assert selected.tolist() == [0, 1, 2, 3, 9, 10, 16, 17, 18, 19]
    assert builder.selected_token_ids(ids, 30).tolist() == ids.tolist()


def test_lite_stratification_has_fixed_size_and_preserves_groups():
    axes = ["web"] * 8 + ["code"] * 2
    languages = ["en"] * 8 + ["tr"] * 2

    first = builder.stratified_indices(axes, languages, 5)
    second = builder.stratified_indices(axes, languages, 5)

    assert np.array_equal(first, second)
    assert len(first) == 5
    assert sum(axes[i] == "web" for i in first) == 4
    assert sum(axes[i] == "code" for i in first) == 1


def test_merge_small_community_uses_largest_adjacent_edge_weight():
    vectors = np.array([[1.0, 0.0]] * 200 + [[0.9, 0.1]] * 3 + [[0.0, 1.0]] * 200, np.float32)
    labels = np.array([0] * 200 + [1] * 3 + [2] * 200, np.int32)
    neighbors = np.zeros((len(labels), 1), np.int32)
    weights = np.zeros((len(labels), 1), np.float32)
    neighbors[200:203, 0] = 0
    weights[200:203, 0] = 0.9

    merged = builder.merge_small_communities(labels, vectors, neighbors, weights, minimum=200)

    assert len(np.unique(merged)) == 2
    assert merged[200] == merged[0]


def test_sbatch_declares_the_approved_single_node_resources():
    text = (Path(__file__).resolve().parents[1] / "tools" / "run_atlas_v2.sbatch").read_text()

    for expected in (
        "#SBATCH --account=c00005", "#SBATCH --partition=a100q",
        "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=64",
        "#SBATCH --gres=gpu:1", "#SBATCH --mem=480G", "#SBATCH --time=10-00:00:00",
        "TARGET_ROWS=10627780", "--scale 3.0",
    ):
        assert expected in text


def test_best_kmeans_picks_k_in_range_on_separated_blobs():
    rng = np.random.default_rng(0)
    blobs = [rng.normal(loc=(i * 8, 0, 0, 0), scale=0.15, size=(50, 4)) for i in range(5)]
    vectors = np.vstack(blobs).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9

    labels, k, score, centres = builder._best_kmeans(vectors, 1, 10, seed=0)

    assert 1 <= k <= 10
    assert labels.min() == 0
    assert int(labels.max()) + 1 == k
    assert centres.shape == (k, 4)
    assert np.isfinite(score)


def test_uint64_set_accepts_once(tmp_path):
    seen = builder.Uint64Set(tmp_path / "seen.u64", slots=1024)
    assert seen.add(7)
    assert not seen.add(7)
    assert seen.add(8)


def test_uint64_set_rehashes_smaller_file(tmp_path):
    path = tmp_path / "seen.u64"
    small = builder.Uint64Set(path, slots=1024)
    assert small.add(7)
    assert small.add(99)
    small.flush()
    del small

    bigger = builder.Uint64Set(path, slots=2048)
    assert not bigger.add(7)
    assert not bigger.add(99)
    assert bigger.add(8)


def test_disk_corpus_checkpoint_roundtrip(tmp_path):
    corpus = builder.DiskCorpus(tmp_path, dim=4)
    corpus.append(np.ones((3, 4), np.float32), ["web"] * 3, ["en"] * 3, ["src"] * 3)
    counts = np.arange(6, dtype=np.int64)
    corpus.save_checkpoint({"used"}, 99, counts)

    restored = builder.DiskCorpus(tmp_path, dim=4)
    consumed, logical, loaded = restored.load_checkpoint()

    assert restored.n == 3
    assert consumed == {"used"}
    assert logical == 99
    np.testing.assert_array_equal(loaded, counts)
    np.testing.assert_allclose(restored.rows_f32(np.array([0])), np.ones((1, 4)))


def test_disk_corpus_keeps_record_lengths_across_a_checkpoint(tmp_path):
    corpus = builder.DiskCorpus(tmp_path, dim=4)
    corpus.append(np.ones((3, 4), np.float32), ["web"] * 3, ["en"] * 3, ["src"] * 3,
                  [120, 2000, 70_000])
    corpus.save_checkpoint(set(), 0, np.zeros(2, dtype=np.int64))

    restored = builder.DiskCorpus(tmp_path, dim=4)
    restored.load_checkpoint()
    assert restored.lengths[:3].tolist() == [120, 2000, 65_535]


def test_prepare_worker_reads_text_exactly_as_the_encoder_does(tmp_path):
    """The parallel ingest tokenizes in workers; the ids must be the encoder's own."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    from dropoutt.atlas.embed import Embedder, QuantizedTable
    from dropoutt.atlas.textnorm import ENCODER_INPUT_V1

    vocab = {"[UNK]": 0, "Informationen": 1, "Über": 2, "Datenschutz": 3, "table": 4,
             "row": 5, "the": 6, "cat": 7, "sat": 8, "on": 9, "mat": 10}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    import atlas_langid_worker as worker

    worker.init(str(path), ENCODER_INPUT_V1.declaration(), 2_000, 6, 80)
    rows = [
        ("INFORMATIONEN ÜBER DATENSCHUTZ | table | row | the cat sat on the mat " * 3, False),
        ("short", False),
        ("the cat sat on the mat and the cat sat on the mat again, twice over, " * 2, False),
    ]
    texts, digests, detections, token_ids = worker.prepare_chunk(rows)

    assert texts[1] is None and digests[1] == 0
    kept = [text for text in texts if text is not None]
    assert len(detections) == len(token_ids) == len(kept) == 2
    encoder = Embedder(QuantizedTable.from_float(np.ones((len(vocab), 4), np.float32), width=4),
                       Tokenizer.from_file(str(path)), "fake", out_dim=4,
                       encoder_input=ENCODER_INPUT_V1)
    expected = encoder.tokenize(kept, max_length=6)
    for row, ids in enumerate(token_ids):
        start, stop = expected.indptr[row], expected.indptr[row + 1]
        assert ids.tolist() == expected.token_ids[start:stop].tolist()
    detected = builder.resolve_languages([("nb", 0.9), ("unknown", 0.9), ("de", 0.1)], ["x", "y", "z"])
    assert detected == ["no", "y", "z"]


def test_weighted_draw_follows_weight_and_is_deterministic():
    weights = np.array([1.0, 1.0, 8.0, 0.0, 2.0])
    draw = builder.weighted_draw(weights, 12)

    assert draw.tolist() == builder.weighted_draw(weights, 12).tolist()
    assert np.all(np.diff(draw) >= 0)
    counts = np.bincount(draw, minlength=5)
    assert counts[2] == 8 and counts[3] == 0
    assert counts[0] + counts[1] + counts[4] == 4
    assert builder.weighted_draw(np.zeros(4), 3).tolist() == [0, 1, 3]


def test_disk_corpus_flushes_each_mapping_before_growth(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "GROW_ROWS", 3)
    corpus = builder.DiskCorpus(tmp_path, dim=2)
    for value in range(4):
        corpus.append(
            np.full((2, 2), value + 1, np.float32),
            [f"axis-{value}"] * 2,
            [f"lang-{value}"] * 2,
            [f"source-{value}"] * 2,
        )
    corpus.flush()

    np.testing.assert_allclose(
        corpus.rows_f32(np.arange(8)),
        np.repeat(np.arange(1, 5, dtype=np.float32), 2)[:, None].repeat(2, axis=1),
    )
    assert corpus.axes() == [f"axis-{value}" for value in range(4) for _ in range(2)]
    assert corpus.languages() == [f"lang-{value}" for value in range(4) for _ in range(2)]
    assert corpus.sources() == [f"source-{value}" for value in range(4) for _ in range(2)]


def test_reservoir_roundtrip(tmp_path):
    path = tmp_path / "reservoir.jsonl"
    reservoir = builder.Reservoir(4, 0)
    reservoir.offer(1, "a", "web", "en", "s")
    reservoir.offer(2, "b", "code", "tr", "t")
    reservoir.save(path)

    loaded = builder.Reservoir.load(path, 4, 0)

    assert loaded.seen == 2
    assert loaded.items == [(1, "a", "web", "en", "s"), (2, "b", "code", "tr", "t")]


def test_v2_profiles_match_the_shared_corpus_design():
    full = builder.ATLAS_V2
    lite = builder.ATLAS_V2_LITE

    assert (full.dim, full.n_l1, full.l2_k_min, full.l2_k_max, full.l2_budget) == (
        128, 128, 1, 10, None,
    )
    assert (lite.dim, lite.n_l1, lite.l2_k_min, lite.l2_k_max, lite.l2_budget) == (
        64, 32, 1, 10, None,
    )
    assert full.pooling == lite.pooling == "sif"
    assert full.pca_k == lite.pca_k == 2
    assert full.max_chars == lite.max_chars == 2_000
    assert full.max_tokens == lite.max_tokens == 512


def test_global_l2_allocator_honors_exact_budget_and_parent_bounds():
    profile = builder.AtlasProfile(
        "test", 4, "mean", 100, 20, 100, 0, 4, 2, 4, 12,
    )
    counts = np.asarray([100, 200, 300, 400], dtype=np.int64)
    curves = [{2: 0.10, 3: 0.20, 4: 0.25} for _ in counts]

    allocated = builder.allocate_l2_budget(counts, curves, profile)

    assert int(allocated.sum()) == 12
    assert np.all((allocated >= 2) & (allocated <= 4))


def test_idf_sample_allocation_is_exact_and_proportional():
    rows = {"a": 70, "b": 20, "c": 10}

    allocated = builder.allocate_idf_sample(rows, 20)

    assert allocated == {"a": 14, "b": 4, "c": 2}
    assert sum(allocated.values()) == 20


def test_manifest_sources_resolves_every_shard_and_consume_keeps_it(tmp_path, monkeypatch):
    source = builder.Source(
        "public/example", None, "train", ("text",), "web", 10, "en",
        target_bytes=1_000, revision="frozen",
    )
    shard = builder.fetch_corpus.cached_shard_path(tmp_path, source.slug)
    shard.parent.mkdir(parents=True)
    text = "A public web record with enough distinct words to pass the minimum text filter."
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": text}) + "\n")
    before = shard.read_bytes()
    requested = asdict(source)
    requested["fields"] = list(source.fields)
    manifest = {
        "totals": {"rows": 1, "logical_bytes": len(text), "sources_with_rows": 1},
        "sources": [{"slug": source.slug, "rows": 1, "requested": requested}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    sources, loaded = builder.manifest_sources(tmp_path)
    assert loaded["totals"]["rows"] == 1
    assert [item.slug for item in sources] == [source.slug]

    monkeypatch.setattr(builder, "_prepare_batch", lambda texts, axes, seen: (texts, [0]))
    monkeypatch.setattr(builder, "_ingest_prepared", lambda *args, **kwargs: 1)
    corpus = builder.DiskCorpus(tmp_path / "work", 4)
    seen = builder.Uint64Set(tmp_path / "seen.u64", slots=1_024)
    consumed: set[str] = set()
    _, _, consumed = builder._consume_cache(
        tmp_path, sources, corpus, object(), seen,
        builder.Reservoir(2, 0), builder.Reservoir(2, 1), None, consumed,
    )

    assert consumed == {source.slug}
    assert shard.read_bytes() == before


def test_v3_profile_is_one_product_with_a_fixed_cell_budget():
    v3 = builder.ATLAS_V3

    assert (v3.dim, v3.n_l1, v3.l2_budget) == (128, 256, 4_096)
    # The budget is what makes L2 a chosen resolution rather than a discovered
    # one; k_max only bounds how lopsided a single region may get.
    assert v3.l2_k_max == 64
    assert v3.pooling == "sif" and v3.pca_k == 2
    # Same encode path as v2, so the corpus does not need re-tokenizing.
    assert (v3.max_chars, v3.max_tokens) == (2_000, 512)


def test_population_budget_is_exact_and_never_starves_a_region():
    counts = np.asarray([1_000_000, 250_000, 40_000, 9_000], dtype=np.int64)
    profile = builder.AtlasProfile(
        "test", 4, "sif", 100, 20, 100, 0, 4, 1, 64, 200,
    )

    allocated = builder.population_l2_budget(counts, profile)

    assert int(allocated.sum()) == 200
    assert allocated.min() >= 1
    # Sub-linear in population: the largest region holds 111x the text of the
    # smallest but must not take 111x the cells.
    assert allocated[0] / allocated[-1] < counts[0] / counts[-1]
    # Monotone in population.
    assert list(allocated) == sorted(allocated, reverse=True)


def test_population_budget_skips_empty_regions_and_respects_ceilings():
    counts = np.asarray([0, 0, 5, 500_000], dtype=np.int64)
    profile = builder.AtlasProfile(
        "test", 4, "sif", 100, 20, 100, 0, 4, 1, 64, 100,
    )

    allocated = builder.population_l2_budget(counts, profile)

    assert list(allocated[:2]) == [0, 0]
    # A region of five records cannot be split into more than four cells.
    assert allocated[2] <= 4
    # The budget asked for 100 but two regions are empty, one holds five
    # records and k_max caps the last at 64, so 68 is everything available.
    # An unreachable budget is clamped to the ceiling, never overshot.
    assert int(allocated.sum()) == 68


def test_fixed_k_kmeans_returns_the_requested_cells_on_separated_blobs():
    rng = np.random.default_rng(0)
    blobs = np.concatenate([
        rng.normal(centre, 0.05, size=(400, 8))
        for centre in (np.eye(8)[0], np.eye(8)[3], np.eye(8)[6])
    ]).astype(np.float32)
    blobs /= np.linalg.norm(blobs, axis=1, keepdims=True)

    labels, n_cells, score, centres = builder._fixed_k_kmeans(blobs, 3, seed=7)

    assert n_cells == 3
    assert len(np.unique(labels)) == 3
    # Three cells is only the right answer if they are the three blobs. A
    # single k-means start on this data once put two centroids in one blob and
    # one across the other two, which also counted to three.
    truth = np.repeat(np.arange(3), 400)
    for cell in range(3):
        assert len(np.unique(truth[labels == cell])) == 1
    # No score was consulted, so none is claimed.
    assert score == 0.0
    assert np.allclose(np.linalg.norm(centres, axis=1), 1.0, atol=1e-5)


def test_fixed_k_kmeans_never_asks_for_more_cells_than_records():
    tiny = np.eye(3, 8, dtype=np.float32)

    labels, n_cells, _, centres = builder._fixed_k_kmeans(tiny, 40, seed=7)

    assert n_cells <= len(tiny)
    assert len(centres) == n_cells
    assert labels.max() < n_cells


def test_cell_labels_drop_words_that_appear_in_most_cells():
    # "with" and "that" lead every cell's raw terms; nothing separates a cell
    # from its neighbours by carrying a word they all carry.
    terms = [
        ["with", "that", "pokemon", "niantic"],
        ["with", "that", "spiele", "kostenlos"],
        ["with", "that", "diptera", "flies"],
        ["with", "that", "insulin", "glycemic"],
    ]

    labels = builder.contrastive_cell_labels(terms, len(terms), width=2)

    assert labels[0] == "pokemon, niantic"
    assert labels[3] == "insulin, glycemic"
    assert not any("with" in label or "that" in label for label in labels)


def test_cell_labels_collapse_inflections_onto_one_stem():
    # Without stemming this spends both slots on one concept.
    terms = [["spiele", "spielen", "kostenlos", "gratis"], ["alpha", "beta", "gamma", "delta"]]

    labels = builder.contrastive_cell_labels(terms, 2, width=2)

    assert labels[0] == "spiele, kostenlos"


def test_cell_labels_are_defined_for_every_cell():
    terms = [["only"], []]

    labels = builder.contrastive_cell_labels(terms, 2)

    assert len(labels) == 2
    assert labels[1] == "cell 1"


def test_curated_region_labels_cover_atlas_v3():
    labels, source = builder.curated_region_labels(builder.ATLAS_V3, 4_096)

    assert source == "curated:region_labels_atlas-v3.json"
    assert len(labels) == 4_096
    assert all(label.strip() for label in labels)
    # Hand names describe subjects, never the language a cell happens to be
    # written in; the old file captioned cells "(German)".
    assert not any(label.endswith(")") for label in labels)


def test_curated_region_labels_are_bound_to_the_packaged_corpus():
    """A name file written for one clustering must not be applied to another."""
    payload = json.loads(
        builder.REGION_LABELS[builder.ATLAS_V3.version].read_text(encoding="utf-8")
    )
    packaged = np.load(
        builder.ROOT / "src" / "dropoutt" / "data" / "atlas" / "atlas-v3.npz",
        allow_pickle=True,
    )["meta"].item()
    meta = json.loads(packaged) if isinstance(packaged, str) else packaged

    assert payload["corpus_hash"] == meta["corpus_hash"]
    labels, source = builder.curated_region_labels(
        builder.ATLAS_V3, 4_096, corpus_hash="0" * 32
    )
    assert labels is None
    assert source == "profile-changed"
