"""Focused, dependency-light checks for the dual Atlas v2 build entry point."""

from __future__ import annotations

import importlib.util
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

    labels, k, score, centres = builder._best_kmeans(vectors, 4, 10, seed=0)

    assert 4 <= k <= 10
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


def test_reservoir_roundtrip(tmp_path):
    path = tmp_path / "reservoir.jsonl"
    reservoir = builder.Reservoir(4, 0)
    reservoir.offer(1, "a", "web", "en", "s")
    reservoir.offer(2, "b", "code", "tr", "t")
    reservoir.save(path)

    loaded = builder.Reservoir.load(path, 4, 0)

    assert loaded.seen == 2
    assert loaded.items == [(1, "a", "web", "en", "s"), (2, "b", "code", "tr", "t")]
