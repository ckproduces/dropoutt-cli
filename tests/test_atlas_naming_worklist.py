"""The member draw that atlas namers and blind readers are shown."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

WORKLIST = Path(__file__).resolve().parents[1] / "tools" / "atlas_naming_worklist.py"
SPEC = importlib.util.spec_from_file_location("atlas_naming_worklist", WORKLIST)
assert SPEC is not None and SPEC.loader is not None
worklist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worklist)


def _cell(size: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    members = rng.permutation(10_000)[:size]
    return members, rng.random(size)


def _ranks(members: np.ndarray, similarity: np.ndarray, picked: np.ndarray) -> list[int]:
    ranked = members[np.argsort(-similarity, kind="stable")].tolist()
    return [ranked.index(int(m)) for m in picked]


def test_spread_sample_takes_one_member_from_every_band_centre_to_edge():
    members, similarity = _cell(300)
    for seed in range(20):
        picked = worklist.spread_sample(members, similarity, 30, np.random.default_rng(seed))
        assert _ranks(members, similarity, picked) == sorted(_ranks(members, similarity, picked))
        assert [rank // 10 for rank in _ranks(members, similarity, picked)] == list(range(30))


def test_spread_sample_varies_within_bands_and_repeats_under_a_seed():
    members, similarity = _cell(300)
    first = worklist.spread_sample(members, similarity, 30, np.random.default_rng(1))
    again = worklist.spread_sample(members, similarity, 30, np.random.default_rng(1))
    other = worklist.spread_sample(members, similarity, 30, np.random.default_rng(2))
    assert first.tolist() == again.tolist()
    assert first.tolist() != other.tolist()


def test_spread_sample_shows_a_small_cell_whole_in_rank_order():
    members, similarity = _cell(12)
    picked = worklist.spread_sample(members, similarity, 30, np.random.default_rng(0))
    assert _ranks(members, similarity, picked) == list(range(12))


def test_spread_sample_passes_over_excluded_members():
    members, similarity = _cell(200)
    excluded = frozenset(members[::5].tolist())
    picked = worklist.spread_sample(
        members, similarity, 16, np.random.default_rng(0), exclude=excluded
    )
    assert len(picked) == 16
    assert not excluded.intersection(picked.tolist())
    assert _ranks(members, similarity, picked) == sorted(_ranks(members, similarity, picked))


def test_spread_sample_tops_up_from_excluded_when_the_cell_is_short():
    members, similarity = _cell(20)
    excluded = frozenset(members[:10].tolist())
    picked = worklist.spread_sample(
        members, similarity, 16, np.random.default_rng(0), exclude=excluded
    )
    assert len(picked) == len(set(picked.tolist())) == 16
    assert set(members[10:].tolist()) <= set(picked.tolist())
    assert _ranks(members, similarity, picked) == sorted(_ranks(members, similarity, picked))


def test_rank_percent_counts_members_closer_to_the_centre():
    percent = worklist.rank_percent(np.array([0.2, 0.9, 0.5, 0.7]))
    assert percent.tolist() == [75.0, 0.0, 50.0, 25.0]
