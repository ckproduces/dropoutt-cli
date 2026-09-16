"""What the atlas encoder reads: text folding, token damping, template phrases."""

from __future__ import annotations

import numpy as np
import pytest

from dropoutt.atlas.apply import Atlas
from dropoutt.atlas.embed import Embedder, QuantizedTable
from dropoutt.atlas.textnorm import (
    ENCODER_INPUT_V1,
    LATEST_VERSION,
    PHRASE_N,
    RAW_INPUT,
    EncoderInput,
    _piece_class,
    fold_caps_runs,
    phrase_hashes,
    phrase_mask,
    source_phrases,
    strip_rules,
    token_factors,
)


def test_caps_runs_are_sentence_cased_and_acronyms_are_left_alone():
    assert (
        fold_caps_runs("INFORMATIONEN ÜBER DATENSCHUTZERKLÄRUNG und mehr")
        == "Informationen Über Datenschutzerklärung und mehr"
    )
    assert fold_caps_runs("The EU and NASA signed it in HTML.") == (
        "The EU and NASA signed it in HTML."
    )
    assert fold_caps_runs("ПРАВИЛА ИСПОЛЬЗОВАНИЯ САЙТА") == "Правила Использования Сайта"
    # Two capitalised words are a heading fragment, not a run.
    assert fold_caps_runs("BREAKING NEWS: markets fell") == "BREAKING NEWS: markets fell"
    # Digits and punctuation between the words do not break a run.
    assert fold_caps_runs("PROJECT 7, NEW SERIES") == "Project 7, New Series"


def test_table_rules_become_spaces_and_prose_punctuation_stays():
    assert strip_rules("|Name||Value|").split() == ["Name", "Value"]
    assert strip_rules("Title\n-----\nBody").split() == ["Title", "Body"]
    assert strip_rules("e.g. a - b, see p. 4") == "e.g. a - b, see p. 4"


@pytest.mark.parametrize(
    ("piece", "kind"),
    [
        ("▁T", "short"),
        ("TT", "short"),
        ("▁de", "short"),
        ("▁的", "word"),   # CJK: no letter case, so a one-character word stays a word
        ("ال", "word"),    # Arabic: likewise
        ("▁|", "symbol"),
        ("▁2015", "digit"),
        ("▁dog", "word"),
        ("▁", "word"),
    ],
)
def test_vocabulary_pieces_are_classified_by_surface(piece, kind):
    assert _piece_class(piece) == kind


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


_VOCAB = {"▁pad": 0, "▁dog": 1, "▁T": 2, "▁|": 3, "▁2015": 4, "▁cat": 5, "▁run": 6, "▁far": 7}


class _Tokenizer:
    """Whitespace-separated integer ids, with a vocabulary of named pieces."""

    def encode_batch(self, texts, add_special_tokens=False):
        return [_Encoding([int(part) for part in text.split()]) for text in texts]

    def get_vocab(self):
        return dict(_VOCAB)


def _encoder():
    rng = np.random.default_rng(1)
    table = QuantizedTable.from_float(rng.normal(size=(8, 4)).astype(np.float32), width=4)
    return Embedder(table, _Tokenizer(), "fake", out_dim=4), table


def test_raw_input_reads_exactly_what_the_encoder_read_before():
    base, _ = _encoder()
    texts = ["1 2 3", "4 5", "1 1 6 7"]
    assert np.array_equal(base.encode(texts), base.bind_input(RAW_INPUT).encode(texts))


def test_damped_tokens_count_for_less_in_the_pool():
    base, table = _encoder()
    policy = EncoderInput(version=1, short_cased_weight=0.2, symbol_weight=0.5, digit_weight=0.25)
    tokens = base.tokenize(["1 2 3 4"])
    actual = base.bind_input(policy).encode_tokenized(tokens)

    weights = np.array([1.0, 0.2, 0.5, 0.25], dtype=np.float32)
    weights /= weights.sum()
    expected = weights @ table.rows(np.array([1, 2, 3, 4]), 4)
    assert np.allclose(actual[0], expected, atol=1e-6)


def test_a_record_made_only_of_damped_tokens_keeps_its_direction():
    base, _ = _encoder()
    policy = EncoderInput(version=1, short_cased_weight=0.2)
    before = base.encode(["2 2"], weighted=False)
    after = base.bind_input(policy).encode(["2 2"], weighted=False)
    assert np.allclose(before, after, atol=1e-6)


def test_factors_leave_word_pieces_untouched():
    _, table = _encoder()
    factors = token_factors(_Tokenizer(), table.n_rows, ENCODER_INPUT_V1)
    assert factors.tolist() == pytest.approx([1.0, 1.0, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0])


def test_phrase_windows_never_cross_a_document_boundary():
    ids = np.array([1, 5, 6, 7, 1, 5, 6, 7], dtype=np.int32)
    table = np.unique(phrase_hashes(np.array([1, 5, 6, 7])))
    one_doc = phrase_mask(ids, np.array([0, 8]), table)
    assert one_doc.tolist() == [True] * 8

    split = phrase_mask(ids, np.array([0, 2, 6, 8]), table)
    assert not split.any()


def test_template_phrases_need_a_share_of_one_source():
    template = [11, 12, 13, 14]
    docs = [np.array([*template, 100 + i, 200 + i, 300 + i]) for i in range(300)]
    found = source_phrases(docs, share=0.1, min_docs=20, min_source_docs=200)
    assert found.tolist() == np.unique(phrase_hashes(np.array(template))).tolist()

    rare = [np.array([*template, 1, 2, 3]) for _ in range(20)] + [
        np.array([500 + i, 600 + i, 700 + i, 800 + i]) for i in range(280)
    ]
    assert not len(source_phrases(rare, share=0.1, min_docs=20, min_source_docs=200))
    assert not len(source_phrases(docs[:100], share=0.1, min_docs=20, min_source_docs=200))


def test_damped_phrases_are_down_weighted_inside_the_pool():
    base, table = _encoder()
    phrase = np.array([1, 5, 6, 7])
    policy = EncoderInput(version=1, phrase_weight=0.1).with_phrases(phrase_hashes(phrase))
    actual = base.bind_input(policy).encode_tokenized(base.tokenize(["1 5 6 7 2"]))

    weights = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
    weights /= weights.sum()
    expected = weights @ table.rows(np.array([1, 5, 6, 7, 2]), 4)
    assert np.allclose(actual[0], expected, atol=1e-6)


def test_policy_round_trips_through_an_artifact_declaration():
    hashes = phrase_hashes(np.array([1, 2, 3, 4, 5, 6]))
    policy = ENCODER_INPUT_V1.with_phrases(hashes)
    declared = policy.declaration()
    restored = EncoderInput.from_artifact(declared, policy.phrase_hashes)

    assert restored == policy
    assert restored.phrase_hashes.tolist() == policy.phrase_hashes.tolist()
    assert declared["phrase_n"] == PHRASE_N

    with pytest.raises(ValueError, match="digest"):
        EncoderInput.from_artifact(declared, policy.phrase_hashes[::-1] + np.uint64(1))
    with pytest.raises(ValueError, match="phrase table"):
        EncoderInput.from_artifact(declared, None)
    with pytest.raises(ValueError, match="Upgrade"):
        EncoderInput.from_artifact({"version": LATEST_VERSION + 1})
    assert EncoderInput.from_artifact(None) is RAW_INPUT


def test_atlas_binds_its_own_policy_and_leaves_raw_maps_alone(tmp_path):
    base, _ = _encoder()
    atlas = Atlas(
        centroids=np.eye(4, dtype=np.float32),
        region_category=np.zeros(4, dtype=np.int32),
        coords=np.zeros((4, 2), dtype=np.float32),
        probe_coef=np.zeros((0, 4), dtype=np.float32),
        probe_intercept=np.zeros(0, dtype=np.float32),
        probe_classes=np.zeros(0, dtype=np.int32),
    )
    assert atlas.bind_embedder(base) is base

    atlas.encoder_input = ENCODER_INPUT_V1
    bound = atlas.bind_embedder(base)
    assert bound.encoder_input == ENCODER_INPUT_V1
