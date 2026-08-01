"""Shared atlas pipeline: extraction, chunking, normalization, soft assign."""

from __future__ import annotations

import json

import numpy as np

from dropoutt.atlas.apply import Atlas
from dropoutt.atlas.chunk import chunk_text
from dropoutt.atlas.embed import Embedder
from dropoutt.atlas.extract import detect_format, extract_text
from dropoutt.atlas.normalize import fit_norm
from dropoutt.atlas.pipeline import pipeline_hash


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


class _StaticModel:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.embedding = np.arange(40, dtype=np.float32).reshape(10, 4)
        self.dim = 4


def test_sparse_sif_pool_matches_weighted_token_average():
    model = _StaticModel()
    base = Embedder(model, "fake", out_dim=3)
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
        expected[row] = weights @ model.embedding[token_ids, :3]

    assert np.allclose(actual, expected, atol=1e-6)
    assert tokens.indptr.tolist() == [0, 3, 3, 5]
    assert ids.tolist() == [1, 2, 3]
    assert np.all(np.diff(log_probs) <= 0)


def test_weighted_encode_batch_tokenizes_each_text_once():
    model = _StaticModel()
    base = Embedder(model, "fake", out_dim=3)
    tokens = base.tokenize(["1 2", "2 3", "3 4"])
    probs, _, _ = base.token_log_prob(tokens)
    model.tokenizer.calls = 0

    result = base.bind_idf(probs).encode(["1 2", "2 3", "3 4"], batch_size=64)

    assert result.shape == (3, 3)
    assert model.tokenizer.calls == 1


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
    cats = atlas.categorize(centroids[:3])
    assert cats.shape == (3,)
