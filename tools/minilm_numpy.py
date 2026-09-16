"""An encoder that is not the atlas's, for judging whether a cell holds one subject.

The atlas's own encoder cannot audit the atlas. A cell built by it is coherent to
it by construction: on atlas-v3 before the encoder input policy, cells whose
members shared only a first letter or ALL-CAPS spelling scored as tight as any
subject cell. A contextual sentence encoder reads the same records differently
enough to tell a subject from a spelling.

``paraphrase-multilingual-MiniLM-L12-v2`` is small, multilingual (50+
languages, cross-lingual paraphrases at cosine 0.88-0.92) and already in the
Hugging Face cache on the build machine. It runs here as plain numpy on its
safetensors weights and ``tokenizers``, so auditing needs no torch: about 40
records a second per process at 128 tokens.
"""

from __future__ import annotations

import json
import os
import struct
from functools import cache
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS = 128
HIDDEN, HEADS, LAYERS = 384, 12, 12


def snapshot() -> Path:
    """Local snapshot directory of :data:`MODEL`; raises if it was never downloaded."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    base = root / ("models--" + MODEL.replace("/", "--")) / "snapshots"
    for candidate in sorted(base.glob("*")) if base.is_dir() else []:
        if (candidate / "model.safetensors").exists() and (candidate / "tokenizer.json").exists():
            return candidate
    raise FileNotFoundError(
        f"{MODEL} is not in the Hugging Face cache; download it once with "
        f"`huggingface-cli download {MODEL}`"
    )


@cache
def _weights() -> dict[str, np.ndarray]:
    path = snapshot() / "model.safetensors"
    with path.open("rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(size))
        buffer = fh.read()
    dtypes = {"F32": np.float32, "I64": np.int64}
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = dtypes[entry["dtype"]]
        start, stop = entry["data_offsets"]
        out[name] = np.frombuffer(
            buffer, dtype=dtype, count=(stop - start) // np.dtype(dtype).itemsize, offset=start
        ).reshape(entry["shape"])
    return out


@cache
def _tokenizer():
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    tokenizer.enable_truncation(MAX_TOKENS)
    return tokenizer


def _layer_norm(x: np.ndarray, prefix: str) -> np.ndarray:
    w = _weights()
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-12) * w[prefix + ".weight"] + w[prefix + ".bias"]


def _linear(x: np.ndarray, prefix: str) -> np.ndarray:
    w = _weights()
    return x @ w[prefix + ".weight"].T + w[prefix + ".bias"]


def _erf(x: np.ndarray) -> np.ndarray:
    """Chebyshev fit of erf, accurate to 1.2e-7 (Numerical Recipes ``erfcc``)."""
    sign = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * a)
    poly = -a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    return (sign * (1.0 - t * np.exp(poly))).astype(np.float32)


def _forward(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    w = _weights()
    batch, length = ids.shape
    head_dim = HIDDEN // HEADS
    x = (
        w["embeddings.word_embeddings.weight"][ids]
        + w["embeddings.position_embeddings.weight"][:length][None]
        + w["embeddings.token_type_embeddings.weight"][0]
    )
    x = _layer_norm(x, "embeddings.LayerNorm")
    bias = ((1.0 - mask[:, None, None, :]) * -1e9).astype(np.float32)
    scale = np.float32(np.sqrt(head_dim))
    for layer in range(LAYERS):
        p = f"encoder.layer.{layer}."
        q = _linear(x, p + "attention.self.query").reshape(batch, length, HEADS, head_dim).transpose(0, 2, 1, 3)
        k = _linear(x, p + "attention.self.key").reshape(batch, length, HEADS, head_dim).transpose(0, 2, 3, 1)
        v = _linear(x, p + "attention.self.value").reshape(batch, length, HEADS, head_dim).transpose(0, 2, 1, 3)
        att = (q @ k) / scale + bias
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        h = (att @ v).transpose(0, 2, 1, 3).reshape(batch, length, HIDDEN)
        x = _layer_norm(x + _linear(h, p + "attention.output.dense"), p + "attention.output.LayerNorm")
        h = _linear(x, p + "intermediate.dense")
        h = 0.5 * h * (1.0 + _erf(h / np.float32(np.sqrt(2.0))))
        x = _layer_norm(x + _linear(h, p + "output.dense"), p + "output.LayerNorm")
    m = mask[:, :, None]
    pooled = (x * m).sum(1) / np.maximum(m.sum(1), 1)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


def encode(texts: list[str], batch: int = 64) -> np.ndarray:
    """Mean-pooled, L2-normalised sentence vectors, (n, 384) float32."""
    encodings = _tokenizer().encode_batch(list(texts))
    order = np.argsort([len(e.ids) for e in encodings], kind="stable")
    out = np.zeros((len(texts), HIDDEN), dtype=np.float32)
    for start in range(0, len(texts), batch):
        rows = order[start:start + batch]
        length = max((len(encodings[i].ids) for i in rows), default=1) or 1
        ids = np.zeros((len(rows), length), dtype=np.int64)
        mask = np.zeros((len(rows), length), dtype=np.float32)
        for j, i in enumerate(rows):
            n = len(encodings[i].ids)
            ids[j, :n] = encodings[i].ids
            mask[j, :n] = 1.0
        out[rows] = _forward(ids, mask)
    return out


def encode_parallel(texts: list[str], workers: int = 4, threads: int = 3) -> np.ndarray:
    """:func:`encode` across worker processes; BLAS threads per worker capped."""
    from concurrent.futures import ProcessPoolExecutor

    if workers <= 1 or len(texts) < 4 * 2048:
        return encode(texts)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    chunks = [texts[i::workers] for i in range(workers)]
    out = np.zeros((len(texts), HIDDEN), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part, vectors in enumerate(pool.map(encode, chunks)):
            out[part::workers] = vectors
    return out


def language_centred(vectors: np.ndarray, languages: list[str], floor: int = 200) -> np.ndarray:
    """Remove each language's mean vector (languages with ``floor`` rows or more).

    The model aligns languages but not perfectly: two records in the same
    language sit a little closer than their subjects warrant. Centring each
    language on its own mean keeps coherence a statement about subject.
    """
    x = np.asarray(vectors, dtype=np.float32).copy()
    langs = np.asarray(languages)
    values, counts = np.unique(langs, return_counts=True)
    for value, count in zip(values, counts, strict=True):
        if count >= floor:
            rows = langs == value
            x[rows] -= np.asarray(vectors, dtype=np.float32)[rows].mean(0)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def cell_coherence(vectors: np.ndarray, cells: np.ndarray, n_cells: int, minimum: int = 5) -> np.ndarray:
    """Mean pairwise cosine of each cell's members; NaN below ``minimum`` members.

    Reference points measured on atlas-v3 (13 Sep 2026): two records drawn from
    anywhere score 0.00, two from different cells of one district 0.17, a median
    cell 0.25; cells a blind reader judged to be grab-bags sat mostly under 0.15.
    """
    cells = np.asarray(cells)
    out = np.full(n_cells, np.nan)
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(n_cells + 1))
    for cell in range(n_cells):
        rows = order[bounds[cell]:bounds[cell + 1]]
        if len(rows) >= minimum:
            sims = vectors[rows] @ vectors[rows].T
            out[cell] = (sims.sum() - np.trace(sims)) / (len(rows) * (len(rows) - 1))
    return out
