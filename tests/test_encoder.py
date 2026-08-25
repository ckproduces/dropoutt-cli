"""The stored encoder: how it is read, how it is quantised, and what that costs.

The atlas is applied through a quantised, 128-column form of the published
weights. Two things have to hold for that to be honest. The conversion must be
deterministic, so that ``encoder_weight_hash`` is a statement about the
coordinate system rather than about one machine. And the loss has to stay where
it was measured: bounded per row, and small enough that records land where they
landed.
"""

from __future__ import annotations

import json
import struct

import numpy as np

from dropoutt.atlas.embed import (
    ENCODE_CHUNK,
    GPU_MIN_DOCS,
    QUANT_FORMAT,
    Embedder,
    QuantizedTable,
    convert,
    read_safetensors,
)


def _write_safetensors(path, array, name="embeddings"):
    raw = array.astype(np.float32).tobytes()
    header = json.dumps(
        {name: {"dtype": "F32", "shape": list(array.shape), "data_offsets": [0, len(raw)]}}
    ).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(raw)


def test_safetensors_is_read_without_a_safetensors_dependency(tmp_path):
    table = np.arange(24, dtype=np.float32).reshape(4, 6)
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, table)

    assert np.array_equal(read_safetensors(path, "embeddings"), table)


def test_conversion_is_deterministic_and_keeps_only_the_columns_in_use(tmp_path):
    """Two installs must derive byte-identical codes from identical weights."""
    rng = np.random.default_rng(3)
    table = rng.normal(size=(200, 256)).astype(np.float32)
    source = tmp_path / "src"
    source.mkdir()
    _write_safetensors(source / "model.safetensors", table)
    (source / "config.json").write_text(json.dumps({"normalize": True}))

    convert(source, tmp_path / "a", width=128)
    convert(source, tmp_path / "b", width=128)

    codes_a = np.load(tmp_path / "a" / "codes.npy")
    codes_b = np.load(tmp_path / "b" / "codes.npy")
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())

    assert codes_a.dtype == np.int8
    assert codes_a.shape == (200, 128)
    assert np.array_equal(codes_a, codes_b)
    assert manifest["format"] == QUANT_FORMAT
    assert manifest["width"] == 128
    assert manifest["scheme"] == "int8-row-scale"
    assert manifest["weight_hash"]
    assert manifest["source_weight_hash"] != manifest["weight_hash"]


def test_a_row_of_zeros_dequantises_to_zeros_rather_than_nan():
    table = np.zeros((3, 8), dtype=np.float32)
    table[1] = 1.0
    quantized = QuantizedTable.from_float(table, width=8)

    restored = quantized.rows(np.arange(3))

    assert np.all(np.isfinite(restored))
    assert not restored[0].any()
    assert not restored[2].any()


def test_the_device_multiply_threshold_is_reachable():
    """It was not. GPU_MIN_DOCS sat above the chunk size the encoder uses, so
    no batch could ever reach it and the device path shipped as dead code."""
    assert GPU_MIN_DOCS < ENCODE_CHUNK


def test_the_encoder_never_reports_a_width_the_table_does_not_have():
    table = QuantizedTable.from_float(np.ones((4, 32), dtype=np.float32), width=32)

    embedder = Embedder(table, object(), "fake", out_dim=128)

    assert embedder.dim == 32
