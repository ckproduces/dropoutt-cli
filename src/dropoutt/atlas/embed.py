"""Embedding backend for the atlas.

Static embeddings only. model2vec's potion models are a token-id lookup into a
matrix followed by a mean pool, so there is no matmul, no ONNX runtime, and no
torch. That is what lets the atlas run on a CPU-only cluster node.

The loader fetches three files rather than calling ``snapshot_download``, because
that function is invoked without ``allow_patterns`` upstream and therefore also
pulls a duplicate ONNX copy of the weights, roughly doubling the download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..compat import HAVE_MODEL2VEC

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"
NEEDED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


class Embedder:
    """Wraps a static embedding model."""

    def __init__(self, model, name: str) -> None:
        self._model = model
        self.name = name

    @property
    def dim(self) -> int:
        return int(self._model.dim)

    def encode(self, texts: list[str], batch_size: int = 1024) -> np.ndarray:
        vecs = self._model.encode(
            texts, batch_size=batch_size, max_length=512, show_progress_bar=False
        )
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


def local_model_dir(model_id: str, cache_root: Path) -> Path:
    return cache_root / "embedder" / model_id.split("/")[-1]


def load(
    model_id: str = DEFAULT_MODEL,
    *,
    cache_root: Path | None = None,
    offline: bool = False,
) -> Embedder | None:
    """Load the embedder, downloading only what is needed. None if unavailable."""
    if not HAVE_MODEL2VEC:
        return None
    from model2vec import StaticModel  # noqa: PLC0415

    if cache_root is None:
        from ..config import cache_dir  # noqa: PLC0415

        cache_root = cache_dir()

    local = local_model_dir(model_id, cache_root)
    have_all = local.exists() and all((local / f).exists() for f in NEEDED_FILES)

    if not have_all:
        if offline:
            return None
        try:
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            local.mkdir(parents=True, exist_ok=True)
            for name in NEEDED_FILES:
                if not (local / name).exists():
                    src = hf_hub_download(model_id, name)
                    (local / name).write_bytes(Path(src).read_bytes())
        except Exception:
            return None

    try:
        return Embedder(StaticModel.from_pretrained(str(local)), model_id)
    except Exception:
        return None
