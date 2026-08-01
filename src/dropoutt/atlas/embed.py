"""Embedding backend for the atlas.

Static embeddings only. model2vec's potion models are a token-id lookup into a
matrix followed by a pool, so there is no matmul, no ONNX runtime, and no
torch. That is what lets the atlas run on a CPU-only cluster node.

The loader fetches three files rather than calling ``snapshot_download``, because
that function is invoked without ``allow_patterns`` upstream and therefore also
pulls a duplicate ONNX copy of the weights, roughly doubling the download.

Pooling is SIF-weighted when an IDF table is supplied (``w = a/(a+p)``), else
plain mean. Output is truncated 256 → 128 (Matryoshka) before return; L2 and
anisotropy correction happen in :mod:`normalize`, not here.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from ..compat import HAVE_MODEL2VEC
from .normalize import EMBED_DIM, SIF_A, sif_weights_from_probs, truncate

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"
NEEDED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


class Embedder:
    """Wraps a static embedding model with optional IDF-weighted pooling."""

    def __init__(
        self,
        model,
        name: str,
        *,
        token_log_prob: dict[int, float] | None = None,
        out_dim: int = EMBED_DIM,
        weight_hash: str = "",
    ) -> None:
        self._model = model
        self.name = name
        self.out_dim = out_dim
        self._token_log_prob = token_log_prob or {}
        self._weight_hash = weight_hash

    @property
    def dim(self) -> int:
        return int(self.out_dim)

    @property
    def full_dim(self) -> int:
        return int(self._model.dim)

    @property
    def weight_hash(self) -> str:
        return self._weight_hash

    def encode(
        self,
        texts: list[str],
        batch_size: int = 1024,
        *,
        weighted: bool | None = None,
    ) -> np.ndarray:
        """Embed texts to ``out_dim`` (default 128). Not L2-normalised yet."""
        use_weighted = bool(self._token_log_prob) if weighted is None else weighted
        if use_weighted and self._token_log_prob:
            return self._encode_weighted(texts, batch_size=batch_size)
        vecs = self._model.encode(
            texts, batch_size=batch_size, max_length=512, show_progress_bar=False
        )
        return truncate(np.asarray(vecs, dtype=np.float32), self.out_dim)

    def _encode_weighted(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        # Direct table lookup keeps token ids aligned with vectors. model2vec's
        # encode_as_sequence returns vectors without ids; re-tokenising to
        # recover them can drift if the internal path differs.
        del batch_size
        tokenizer = self._model.tokenizer
        table = np.asarray(self._model.embedding, dtype=np.float32)
        # Dense log-prob table: O(1) lookup beats a Python dict per token.
        max_id = int(table.shape[0])
        log_probs = np.full(max_id, -12.0, dtype=np.float32)
        for tid, lp in self._token_log_prob.items():
            if 0 <= tid < max_id:
                log_probs[tid] = lp
        out = np.zeros((len(texts), self.out_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False).ids[:512]
            if not ids:
                continue
            idx = np.asarray(ids, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < max_id)]
            if idx.size == 0:
                continue
            vecs = table[idx]
            probs = np.exp(log_probs[idx])
            weights = sif_weights_from_probs(probs, a=SIF_A)
            weights = weights / (weights.sum() + 1e-9)
            out[i] = (weights @ vecs)[: self.out_dim]
        return out

    def bind_idf(self, token_log_prob: dict[int, float]) -> Embedder:
        """Return a new wrapper sharing the model but using ``token_log_prob``."""
        return Embedder(
            self._model,
            self.name,
            token_log_prob=token_log_prob,
            out_dim=self.out_dim,
            weight_hash=self._weight_hash,
        )


def local_model_dir(model_id: str, cache_root: Path) -> Path:
    return cache_root / "embedder" / model_id.rsplit("/", maxsplit=1)[-1]


#: One embedder per process. Loading the weights is a second and a half of
#: fixed cost, and the CLI starts it in the background before the scan needs it,
#: which only works if the scan then finds it here. The lock matters as much as
#: the cache: without it the background thread and the scan both start a load,
#: and two loads of the same half-gigabyte are slower than one.
_LOADED: dict[tuple[str, bool, int], Embedder | None] = {}
_LOAD_LOCK = threading.Lock()


def load(
    model_id: str = DEFAULT_MODEL,
    *,
    cache_root: Path | None = None,
    offline: bool = False,
    out_dim: int = EMBED_DIM,
) -> Embedder | None:
    """Load the embedder, downloading only what is needed. None if unavailable."""
    if cache_root is not None:
        return _load_uncached(
            model_id, cache_root=cache_root, offline=offline, out_dim=out_dim
        )
    key = (model_id, offline, out_dim)
    with _LOAD_LOCK:
        if key not in _LOADED:
            _LOADED[key] = _load_uncached(
                model_id, offline=offline, out_dim=out_dim
            )
        return _LOADED[key]


def _load_uncached(
    model_id: str,
    *,
    cache_root: Path | None = None,
    offline: bool = False,
    out_dim: int = EMBED_DIM,
) -> Embedder | None:
    if not HAVE_MODEL2VEC:
        return None
    from model2vec import StaticModel

    if cache_root is None:
        from ..config import cache_dir

        cache_root = cache_dir()

    local = local_model_dir(model_id, cache_root)
    have_all = local.exists() and all((local / f).exists() for f in NEEDED_FILES)

    if not have_all:
        if offline:
            return None
        try:
            from huggingface_hub import hf_hub_download

            local.mkdir(parents=True, exist_ok=True)
            for name in NEEDED_FILES:
                if not (local / name).exists():
                    src = hf_hub_download(model_id, name)
                    (local / name).write_bytes(Path(src).read_bytes())
        except Exception:
            return None

    try:
        wh = _file_hash(local / "model.safetensors")
        return Embedder(
            StaticModel.from_pretrained(str(local)),
            model_id,
            out_dim=out_dim,
            weight_hash=wh,
        )
    except Exception:
        return None


def _file_hash(path: Path | None) -> str:
    if path is not None and path.exists():
        h = hashlib.blake2b(digest_size=16)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return ""


def embedding_weight_hash(model_id: str = DEFAULT_MODEL) -> str:
    """Blake2b of the local ``model.safetensors``, or empty if missing."""
    from ..config import cache_dir

    path = local_model_dir(model_id, cache_dir()) / "model.safetensors"
    return _file_hash(path if path.exists() else None)
