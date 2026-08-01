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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..compat import HAVE_MODEL2VEC
from .normalize import EMBED_DIM, SIF_A, sif_weights_from_probs, truncate

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"
NEEDED_FILES = ("config.json", "model.safetensors", "tokenizer.json")


@dataclass(frozen=True)
class TokenizedCorpus:
    """Compact token cache shared by IDF fitting and SIF pooling.

    ``token_ids[indptr[i]:indptr[i + 1]]`` is document ``i``. Keeping the
    flattened representation avoids millions of Python integers and maps
    directly to SciPy's CSR constructor.
    """

    token_ids: np.ndarray
    indptr: np.ndarray
    n_docs: int
    max_length: int

    @property
    def n_tokens(self) -> int:
        return int(self.token_ids.size)


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
            tokens = self.tokenize(texts, batch_size=max(batch_size, 1024))
            return self.encode_tokenized(tokens)
        vecs = self._model.encode(
            texts, batch_size=batch_size, max_length=512, show_progress_bar=False
        )
        return truncate(np.asarray(vecs, dtype=np.float32), self.out_dim)

    def tokenize(
        self,
        texts: list[str],
        *,
        batch_size: int = 8192,
        max_length: int = 512,
    ) -> TokenizedCorpus:
        """Batch-tokenize once into flat token IDs and document offsets."""

        tokenizer = self._model.tokenizer
        parts: list[np.ndarray] = []
        lengths = np.zeros(len(texts), dtype=np.int64)

        for start in range(0, len(texts), batch_size):
            encodings = tokenizer.encode_batch(
                texts[start : start + batch_size], add_special_tokens=False
            )
            for offset, encoding in enumerate(encodings):
                ids = encoding.ids[:max_length]
                lengths[start + offset] = len(ids)
                if ids:
                    parts.append(np.asarray(ids, dtype=np.int32))

        token_ids = (
            np.concatenate(parts)
            if parts
            else np.zeros(0, dtype=np.int32)
        )
        indptr = np.empty(len(texts) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        return TokenizedCorpus(
            token_ids=token_ids,
            indptr=indptr,
            n_docs=len(texts),
            max_length=max_length,
        )

    def encode_tokenized(self, tokens: TokenizedCorpus) -> np.ndarray:
        """SIF-pool a token cache with one sparse-dense matrix multiply."""

        from scipy.sparse import csr_matrix

        table = np.asarray(self._model.embedding)
        if tokens.n_tokens == 0:
            return np.zeros((tokens.n_docs, self.out_dim), dtype=np.float32)
        if int(tokens.token_ids.min()) < 0 or int(tokens.token_ids.max()) >= len(table):
            raise ValueError("token cache contains ids outside the embedding vocabulary")

        # Subsetting is material: potion has ~500K rows (~512 MB at 256-d), while
        # an Atlas corpus normally observes ~100K. The sparse operand indexes
        # this compact table rather than the full vocabulary.
        observed_ids, columns = np.unique(tokens.token_ids, return_inverse=True)

        # Dense probabilities make all token-weight lookups vectorized. Tokens
        # omitted from the shipped top-frequency table retain the previous
        # conservative p=exp(-12) fallback.
        log_probs = np.full(len(table), -12.0, dtype=np.float32)
        for token_id, log_prob in self._token_log_prob.items():
            if 0 <= token_id < len(table):
                log_probs[token_id] = log_prob
        probs = np.exp(log_probs[tokens.token_ids])
        weights = sif_weights_from_probs(probs, a=SIF_A)

        row_lengths = np.diff(tokens.indptr)
        row_sums = np.zeros(tokens.n_docs, dtype=np.float32)
        nonempty = np.flatnonzero(row_lengths)
        if nonempty.size:
            row_sums[nonempty] = np.add.reduceat(weights, tokens.indptr[nonempty])
        weights *= np.repeat(
            np.divide(
                1.0,
                row_sums,
                out=np.zeros_like(row_sums),
                where=row_sums > 0,
            ),
            row_lengths,
        )

        matrix = csr_matrix(
            (weights, columns.astype(np.int32, copy=False), tokens.indptr),
            shape=(tokens.n_docs, len(observed_ids)),
            dtype=np.float32,
        )
        matrix.sum_duplicates()
        vectors = matrix @ np.asarray(table[observed_ids, : self.out_dim], dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def token_log_prob(
        self,
        tokens: TokenizedCorpus,
        *,
        max_tokens: int | None = None,
    ) -> tuple[dict[int, float], np.ndarray, np.ndarray]:
        """Fit unigram probabilities from an existing token cache."""

        if tokens.n_tokens == 0:
            empty_ids = np.zeros(0, dtype=np.int32)
            empty_probs = np.zeros(0, dtype=np.float32)
            return {}, empty_ids, empty_probs

        token_ids, counts = np.unique(tokens.token_ids, return_counts=True)
        if max_tokens is not None and len(token_ids) > max_tokens:
            selected = np.argpartition(counts, -max_tokens)[-max_tokens:]
            token_ids = token_ids[selected]
            counts = counts[selected]
        order = np.argsort(-counts, kind="stable")
        token_ids = token_ids[order].astype(np.int32, copy=False)
        log_probs = np.log(counts[order] / tokens.n_tokens).astype(np.float32)
        mapping = {
            int(token_id): float(log_prob)
            for token_id, log_prob in zip(token_ids, log_probs, strict=True)
        }
        return mapping, token_ids, log_probs

    def _encode_weighted(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """Compatibility shim for callers using the old private helper."""

        tokens = self.tokenize(texts, batch_size=max(batch_size, 1024))
        if not self._token_log_prob:
            return np.zeros((len(texts), self.out_dim), dtype=np.float32)
        return self.encode_tokenized(tokens)

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
