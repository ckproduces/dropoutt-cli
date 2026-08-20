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

#: Documents encoded per pass in the weighted path. At 512 tokens each this is
#: about eight million token ids, so the six transient arrays that
#: :meth:`Embedder.encode_tokenized` builds over them are tens of megabytes
#: rather than gigabytes. Large enough that the sparse multiply is still one
#: sizeable BLAS call rather than thousands of small ones.
ENCODE_CHUNK = 16_384


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
    def weight_hash(self) -> str:
        return self._weight_hash

    def encode(
        self,
        texts: list[str],
        batch_size: int = 1024,
        *,
        weighted: bool | None = None,
    ) -> np.ndarray:
        """Embed texts to ``out_dim`` (default 128). Not L2-normalised yet.

        Chunked, and that is the whole reason this method is not two lines. The
        weighted path used to tokenize the entire sample at once, which for two
        hundred thousand documents of up to 512 tokens means a hundred million
        token ids — and then, in ``encode_tokenized``, six more arrays of that
        same length for the probabilities, the SIF weights, the column indices
        and the sparse matrix. Together that is several gigabytes of transient,
        all of it to produce a result that is 200,000 x 128 floats: a hundred
        megabytes. It was the largest single allocation a scan made, and it was
        pure working set.

        Chunking changes nothing about the answer — SIF pooling is per document,
        so document *i*'s vector does not depend on document *j* — and caps the
        transient at :data:`ENCODE_CHUNK` documents' worth.
        """
        use_weighted = bool(self._token_log_prob) if weighted is None else weighted
        if use_weighted and self._token_log_prob:
            if len(texts) <= ENCODE_CHUNK:
                return self.encode_tokenized(
                    self.tokenize(texts, batch_size=max(batch_size, 1024))
                )
            out = np.zeros((len(texts), self.out_dim), dtype=np.float32)
            for start in range(0, len(texts), ENCODE_CHUNK):
                part = texts[start : start + ENCODE_CHUNK]
                out[start : start + len(part)] = self.encode_tokenized(
                    self.tokenize(part, batch_size=max(batch_size, 1024))
                )
            return out
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
        """SIF-pool a token cache with one sparse-dense matrix multiply.

        The multiply is the only real arithmetic in a scan: two hundred thousand
        documents by up to 512 tokens by 128 dimensions is around thirteen
        billion multiply-accumulates. On a CPU that is seconds; on a GPU it is
        not, so :func:`_gpu_matmul` takes it when the machine has one and the
        problem is big enough to pay for moving the operands across the bus.
        Everything before this line — tokenizing, weighting — stays on the CPU,
        because it is memory shuffling rather than arithmetic.
        """

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
        dense = np.asarray(table[observed_ids, : self.out_dim], dtype=np.float32)
        vectors = _gpu_matmul(matrix, dense)
        if vectors is None:
            vectors = matrix @ dense
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


#: Below this many documents the sparse multiply finishes on a CPU before a GPU
#: has finished being handed the operands. Moving a small matrix to a device and
#: back is dominated by the transfer and by the one-off cost of initialising the
#: runtime, which for CUDA is a second or two of the scan's total.
GPU_MIN_DOCS = 20_000


def _gpu_matmul(matrix, dense: np.ndarray) -> np.ndarray | None:
    """Sparse-dense multiply on an accelerator, or None to stay on the CPU.

    Returns None — rather than raising — for every reason not to: no GPU, no
    torch to address it with, a problem too small to be worth the transfer, or
    anything at all going wrong on the device. A coverage number that silently
    depends on which machine produced it would be a bug; a coverage number that
    took two seconds longer is not. So the CPU path is always the fallback and
    always produces the same answer.
    """
    from ..hardware import accelerator

    if matrix.shape[0] < GPU_MIN_DOCS:
        return None
    device = accelerator()
    if device == "cpu":
        return None
    try:
        import torch

        if device == "rocm":
            # ROCm builds of torch present themselves as CUDA.
            device = "cuda"
        if device == "cuda" and not torch.cuda.is_available():
            return None
        if device == "mps" and not torch.backends.mps.is_available():
            return None
        coo = matrix.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col]).astype(np.int64)
        ).to(device)
        values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
        sparse = torch.sparse_coo_tensor(
            indices, values, tuple(matrix.shape), device=device
        ).coalesce()
        rhs = torch.from_numpy(dense).to(device)
        return torch.sparse.mm(sparse, rhs).cpu().numpy()
    except Exception:
        return None


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


# A note for whoever profiles this next, so the same dead end is not walked
# twice. `StaticModel.from_pretrained` costs about 1.3 GB resident for
# potion-multilingual-128M, and the weights are 489 MB — a 500,353 x 256 float32
# table, of which the atlas uses the first 128 columns, because the potion
# models are Matryoshka and the atlas is a 128-dimensional space. Reading only
# those columns straight out of the safetensors file and building the
# StaticModel by hand does work and saves about 255 MB.
#
# It is not done, because it is not equivalent. The model config sets
# `normalize: true`, so `StaticModel.encode` L2-normalises its output — over
# whatever width the table has. Loading 128 columns normalises over 128; loading
# 256 and truncating afterwards, which is what `Embedder.encode` does, leaves a
# vector that is not unit length. The two differ by up to 0.057 per component on
# real text. The weighted path is unaffected (it reads the table directly and
# never calls `encode`), but the unweighted one is, and an embedding that
# depends on which loader ran is a coordinate system that is no longer
# comparable across scans. A quarter of a gigabyte is not worth that.
#
# The saving is real and still available: it needs the truncation and the
# normalisation to be the same operation in both paths first.


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
