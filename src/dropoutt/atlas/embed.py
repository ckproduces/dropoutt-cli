"""Embedding backend for the atlas: one encoder, quantised, 128 columns wide.

The atlas is a 128-dimensional space. The encoder it was built from —
``minishlab/potion-multilingual-128M`` — is a 500,353 x 256 float32 lookup
table, 489 MB on disk and about 1.3 GB resident once model2vec has copied it in.
The potion models are Matryoshka, so the first 128 columns *are* the
128-dimensional model: **half of that table was downloaded, held in memory, and
never read**, and the half that was read was carried at four bytes a weight for
a coordinate system whose cells are nowhere near that fine.

So the encoder is stored the way it is used. The first 128 columns, quantised to
one byte per weight with a per-row scale:

    row = codes[i].astype(float32) * scale[i]

61 MB of codes and 2 MB of scales against 489 MB, and measured on 9,000 real
multilingual documents through the shipped atlas, **99.74% of records land in
the same cell and 99.79% in the same subject area**. There is no second,
uncompressed build. Shipping both would mean two coordinate systems with one
name, and a coverage number whose meaning depended on which one a user happened
to have.

The conversion runs once, on the machine, from the published weights, and the
original is deleted afterwards. It is deterministic — a fixed slice, a fixed
scale, round-half-to-even — so every install derives byte-identical codes from
byte-identical weights, and ``encoder_weight_hash`` in the report stays a real
statement about comparability rather than a per-machine accident.

Two consequences worth naming:

**The normalisation width stopped being ambiguous.** model2vec's ``encode``
L2-normalises over whatever width the table has, so loading 256 columns and
truncating afterwards gave a different vector from loading 128 — by up to 0.057
per component, which is why an earlier attempt at this was reverted. A table
that is 128 wide has one answer.

**No torch, no ONNX runtime, and now no model2vec.** Pooling is SIF-weighted
when an IDF table is supplied (``w = a/(a+p)``), else a plain mean; L2 and the
anisotropy correction happen in :mod:`normalize`, not here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..compat import HAVE_TOKENIZERS
from .normalize import EMBED_DIM, SIF_A, sif_weights_from_probs, truncate

DEFAULT_MODEL = "minishlab/potion-multilingual-128M"

#: What the conversion downloads. Nothing else: ``snapshot_download`` is invoked
#: without ``allow_patterns`` upstream and therefore also pulls a duplicate ONNX
#: copy of the weights, roughly doubling an already large download.
SOURCE_FILES = ("config.json", "model.safetensors", "tokenizer.json")

#: Subdirectory holding the converted encoder.
QUANTIZED_DIR = "encoder-int8"
MANIFEST = "manifest.json"

#: Bumped when the conversion changes in a way that moves the vectors. An
#: encoder cached by an older build is reconverted rather than reinterpreted.
QUANT_FORMAT = 1

#: Documents encoded per pass in the weighted path. At 512 tokens each this is
#: about eight million token ids, so the transient arrays that
#: :meth:`Embedder.encode_tokenized` builds over them are tens of megabytes
#: rather than gigabytes. Large enough that the sparse multiply is still one
#: sizeable call rather than thousands of small ones.
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


class QuantizedTable:
    """A row-scaled int8 embedding table.

    ``rows`` is the only accessor the pooling path uses, and it dequantises just
    the rows asked for. A scan touches around a fifth of the vocabulary, so on a
    memory-mapped table the rest is never paged in at all.
    """

    def __init__(self, codes: np.ndarray, scale: np.ndarray) -> None:
        if codes.ndim != 2 or scale.ndim != 1 or codes.shape[0] != scale.shape[0]:
            raise ValueError("codes and scale disagree about the vocabulary")
        self.codes = codes
        # A row of exact zeros has no scale to speak of; using one keeps the
        # multiply below from producing NaN where it should produce zero.
        self.scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)

    @property
    def n_rows(self) -> int:
        return int(self.codes.shape[0])

    @property
    def width(self) -> int:
        return int(self.codes.shape[1])

    def rows(self, ids: np.ndarray, width: int | None = None) -> np.ndarray:
        stop = self.width if width is None else min(width, self.width)
        picked = np.asarray(self.codes[ids, :stop], dtype=np.float32)
        picked *= self.scale[ids][:, None]
        return picked

    @classmethod
    def from_float(cls, table: np.ndarray, width: int = EMBED_DIM) -> QuantizedTable:
        """Quantise a float table to one byte per weight, scaled per row.

        Per row rather than per table because the rows of a static embedding
        differ in magnitude by orders of magnitude — a frequent subword and a
        rare one are not on the same scale — and one global scale would spend
        the whole int8 range on the largest rows and quantise the rest to noise.
        """
        head = np.ascontiguousarray(table[:, :width], dtype=np.float32)
        peak = np.abs(head).max(axis=1)
        scale = (peak / 127.0).astype(np.float32)
        safe = np.where(scale > 0, scale, np.float32(1.0))
        codes = np.rint(head / safe[:, None]).clip(-127, 127).astype(np.int8)
        return cls(codes, scale)


class Embedder:
    """Wraps the quantised table with optional IDF-weighted pooling."""

    def __init__(
        self,
        table: QuantizedTable,
        tokenizer,
        name: str,
        *,
        token_log_prob: dict[int, float] | None = None,
        out_dim: int = EMBED_DIM,
        weight_hash: str = "",
        normalize: bool = True,
    ) -> None:
        self._table = table
        self._tokenizer = tokenizer
        self.name = name
        self.out_dim = min(int(out_dim), table.width)
        self._token_log_prob = token_log_prob or {}
        self._weight_hash = weight_hash
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return int(self.out_dim)

    @property
    def weight_hash(self) -> str:
        return self._weight_hash

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Rows in the embedding table, which is the token-id space."""
        return self._table.n_rows

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
        token ids — and then several more arrays of that same length for the
        probabilities, the SIF weights, the column indices and the sparse
        matrix. Together that is several gigabytes of transient, all of it to
        produce a result that is 200,000 x 128 floats: a hundred megabytes. It
        was the largest single allocation a scan made, and it was pure working
        set.

        Chunking changes nothing about the answer — pooling is per document, so
        document *i*'s vector does not depend on document *j* — and caps the
        transient at :data:`ENCODE_CHUNK` documents' worth.
        """
        use_weighted = bool(self._token_log_prob) if weighted is None else weighted
        if len(texts) <= ENCODE_CHUNK:
            return self._encode_chunk(texts, batch_size, use_weighted)
        out = np.zeros((len(texts), self.out_dim), dtype=np.float32)
        for start in range(0, len(texts), ENCODE_CHUNK):
            part = texts[start : start + ENCODE_CHUNK]
            out[start : start + len(part)] = self._encode_chunk(
                part, batch_size, use_weighted
            )
        return out

    def _encode_chunk(
        self, texts: list[str], batch_size: int, weighted: bool
    ) -> np.ndarray:
        tokens = self.tokenize(texts, batch_size=max(batch_size, 1024))
        pooled = self.pool(tokens, weighted=weighted)
        if weighted:
            return pooled
        # The unweighted path is what the model config calls for: a mean of the
        # token rows, L2-normalised if the model says so. It normalises over the
        # 128 columns that exist, which is the only width there is now.
        if self._normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            np.divide(pooled, norms, out=pooled, where=norms > 0)
        return truncate(pooled, self.out_dim)

    def tokenize(
        self,
        texts: list[str],
        *,
        batch_size: int = 8192,
        max_length: int = 512,
    ) -> TokenizedCorpus:
        """Batch-tokenize once into flat token IDs and document offsets.

        ``encode_batch_fast`` rather than ``encode_batch``: the only thing read
        from an encoding here is ``ids``, and the slower call exists to also
        track the character offset of every token, which is a fifth of the cost
        of tokenizing the atlas sample and is thrown away.
        """
        from itertools import chain

        tokenizer = self._tokenizer
        encode = getattr(tokenizer, "encode_batch_fast", None) or tokenizer.encode_batch
        chunks: list[np.ndarray] = []
        lengths = np.zeros(len(texts), dtype=np.int64)

        for start in range(0, len(texts), batch_size):
            encodings = encode(
                texts[start : start + batch_size], add_special_tokens=False
            )
            rows = [
                ids[:max_length] if len(ids) > max_length else ids
                for ids in (encoding.ids for encoding in encodings)
            ]
            lengths[start : start + len(rows)] = [len(ids) for ids in rows]
            total = int(sum(len(ids) for ids in rows))
            if total:
                chunks.append(
                    np.fromiter(chain.from_iterable(rows), dtype=np.int32, count=total)
                )

        token_ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int32)
        indptr = np.empty(len(texts) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        return TokenizedCorpus(
            token_ids=token_ids,
            indptr=indptr,
            n_docs=len(texts),
            max_length=max_length,
        )

    def pool(self, tokens: TokenizedCorpus, *, weighted: bool = True) -> np.ndarray:
        """Pool a token cache into one vector per document.

        The multiply is the only real arithmetic in a scan: two hundred thousand
        documents by up to 512 tokens by 128 dimensions is around thirteen
        billion multiply-accumulates. On a CPU that is seconds; on a GPU it is
        not, so :func:`_gpu_matmul` takes it when the machine has one and the
        problem is big enough to pay for moving the operands across the bus.
        Everything before this line — tokenizing, weighting — stays on the CPU,
        because it is memory shuffling rather than arithmetic.
        """
        from scipy.sparse import csr_matrix

        if tokens.n_tokens == 0:
            return np.zeros((tokens.n_docs, self.out_dim), dtype=np.float32)
        if int(tokens.token_ids.min()) < 0 or int(tokens.token_ids.max()) >= self._table.n_rows:
            raise ValueError("token cache contains ids outside the embedding vocabulary")

        # Subsetting is material: the vocabulary has half a million rows and an
        # atlas corpus normally observes around a fifth of them. The sparse
        # operand indexes this compact table rather than the whole vocabulary,
        # and on a memory-mapped file the untouched rows are never read at all.
        observed_ids, columns = np.unique(tokens.token_ids, return_inverse=True)

        row_lengths = np.diff(tokens.indptr)
        if weighted and self._token_log_prob:
            # Dense probabilities make all token-weight lookups vectorized.
            # Tokens omitted from the shipped top-frequency table retain the
            # previous conservative p=exp(-12) fallback.
            log_probs = np.full(self._table.n_rows, -12.0, dtype=np.float32)
            for token_id, log_prob in self._token_log_prob.items():
                if 0 <= token_id < self._table.n_rows:
                    log_probs[token_id] = log_prob
            probs = np.exp(log_probs[tokens.token_ids])
            weights = sif_weights_from_probs(probs, a=SIF_A)
        else:
            weights = np.ones(tokens.n_tokens, dtype=np.float32)

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
        dense = self._table.rows(observed_ids, self.out_dim)
        vectors = _gpu_matmul(matrix, dense)
        if vectors is None:
            vectors = matrix @ dense
        return np.asarray(vectors, dtype=np.float32)

    def encode_tokenized(self, tokens: TokenizedCorpus) -> np.ndarray:
        """SIF-pool a token cache with one sparse-dense matrix multiply."""
        return self.pool(tokens, weighted=True)

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

    def bind_idf(self, token_log_prob: dict[int, float]) -> Embedder:
        """Return a new wrapper sharing the table but using ``token_log_prob``."""
        return Embedder(
            self._table,
            self._tokenizer,
            self.name,
            token_log_prob=token_log_prob,
            out_dim=self.out_dim,
            weight_hash=self._weight_hash,
            normalize=self._normalize,
        )


#: Below this many documents the sparse multiply finishes on a CPU before a GPU
#: has finished being handed the operands. Moving a small matrix to a device and
#: back is dominated by the transfer and by the one-off cost of initialising the
#: runtime, which for CUDA is a second or two of the scan's total.
#:
#: **It must stay below** :data:`ENCODE_CHUNK`. It did not: this was 20,000 while
#: the encode path chunks at 16,384, so no batch could ever reach the threshold
#: and the device path shipped in 1.1 as unreachable code. There is a test that
#: keeps the two in that order. The path itself engages only when
#: ``DROPOUTT_DEVICE`` is set — see :func:`_gpu_matmul` for why.
GPU_MIN_DOCS = 4_096


def _gpu_matmul(matrix, dense: np.ndarray) -> np.ndarray | None:
    """Sparse-dense multiply on an accelerator, or None to stay on the CPU.

    Returns None — rather than raising — for every reason not to: no device
    named, no torch to address it with, a problem too small to be worth the
    transfer, or anything at all going wrong on the device. A coverage number
    that silently depended on which machine produced it would be a bug — and
    a device multiply accumulates in a different order than SciPy's, so its
    float32 result differs in the last ulps and a near-tie cell assignment can
    flip. That is why the device path runs only when ``DROPOUTT_DEVICE`` names
    a device: by default every machine measures the same numbers, and a user
    who opts in has said which two reports of theirs are comparable. The CPU
    path is always the fallback.
    """
    import os

    from ..hardware import accelerator

    if matrix.shape[0] < GPU_MIN_DOCS:
        return None
    if not (os.environ.get("DROPOUTT_DEVICE") or "").strip():
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


# --------------------------------------------------------------------------
# The stored encoder
# --------------------------------------------------------------------------


def local_model_dir(model_id: str, cache_root: Path) -> Path:
    return cache_root / "embedder" / model_id.rsplit("/", maxsplit=1)[-1]


def read_safetensors(path: Path, name: str) -> np.ndarray:
    """One tensor out of a safetensors file, without a safetensors dependency.

    The format is an eight-byte little-endian header length, a JSON header
    naming each tensor's dtype, shape and byte range, and then the raw buffer.
    Reading it here rather than adding a package for it keeps the conversion —
    which runs once, on a file this package already downloads — from putting
    another wheel on every install.
    """
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        entry = header[name]
        dtype = {"F32": np.float32, "F16": np.float16, "BF16": None}.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported tensor dtype {entry['dtype']!r}")
        start, stop = entry["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(stop - start)
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def convert(source: Path, target: Path, *, width: int = EMBED_DIM) -> None:
    """Quantise the published weights into the form this package uses.

    Deterministic by construction, which is the property that matters: every
    install derives the same codes from the same published file, so two scans on
    two machines are measured in the same coordinate system and
    ``encoder_weight_hash`` says so truthfully.
    """
    table = read_safetensors(source / "model.safetensors", "embeddings")
    quantized = QuantizedTable.from_float(table, width=width)
    normalize = True
    config = source / "config.json"
    if config.exists():
        try:
            normalize = bool(json.loads(config.read_text()).get("normalize", True))
        except Exception:
            normalize = True

    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "codes.npy", quantized.codes)
    np.save(target / "scale.npy", quantized.scale)
    (target / MANIFEST).write_text(
        json.dumps(
            {
                "format": QUANT_FORMAT,
                "scheme": "int8-row-scale",
                "width": quantized.width,
                "rows": quantized.n_rows,
                "normalize": normalize,
                "source_weight_hash": _file_hash(source / "model.safetensors"),
                "weight_hash": _file_hash(target / "codes.npy"),
            },
            indent=2,
        )
    )


def _load_quantized(directory: Path) -> tuple[QuantizedTable, dict] | None:
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("format", 0)) != QUANT_FORMAT:
            return None
        # Memory-mapped: a scan reads about a fifth of the vocabulary, and the
        # rest never becomes resident.
        codes = np.load(directory / "codes.npy", mmap_mode="r")
        scale = np.load(directory / "scale.npy")
    except Exception:
        return None
    return QuantizedTable(codes, scale), manifest


#: One embedder per process. Building it is a fixed cost the CLI starts in the
#: background before the scan needs it, which only works if the scan then finds
#: it here. The lock matters as much as the cache: without it the background
#: thread and the scan both start a load.
_LOADED: dict[tuple[str, bool, int], Embedder | None] = {}
_LOAD_LOCK = threading.Lock()


def load(
    model_id: str = DEFAULT_MODEL,
    *,
    cache_root: Path | None = None,
    offline: bool = False,
    out_dim: int = EMBED_DIM,
) -> Embedder | None:
    """Load the encoder, converting or downloading only if needed. None if unavailable."""
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
    if not HAVE_TOKENIZERS:
        return None
    if cache_root is None:
        from ..config import cache_dir

        cache_root = cache_dir()

    local = local_model_dir(model_id, cache_root)
    quantized_dir = local / QUANTIZED_DIR
    stored = _load_quantized(quantized_dir)

    if stored is None or not (local / "tokenizer.json").exists():
        if not _fetch_source(model_id, local, offline=offline):
            return None
        try:
            convert(local, quantized_dir)
        except Exception:
            # A weights file that cannot be converted is not going to convert
            # next time either — this is what a corrupt or truncated download
            # looks like. Removing it makes the next run re-fetch instead of
            # failing here forever with no path to repair but deleting the
            # cache by hand.
            with contextlib.suppress(OSError):
                (local / "model.safetensors").unlink()
            return None
        stored = _load_quantized(quantized_dir)
        if stored is None:
            return None

    table, manifest = stored
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(local / "tokenizer.json"))
    except Exception:
        # Same repair contract as the weights: an unreadable tokenizer.json is
        # removed so the next run re-downloads it, rather than being found on
        # disk, skipped by the fetch, and failing here on every scan.
        with contextlib.suppress(OSError):
            (local / "tokenizer.json").unlink()
        return None

    # The published weights are four times the size of what is kept and are not
    # read again. Leaving half a gigabyte in the cache to never be opened is
    # not a saving anyone asked for. Deleted only now, after the tokenizer
    # loaded: everything the encoder needs is known to work, so nothing can
    # still fail in a way whose repair would need this file re-downloaded.
    source = local / "model.safetensors"
    if source.exists():
        with contextlib.suppress(OSError):
            source.unlink()

    return Embedder(
        table,
        tokenizer,
        model_id,
        out_dim=out_dim,
        weight_hash=str(manifest.get("weight_hash", "")),
        normalize=bool(manifest.get("normalize", True)),
    )


def _fetch_source(model_id: str, local: Path, *, offline: bool) -> bool:
    """Download whatever of the published model is still missing.

    ``local_dir`` rather than the shared hub cache, for two reasons. The hub
    cache would keep a second 489 MB copy of the weights that the deletion in
    :func:`_load_uncached` never touches — the whole point of quantising is
    that the float32 table does not stay on disk. And ``local_dir`` downloads
    are written through a temp file and renamed, so an interrupted download
    leaves no half-written ``model.safetensors`` behind to be mistaken for the
    real one on the next run.
    """
    wanted = [name for name in SOURCE_FILES if not (local / name).exists()]
    if not wanted:
        return True
    if offline:
        return False
    try:
        from huggingface_hub import hf_hub_download

        local.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            hf_hub_download(model_id, name, local_dir=local)
    except Exception:
        return False
    return True


def _file_hash(path: Path | None) -> str:
    if path is not None and path.exists():
        h = hashlib.blake2b(digest_size=16)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return ""


def embedding_weight_hash(model_id: str = DEFAULT_MODEL) -> str:
    """Blake2b of the local quantised codes, or empty if not converted yet."""
    from ..config import cache_dir

    path = local_model_dir(model_id, cache_dir()) / QUANTIZED_DIR / "codes.npy"
    return _file_hash(path if path.exists() else None)
