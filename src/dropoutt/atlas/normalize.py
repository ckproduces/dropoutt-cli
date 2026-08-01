"""Frozen normalization constants for atlas embeddings.

Static embeddings are anisotropic — a shared component makes everything look
similar. Fit once on the Atlas corpus, ship as constants. The client applies
them; it never recomputes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: SIF smoothing; ``w = a / (a + p(word))``.
SIF_A = 1e-3
#: Truncate Matryoshka dims: full potion is 256, atlas geometry uses 128.
EMBED_DIM_FULL = 256
EMBED_DIM = 128


@dataclass
class NormConstants:
    """Mean vector, optional PCA directions, applied in that order then L2."""

    mean: np.ndarray                 # (dim,)
    pca_components: np.ndarray       # (k, dim) — may be empty (0, dim)
    dim: int = EMBED_DIM

    def apply(self, embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] > self.dim:
            x = x[:, : self.dim]
        elif x.shape[1] < self.dim:
            raise ValueError(
                f"embedding dim {x.shape[1]} is narrower than atlas dim {self.dim}"
            )
        x = x - self.mean.reshape(1, -1)
        if self.pca_components.size:
            # all-but-the-top: project out the leading principal components
            comps = self.pca_components
            x = x - (x @ comps.T) @ comps
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return (x / norms).astype(np.float32)


def fit_norm(
    embeddings: np.ndarray,
    *,
    pca_k: int = 2,
    dim: int = EMBED_DIM,
) -> NormConstants:
    """Fit mean and top PCA directions on already-truncated embeddings."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.shape[1] > dim:
        x = x[:, :dim]
    mean = x.mean(axis=0)
    centered = x - mean
    comps = np.zeros((0, dim), dtype=np.float32)
    if pca_k > 0 and len(x) > pca_k + 10:
        # Economy SVD — only need top-k right singular vectors.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        comps = vt[:pca_k].astype(np.float32)
    return NormConstants(mean=mean.astype(np.float32), pca_components=comps, dim=dim)


def sif_weights_from_probs(probs: np.ndarray, a: float = SIF_A) -> np.ndarray:
    """``w = a / (a + p)`` for a vector of unigram probabilities."""
    return (a / (a + np.maximum(probs, 1e-12))).astype(np.float32)


def truncate(embeddings: np.ndarray, dim: int = EMBED_DIM) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim == 1:
        return x[:dim].copy()
    return x[:, :dim].copy()
