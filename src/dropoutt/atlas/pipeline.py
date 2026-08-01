"""Shared atlas pipeline helpers and version hash.

Client and build import the same extraction, chunking, embedding, and
normalization. ``pipeline_hash`` seals the versioned constants that define
comparability; every stored result must carry it next to ``atlas_version``.
"""

from __future__ import annotations

import hashlib
import json

from .chunk import CHUNKER_VERSION, DEFAULT_MAX_WORDS, DEFAULT_TARGET_WORDS
from .embed import DEFAULT_MODEL
from .normalize import EMBED_DIM, SIF_A

PIPELINE_VERSION = "atlas-pipeline-v1"

#: Declared steps. Changing any value changes the hash.
PIPELINE_DECLARATION = {
    "pipeline_version": PIPELINE_VERSION,
    "embed_model": DEFAULT_MODEL,
    "embed_dim": EMBED_DIM,
    "pooling": "sif",
    "sif_a": SIF_A,
    "chunker": CHUNKER_VERSION,
    "chunk_target_words": DEFAULT_TARGET_WORDS,
    "chunk_max_words": DEFAULT_MAX_WORDS,
    "normalization": ["mean_removal", "all_but_the_top", "l2"],
    "pca_k": 2,
    "assignment": "soft_topk",
    "soft_k": 5,
    "extraction": "format-aware-v1",
}


def pipeline_hash(extra: dict | None = None) -> str:
    payload = dict(PIPELINE_DECLARATION)
    if extra:
        payload.update(extra)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=16).hexdigest()
