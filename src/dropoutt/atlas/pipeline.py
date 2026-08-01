"""Shared atlas pipeline helpers and version hash.

Client and build import the same extraction, chunking, embedding, and
normalization. ``pipeline_hash`` seals the versioned constants that define
comparability; every stored result must carry it next to ``atlas_version``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from .chunk import CHUNKER_VERSION, DEFAULT_MAX_WORDS, DEFAULT_TARGET_WORDS
from .embed import DEFAULT_MODEL
from .normalize import EMBED_DIM, SIF_A

PIPELINE_VERSION = "atlas-pipeline-v2"

#: Declared steps. Changing any value changes the hash.
PIPELINE_DECLARATION = {
    "pipeline_version": PIPELINE_VERSION,
    "embed_model": DEFAULT_MODEL,
    "embed_dim": EMBED_DIM,
    "pooling": "sif",
    "pooling_implementation": "batch-tokenize-csr-matmul",
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


def population_crosswalk(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    n_current: int,
    n_previous: int,
    previous_version: str,
) -> dict:
    """Map cells by Jaccard over a shared reference population."""

    pairs = np.bincount(
        current.astype(np.int64) * n_previous + previous.astype(np.int64),
        minlength=n_current * n_previous,
    ).reshape(n_current, n_previous)
    current_size = np.bincount(current, minlength=n_current)
    previous_size = np.bincount(previous, minlength=n_previous)
    best_previous = pairs.argmax(axis=1)
    best_current = pairs.argmax(axis=0)
    previous_targets = Counter(best_previous.tolist())
    current_targets = Counter(best_current.tolist())
    cells: list[dict] = []
    unchanged = split = merged = new = 0
    for cell, old in enumerate(best_previous.tolist()):
        overlap = int(pairs[cell, old])
        union = int(current_size[cell] + previous_size[old] - overlap)
        jaccard = overlap / max(union, 1)
        if best_current[old] == cell and jaccard >= 0.8:
            relation = "unchanged"
            unchanged += 1
        elif previous_targets[old] > 1 and jaccard >= 0.1:
            relation = "split"
            split += 1
        elif current_targets[cell] > 1 and jaccard >= 0.1:
            relation = "merged"
            merged += 1
        else:
            relation = "new"
            new += 1
        cells.append({
            "cell_id": cell,
            "previous_cell_id": old,
            "population_jaccard": round(jaccard, 5),
            "relationship": relation,
        })
    retired = [
        old
        for old in range(n_previous)
        if int(pairs[:, old].max()) == 0
    ]
    return {
        "previous_version": previous_version,
        "method": "jaccard_over_shared_reference_record_ids",
        "cells": cells,
        "summary": {
            "unchanged": unchanged,
            "split": split,
            "merged": merged,
            "new": new,
            "retired": len(retired),
        },
        "retired_previous_cells": retired,
    }
