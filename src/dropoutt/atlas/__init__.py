"""The frozen atlas: a shared coordinate system for comparing datasets.

Not a collection of good datasets. A coordinate system, like latitude and
longitude, containing no notion of quality. See docs/atlas.md.
"""

from __future__ import annotations

from .apply import OFF_ATLAS_SUPPRESS, Atlas, bundled_atlas_path, load_bundled  # noqa: F401
from .embed import DEFAULT_MODEL, Embedder  # noqa: F401
from .embed import load as load_embedder  # noqa: F401

__all__ = [
    "Atlas", "load_bundled", "bundled_atlas_path", "OFF_ATLAS_SUPPRESS",
    "Embedder", "load_embedder", "DEFAULT_MODEL",
]
