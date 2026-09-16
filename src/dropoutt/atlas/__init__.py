"""The frozen atlas: a shared coordinate system for comparing datasets.

Not a collection of good datasets. A coordinate system, like latitude and
longitude, containing no notion of quality. See docs/atlas.md.
"""

from __future__ import annotations

from .apply import (
    OFF_ATLAS_HIGH,
    OFF_ATLAS_NOTABLE,
    Atlas,
    bundled_atlas_path,
    load_bundled,
)
from .embed import DEFAULT_MODEL, Embedder
from .embed import load as load_embedder
from .normalize import EMBED_DIM, NormConstants
from .pipeline import pipeline_hash, profile_declaration
from .profiles import (
    ATLAS_V2,
    ATLAS_V2_LITE,
    ATLAS_V3,
    DEFAULT_ATLAS_VERSION,
    AtlasProfile,
    get_profile,
)

__all__ = [
    "ATLAS_V2",
    "ATLAS_V2_LITE",
    "ATLAS_V3",
    "DEFAULT_ATLAS_VERSION",
    "DEFAULT_MODEL",
    "EMBED_DIM",
    "OFF_ATLAS_HIGH",
    "OFF_ATLAS_NOTABLE",
    "Atlas",
    "AtlasProfile",
    "Embedder",
    "NormConstants",
    "bundled_atlas_path",
    "get_profile",
    "load_bundled",
    "load_embedder",
    "pipeline_hash",
    "profile_declaration",
]
