"""Immutable runtime declarations for the Atlas products."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AtlasProfile:
    """Parameters that define one frozen atlas coordinate system."""

    version: str
    dim: int
    pooling: str
    max_chars: int
    max_tokens: int
    default_sample: int
    pca_k: int
    n_l1: int
    l2_k_min: int
    l2_k_max: int


ATLAS_V2 = AtlasProfile(
    version="atlas-v2",
    dim=256,
    pooling="sif",
    max_chars=4_000,
    max_tokens=1_024,
    default_sample=200_000,
    pca_k=2,
    n_l1=256,
    l2_k_min=4,
    l2_k_max=10,
)

ATLAS_V2_LITE = AtlasProfile(
    version="atlas-v2-lite",
    dim=16,
    pooling="mean",
    max_chars=1_024,
    max_tokens=256,
    default_sample=50_000,
    pca_k=1,
    n_l1=16,
    l2_k_min=4,
    l2_k_max=10,
)

# v1 has no profile metadata in its artifact. It remains loadable for existing
# fingerprints, but is never selected unless named explicitly.
ATLAS_V1_LITE = AtlasProfile(
    version="atlas-v1-lite",
    dim=128,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=200_000,
    pca_k=2,
    n_l1=48,
    l2_k_min=0,
    l2_k_max=0,
)

PROFILES = {profile.version: profile for profile in (ATLAS_V2, ATLAS_V2_LITE, ATLAS_V1_LITE)}
DEFAULT_ATLAS_VERSION = ATLAS_V2_LITE.version


def get_profile(version: str | None = None) -> AtlasProfile:
    """Return a declared atlas profile or reject unknown coordinate systems."""
    key = DEFAULT_ATLAS_VERSION if version is None else version
    try:
        return PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown atlas {key!r}; expected one of: {choices}") from exc
