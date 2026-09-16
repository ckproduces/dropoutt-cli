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
    l2_budget: int | None = None


# atlas-v3 is one product with two nested resolutions, not two products. L1 is a
# strict prefix of L2, so a coarse report is an exact union of fine cells and the
# two stay comparable — the property v1 had and the v2 split gave up.
#
# The cell count is chosen, not discovered. Measured on the v2 corpus, cosine
# silhouette across k is flat inside noise (0.118 at k=2, 0.124 at k=4, 0.106 at
# k=32) and stays near zero at k=256 under every whitening variant tried, so
# argmax-silhouette returns noise and reliably picks the smallest k — which is
# why 128 L1 regions produced only 296 cells. A frozen coordinate system picks a
# resolution the way latitude does; it does not ask the data where the lines are.
#
# 4096 is where three limits meet: reseed AMI falls from 0.75 at k=256 to 0.58 at
# k=4096 (boundaries get more arbitrary as k rises), the runtime payload reaches
# 17.3 MB unpacked, and the full corpus still leaves roughly 23,000 records per
# cell — 100x the calibration floor.
#
# The sample is 500,000 rather than the 200,000 the v2 products take, because
# the map it is placed on is fourteen times finer and the same sample would be
# spread that much thinner. A density ratio is a count divided by a count, so
# its relative error is 1/sqrt(records in the cell): 200,000 leaves 49 records
# in an average-sized cell of 4,096 and a +/-14% ratio, where 500,000 leaves 122
# and +/-9%. Real corpora concentrate, so the cells they actually occupy do
# better than that; the cells they barely touch are the ones this buys.
#
# The ceiling is memory, not time. Measured on this encoder, tokenising and
# encoding runs at 17.5k records/s, so 500,000 costs about half a minute. What
# it costs in memory is 5 KB of retained text per record in the parent plus
# each live shard's headroom copy, which is where
# :data:`dropoutt.hardware.MAX_SAMPLE_BUDGET` comes from.
ATLAS_V3 = AtlasProfile(
    version="atlas-v3",
    dim=128,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=500_000,
    pca_k=2,
    n_l1=256,
    l2_k_min=1,
    l2_k_max=64,
    l2_budget=4_096,
)

ATLAS_V2 = AtlasProfile(
    version="atlas-v2",
    dim=128,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=200_000,
    pca_k=2,
    n_l1=128,
    l2_k_min=1,
    l2_k_max=10,
)

ATLAS_V2_LITE = AtlasProfile(
    version="atlas-v2-lite",
    dim=64,
    pooling="sif",
    max_chars=2_000,
    max_tokens=512,
    default_sample=50_000,
    pca_k=2,
    n_l1=32,
    l2_k_min=1,
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

PROFILES = {
    profile.version: profile
    for profile in (ATLAS_V3, ATLAS_V2, ATLAS_V2_LITE, ATLAS_V1_LITE)
}
# atlas-v3 is the product: 4,096 cells over 256 regions on a 300 GiB corpus
# with per-record language detection. The v2 pair stays selectable for
# fingerprints that were placed on it.
DEFAULT_ATLAS_VERSION = ATLAS_V3.version

#: Products the interactive picker offers. v1 stays loadable when named.
PRODUCT_CHOICES = (
    (
        ATLAS_V3.version,
        f"{ATLAS_V3.version}  — detailed ({ATLAS_V3.l2_budget:,} cells, "
        f"{ATLAS_V3.n_l1} regions)",
    ),
    (ATLAS_V2.version, f"{ATLAS_V2.version}  — full ({ATLAS_V2.dim}-d)"),
    (ATLAS_V2_LITE.version, f"{ATLAS_V2_LITE.version}  — fast ({ATLAS_V2_LITE.dim}-d)"),
)


def get_profile(version: str | None = None) -> AtlasProfile:
    """Return a declared atlas profile or reject unknown coordinate systems."""
    key = DEFAULT_ATLAS_VERSION if version is None else version
    try:
        return PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown atlas {key!r}; expected one of: {choices}") from exc
