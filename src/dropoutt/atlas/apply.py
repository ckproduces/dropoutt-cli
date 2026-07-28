"""Loading a frozen atlas and assigning records to regions.

The atlas is a coordinate system, not a quality reference. Nothing in this module
knows whether a region is good; it only knows which bin a record lands in, so
that two datasets scanned on different machines land in the *same* bins and can
be compared.

Two honesty rules are enforced here rather than left to callers.

Records too far from every centroid are **off-atlas**, and off-atlas rate is
reported per language as well as globally. For a language the embedding model
represents poorly, topical assignment is unreliable no matter how good the
clustering is, and averaging that away would hide it.

Above the off-atlas threshold, coverage numbers are **suppressed rather than
displayed**, because a coverage histogram over records that did not really land
anywhere is worse than no histogram.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Above this share of off-atlas records, coverage is suppressed for that group.
OFF_ATLAS_SUPPRESS = 0.10


@dataclass
class Atlas:
    """A frozen coordinate system."""

    centroids: np.ndarray            # (n_regions, dim), L2-normalised
    region_category: np.ndarray      # (n_regions,) taxonomy id per region
    coords: np.ndarray               # (n_regions, 2) for display
    probe_coef: np.ndarray
    probe_intercept: np.ndarray
    probe_classes: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    artifact_hash: str = ""

    @property
    def n_regions(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def embed_model(self) -> str:
        return str(self.meta.get("embed_model", "unknown"))

    @property
    def off_threshold(self) -> float:
        return float(self.meta.get("off_atlas_threshold", 0.35))

    @property
    def region_terms(self) -> list[str]:
        return list(self.meta.get("region_terms", []))

    @classmethod
    def load(cls, path: str | Path) -> "Atlas":
        import hashlib  # noqa: PLC0415

        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.blake2b(raw, digest_size=8).hexdigest()
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"][0]))
        return cls(
            centroids=data["centroids"],
            region_category=data["region_category"],
            coords=data["coords"],
            probe_coef=data["probe_coef"],
            probe_intercept=data["probe_intercept"],
            probe_classes=data["probe_classes"],
            meta=meta,
            artifact_hash=digest,
        )

    # -- assignment -------------------------------------------------------

    def assign(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (region index, cosine similarity) for each row.

        A region index of -1 means off-atlas.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        sims = emb @ self.centroids.T
        best = sims.argmax(axis=1)
        score = sims.max(axis=1)
        best = np.where(score >= self.off_threshold, best, -1)
        return best, score

    def categorize(self, embeddings: np.ndarray) -> np.ndarray:
        """Level-0 taxonomy category per row, from the supervised probe."""
        emb = np.asarray(embeddings, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        logits = emb @ self.probe_coef.T + self.probe_intercept
        return self.probe_classes[logits.argmax(axis=1)]

    # -- coverage ---------------------------------------------------------

    def coverage(
        self,
        regions: np.ndarray,
        categories: np.ndarray,
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the coverage report, suppressing it where it would mislead."""
        total = int(len(regions))
        if total == 0:
            return {"status": "no records"}

        off = int((regions < 0).sum())
        off_rate = off / total

        result: dict[str, Any] = {
            "records": total,
            "off_atlas": off,
            "off_atlas_rate": round(off_rate, 4),
            "atlas_version": self.meta.get("version"),
            "embed_model": self.embed_model,
            "l0_holdout_accuracy": self.meta.get("l0_holdout_accuracy"),
            "region_purity_by_taxonomy": self.meta.get("region_purity_by_taxonomy"),
        }

        if languages:
            per_lang: dict[str, dict[str, float]] = {}
            for lang in sorted(set(languages)):
                idx = [i for i, x in enumerate(languages) if x == lang]
                if len(idx) < 20:
                    continue
                lang_off = sum(1 for i in idx if regions[i] < 0) / len(idx)
                per_lang[lang] = {
                    "records": len(idx),
                    "off_atlas_rate": round(lang_off, 4),
                    "reliable": lang_off <= OFF_ATLAS_SUPPRESS,
                }
            result["per_language"] = per_lang

        if off_rate > OFF_ATLAS_SUPPRESS:
            result["status"] = "suppressed"
            result["reason"] = (
                f"{off_rate:.0%} of records are off-atlas, above the "
                f"{OFF_ATLAS_SUPPRESS:.0%} threshold. This atlas does not fit this corpus, "
                f"so coverage numbers would be misleading and have been withheld."
            )
            return result

        on = regions[regions >= 0]
        region_counts = np.bincount(on, minlength=self.n_regions)
        cat_counts: dict[str, int] = {}
        for c in categories:
            key = str(int(c))
            cat_counts[key] = cat_counts.get(key, 0) + 1

        nonzero = int((region_counts > 0).sum())
        mass = region_counts / max(region_counts.sum(), 1)
        entropy = float(-(mass[mass > 0] * np.log(mass[mass > 0])).sum())

        result.update({
            "status": "ok",
            "regions_occupied": nonzero,
            "regions_total": self.n_regions,
            "region_entropy": round(entropy, 4),
            "max_region_entropy": round(float(np.log(self.n_regions)), 4),
            "by_category": cat_counts,
            # The full histogram, sparse. `top_regions` below is a display
            # convenience capped at twelve, and comparing two fingerprints on
            # that head alone silently computes shares over a fraction of each
            # corpus. Only occupied regions are stored, so this costs a few
            # hundred small integers and makes `dropoutt diff` exact.
            "region_counts": {
                str(int(r)): int(region_counts[r])
                for r in np.nonzero(region_counts)[0]
            },
            "top_regions": [
                {"region": int(r), "records": int(region_counts[r]),
                 "terms": self.region_terms[r] if r < len(self.region_terms) else ""}
                for r in np.argsort(-region_counts)[:12]
                if region_counts[r] > 0
            ],
        })
        return result


def bundled_atlas_path() -> Path | None:
    from importlib import resources  # noqa: PLC0415

    try:
        ref = resources.files("dropoutt.data") / "atlas" / "atlas-lite-v0.npz"
        path = Path(str(ref))
        return path if path.exists() else None
    except Exception:
        return None


def load_bundled() -> Atlas | None:
    path = bundled_atlas_path()
    if path is None:
        return None
    try:
        return Atlas.load(path)
    except Exception:
        return None
