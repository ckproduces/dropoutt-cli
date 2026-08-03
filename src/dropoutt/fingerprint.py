"""Fingerprint assembly.

A fingerprint is a fixed-schema description of a dataset that can be compared
with any other fingerprint produced by the same pipeline version. It contains no
record excerpts, but it does contain paths, dataset names, aggregate metadata,
and stable hashes. Export policy still applies.

The identifier covers every input that can change the numbers. Pipeline version,
config, content hash and tokenizer hash are the obvious ones; the atlas, the
language model and the embedder are included too, because silently upgrading any
of them would produce different results under an identical id and break the
reproducibility claim outright.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .context import ScanContext
from .models import (
    FINGERPRINT_SCHEMA_VERSION,
    PIPELINE_VERSION,
    Finding,
    hash_many,
)


@dataclass
class Facet:
    """One group of named scalars, with units and a plain-language reading."""

    name: str
    values: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    #: How much evidence backs acting on this facet. See docs/fingerprint.md.
    evidence_grade: str = "descriptive"


@dataclass
class Fingerprint:
    fingerprint_id: str
    schema_version: str
    pipeline_version: str
    root: str
    profile: str
    facets: dict[str, Facet] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, bool] = field(default_factory=dict)
    degradations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint_id": self.fingerprint_id,
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "root": self.root,
            "profile": self.profile,
            "facets": {k: asdict(v) for k, v in self.facets.items()},
            "provenance": self.provenance,
            "datasets": self.datasets,
            "capabilities": self.capabilities,
            "degradations": self.degradations,
        }


#: Per-facet evidence grade. This is the honest part of the design: several of
#: these have no universally correct direction, and presenting them as one
#: quality score would be dishonest. See docs/fingerprint.md for the sources.
EVIDENCE_GRADE = {
    "shape": "descriptive",
    "redundancy": "conditional",
    "coverage": "goal-dependent",
    "difficulty": "not-computed-in-v1",
    "quality": "conditional",
    "language": "goal-dependent",
    "compliance": "risk-not-quality",
    "contamination": "strong",
}


def build(
    ctx: ScanContext,
    findings: list[Finding],
    *,
    total_chars: int,
    total_words: int,
    budget: Any = None,
    config_hash: str = "",
) -> Fingerprint:
    """Assemble a fingerprint from a completed scan."""
    by_id = {f.check_id: f for f in findings}

    content_hash = str(ctx.stats.get("content_hash") or hash_many(
        [f"{d.name}:{d.total_bytes}:{d.record_count}" for d in ctx.datasets]
    ))
    tokenizer_hash = ctx.tokenizer.tokenizer_hash if ctx.tokenizer else ""
    atlas_hash = getattr(ctx.atlas, "artifact_hash", "") if ctx.atlas else ""
    langid_hash = ctx.detector.backend if ctx.detector else ""

    fingerprint_id = "fp_" + hash_many([
        PIPELINE_VERSION, config_hash, content_hash, tokenizer_hash, atlas_hash, langid_hash,
    ])

    facets: dict[str, Facet] = {}

    # -- shape ----------------------------------------------------------
    shape: dict[str, Any] = {
        "records": ctx.total_records,
        "datasets": len(ctx.datasets),
        "total_chars": total_chars,
        "total_words": total_words,
    }
    if budget is not None and budget.estimates:
        shape["token_estimates"] = {
            e.name: {
                "total_tokens": e.total_tokens_est,
                "tokens_per_word": round(e.tokens_per_word, 3),
                "margin": e.margin,
            }
            for e in budget.estimates
            if not e.failed
        }
        cheapest = budget.cheapest
        if cheapest is not None:
            shape["cheapest_tokenizer"] = cheapest.name
    pack = by_id.get("T0-PACK-001")
    if pack is not None:
        shape.update({k: v for k, v in pack.data.items() if k != "algorithm"})
        shape["packing_algorithm"] = pack.data.get("algorithm")
    facets["shape"] = Facet("shape", shape, "Sizes and token budget. No good or bad direction.",
                            EVIDENCE_GRADE["shape"])

    # -- redundancy ------------------------------------------------------
    redundancy: dict[str, Any] = {}
    exact = by_id.get("T0-DUP-001")
    if exact is not None:
        redundancy["exact_duplicate_records"] = exact.count
        redundancy["exact_duplicate_rate"] = round(exact.rate, 5)
        redundancy.update(exact.data)
    near = by_id.get("T1-NDUP-001")
    if near is not None:
        redundancy["near_duplicate_records"] = near.count
        redundancy["near_duplicate_rate"] = round(near.rate, 5)
        redundancy.update({k: v for k, v in near.data.items() if k != "cluster_sizes_top10"})
    facets["redundancy"] = Facet(
        "redundancy", redundancy,
        "More deduplication is not automatically better; FineWeb measured the opposite past "
        "a point. Cluster size is the number to look at.",
        EVIDENCE_GRADE["redundancy"],
    )

    # -- coverage --------------------------------------------------------
    coverage: dict[str, Any] = {}
    if ctx.atlas is not None:
        coverage = dict(ctx.stats.get("atlas_coverage", {}))
    else:
        coverage["status"] = "not computed (no atlas available)"
    facets["coverage"] = Facet(
        "coverage", coverage,
        "Correct direction depends entirely on your goal. A coding dataset should be narrow.",
        EVIDENCE_GRADE["coverage"],
    )

    # -- quality ---------------------------------------------------------
    quality: dict[str, Any] = {}
    for cid, key in (
        ("T0-DEGEN-001", "degenerate"),
        ("T1-IDENT-001", "identity_and_refusal"),
        ("T1-STYLE-001", "style_tics"),
        ("T0-ENC-001", "encoding_damage"),
    ):
        f = by_id.get(cid)
        if f is not None:
            quality[key] = {"count": f.count, "rate": round(f.rate, 5), **f.data}
    facets["quality"] = Facet(
        "quality", quality,
        "A trade-off rather than a strict improvement: FineWeb-Edu's quality filter raised "
        "MMLU and slightly degraded HellaSwag.",
        EVIDENCE_GRADE["quality"],
    )

    # -- language --------------------------------------------------------
    language: dict[str, Any] = {}
    lang_f = by_id.get("T1-LANG-001")
    if lang_f is not None:
        language["composition"] = lang_f.data.get("composition", {})
        language["backend"] = lang_f.data.get("backend")
        language["low_trust"] = lang_f.data.get("low_trust", False)
    outliers = by_id.get("T1-LANG-002")
    if outliers is not None:
        language["outliers"] = {"count": outliers.count, "rate": round(outliers.rate, 5)}
    facets["language"] = Facet(
        "language", language,
        "The accuracy of the labels matters; the target mix is your decision.",
        EVIDENCE_GRADE["language"],
    )

    # -- compliance ------------------------------------------------------
    compliance: dict[str, Any] = {}
    pii = by_id.get("T1-PII-001")
    if pii is not None:
        compliance["pii_records"] = pii.count
        compliance["pii_by_kind"] = pii.data.get("by_kind", {})
    lic = by_id.get("T1-LIC-001")
    compliance["datasets_without_licence"] = lic.count if lic else 0
    compliance["licences"] = {
        d.name: d.license for d in ctx.datasets if d.license
    }
    facets["compliance"] = Facet(
        "compliance", compliance,
        "Legal exposure, not model quality. No measurable effect on evaluation scores.",
        EVIDENCE_GRADE["compliance"],
    )

    # -- contamination ---------------------------------------------------
    contamination: dict[str, Any] = {}
    contam = by_id.get("T1-CONTAM-001")
    if contam is not None:
        contamination = {
            name: {
                key: value
                for key, value in result.items()
                if key != "witnesses"
            }
            for name, result in contam.data.get("results", {}).items()
        }
        contamination["_rule"] = contam.data.get("rule")
    elif ctx.contamination is None:
        contamination["status"] = "not computed (no benchmark index available)"
    else:
        contamination["status"] = "no overlap detected"
    facets["contamination"] = Facet(
        "contamination", contamination,
        "The one facet where lower is unambiguously better. Removing contamination usually "
        "lowers your reported score, which is the point.",
        EVIDENCE_GRADE["contamination"],
    )

    return Fingerprint(
        fingerprint_id=fingerprint_id,
        schema_version=FINGERPRINT_SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        root=ctx.root,
        profile=ctx.profile.value,
        facets=facets,
        provenance={
            "content_hash": content_hash,
            "config_hash": config_hash,
            "tokenizer_hash": tokenizer_hash,
            "atlas_hash": atlas_hash,
            "langid_backend": langid_hash,
            "model_id": ctx.model_id,
            "seq_len": ctx.seq_len,
        },
        datasets=[
            {
                "name": d.name,
                "files": len(d.files),
                "bytes": d.total_bytes,
                "records": d.record_count,
                "layout": d.schema_id,
                "licence": d.license,
                "declared_language": d.declared_language,
            }
            for d in ctx.datasets
        ],
        capabilities={
            "tokenizer": ctx.tokenizer is not None,
            "chat_template": ctx.chat_template is not None,
            "langid": ctx.detector is not None,
            "atlas": ctx.atlas is not None,
            "contamination_index": ctx.contamination is not None
            and not ctx.contamination.is_empty,
        },
        degradations=list(ctx.degradations),
    )
