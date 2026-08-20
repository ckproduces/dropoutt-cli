"""One complete reading of a scan, in plain data, for every output format.

:mod:`dropoutt.report.summary` decides *what a scan says*. This module decides
*what a report contains*: it takes that reading, plus the fingerprint, the token
budget and the handful of things only the raw result knows, and flattens all of
it into one JSON-serialisable dictionary.

The reason it exists is that the four output formats had drifted apart. The HTML
page carried the dataset table, the density grid, the places lists, the
imbalances, the off-map diagnosis, the degradations and the provenance block.
The Markdown file carried about a third of that. The terminal carried a tenth.
And there was no JSON report at all — only ``findings.jsonl``, which is the
findings and nothing else, and ``fingerprint.json``, which is deliberately
free of anything quotable.

That is a bad shape for a tool whose users mostly are not looking at a browser.
A scan runs in CI and the reviewer sees a pull-request comment; a scan runs in a
batch job and the operator sees a log; a scan runs in a pipeline and the next
stage sees a file. Each of those readers was being handed a strictly worse
report than the one person with a browser, and none of them was told what they
were missing.

So: **one payload, four renderings**. Everything here appears in every format
that can express it. HTML draws the density grid as coloured squares and
Markdown prints the same ratios as a table; the terminal prints the ten largest
rows and says how many it left out. What none of them do any more is *omit* a
section the others have.

Two rules carry over from the page and are enforced here rather than in each
renderer. Everything quoted from the corpus passes through
:func:`dropoutt.report.escaping.safe_snippet`, and ``--no-evidence`` removes
excerpts and source locations everywhere at once, because this is the only place
they are assembled.
"""

from __future__ import annotations

from typing import Any

from .atlas_story import density_ratio, format_reach
from .escaping import safe_snippet
from .summary import ScanSummary, budget_method, budget_rows


def skipped_checks(result) -> list:
    """Checks that could not run, one per distinct reason.

    Deduplicated on the reason rather than the check: fifteen checks blocked by
    "needs a tokenizer" is one thing to fix, and printing it fifteen times
    teaches the reader to skip the section.
    """
    seen: set[str] = set()
    rows = []
    for entry in result.skipped:
        if entry.reason in seen:
            continue
        seen.add(entry.reason)
        rows.append(entry)
    return rows


def overlap_matrix(result) -> list[dict]:
    """Directed dataset-overlap rows, largest first."""
    for finding in result.findings:
        if finding.check_id == "T1-OVERLAP-001":
            return list(finding.data.get("matrix", [])[:20])
    return []


def atlas_identity(result) -> dict[str, Any] | None:
    """The coordinate system this report's coverage numbers were measured in.

    None when the scan placed nothing, so a report says nothing rather than
    naming an atlas that produced no numbers. A coverage number is only
    comparable against another measured in the same atlas, with the same
    encoder weights and the same pipeline, so all three are carried.
    """
    cov = (result.ctx.stats or {}).get("atlas_coverage") or {}
    if not cov.get("atlas_version"):
        return None
    return {
        "version": cov.get("atlas_version", ""),
        "pipeline_hash": cov.get("pipeline_hash", "") or "",
        "encoder_weight_hash": cov.get("encoder_weight_hash", "") or "",
        "embed_model": cov.get("embed_model", "") or "",
        "normalization_variant": cov.get("normalization_variant") or "",
        "n_l1": cov.get("n_l1"),
        "n_regions": cov.get("regions_total"),
    }


def off_map_examples(result, summary: ScanSummary, *, include_evidence: bool) -> list[dict]:
    """The records furthest from anything on the map, safe to print."""
    if not include_evidence or summary.atlas is None:
        return []
    return [
        {
            "score": round(float(ex.get("score", 0.0)), 4),
            "chars": ex.get("chars"),
            "dataset": ex.get("dataset", ""),
            "language": ex.get("language", ""),
            "excerpt": safe_snippet(str(ex.get("excerpt", "")).replace("\n", " "), 160),
        }
        for ex in summary.atlas.off_examples[:4]
    ]


def token_note(result, summary: ScanSummary) -> str:
    """How the headline token number was arrived at, in three words.

    "Estimated from a sample" with no interval on it tells a reader the number
    is soft and gives them nothing to do about it, so the caption appears only
    when there is an interval to print.
    """
    if summary.tokens and result.ctx.tokenizer is not None:
        return "counted exactly"
    if summary.tokens and summary.token_margin:
        return f"estimated, ±{summary.token_margin * 100:.2f}%"
    return ""


def build(
    result,
    fp,
    budget: Any = None,
    *,
    include_evidence: bool = True,
    summary: ScanSummary | None = None,
) -> dict[str, Any]:
    """Everything a report can say about this scan, as plain data."""
    from .summary import build as build_summary

    s = summary or build_summary(result, budget=budget, include_evidence=include_evidence)
    ctx = result.ctx
    has_tokenizer = ctx.tokenizer is not None

    payload: dict[str, Any] = {
        "schema": "dropoutt.report/1",
        "pipeline_version": fp.pipeline_version,
        "fingerprint_id": fp.fingerprint_id,
        "root": ctx.root,
        "profile": s.profile,
        "elapsed_seconds": round(s.elapsed, 3),
        "shards": int(ctx.stats.get("shards", 1) or 1),
        "includes_evidence": include_evidence,
        "blocking_enabled": s.blocking_enabled,
        "verdict": {
            "headline": s.verdict,
            "tone": s.tone,
            "detail": s.subtitle,
        },
        "composition": _composition(result, s, fp),
        "problems": [_problem(p, include_evidence=include_evidence) for p in s.problems],
        "notes": [_problem(p, include_evidence=include_evidence) for p in s.notes],
        "token_budget": _budget(budget, s, has_tokenizer=has_tokenizer),
        "atlas": _atlas(result, s, include_evidence=include_evidence),
        "not_checked": [
            {
                "check_id": entry.check_id,
                "title": entry.title,
                "reason": entry.reason,
                "unlock": entry.unlock,
            }
            for entry in skipped_checks(result)
        ],
        "degraded": list(ctx.degradations),
        "provenance": dict(fp.provenance),
        "capabilities": dict(fp.capabilities),
    }
    return payload


# --------------------------------------------------------------------------


def _composition(result, s: ScanSummary, fp) -> dict[str, Any]:
    comp = s.composition
    return {
        "records": s.records,
        "datasets": s.datasets,
        "files": s.files,
        "total_characters": comp.total_chars,
        "mean_characters_per_record": comp.mean_chars,
        "tokens": s.tokens,
        "token_note": token_note(result, s),
        "languages": [
            {"code": code, "share": round(share, 6)} for code, share in s.languages
        ],
        "language_line": s.language_line,
        "layouts": [
            {"label": label, "share": round(share, 6), "confidence": round(conf, 4)}
            for label, share, conf in comp.layouts
        ],
        "structured_share": round(comp.structured_share, 6),
        "structured_line": comp.structured_line,
        "unparseable_share": round(comp.unparseable_share, 6),
        "chat_templates_in_text": [
            {"family": name, "records": count, "share": round(share, 6)}
            for name, count, share in comp.templates
        ],
        "target_template": comp.template_target,
        "datasets_without_licence": comp.unlicensed,
        "dataset_table": [dict(row) for row in fp.datasets],
        "dataset_overlap": overlap_matrix(result),
    }


def _problem(problem, *, include_evidence: bool) -> dict[str, Any]:
    return {
        "check_id": problem.check_id,
        "title": problem.title,
        "severity": problem.severity.value,
        "consequence": (
            "will block" if problem.is_blocking
            else "would block" if problem.would_block
            else problem.severity.value
        ),
        "would_block_under": list(problem.would_block),
        "affected": problem.affected,
        "considered": problem.considered,
        "share": round(problem.share, 6),
        "unit": problem.unit,
        "total_unit": problem.total_unit,
        "scale": problem.scale,
        "wasted_tokens": problem.tokens,
        "detail": problem.detail,
        "fix": problem.fix,
        "by_dataset": dict(problem.by_dataset),
        "evidence": [
            {
                "source_file": ev.source_file,
                "source_index": ev.source_index,
                "excerpt": safe_snippet(str(ev.excerpt).replace("\n", " "), 240),
                "partner_excerpt": (
                    safe_snippet(str(ev.partner_excerpt).replace("\n", " "), 240)
                    if ev.partner_excerpt else ""
                ),
                "score": ev.score,
            }
            for ev in problem.evidence
        ] if include_evidence else [],
    }


def _budget(budget, s: ScanSummary, *, has_tokenizer: bool) -> dict[str, Any]:
    rows = budget_rows(budget)
    return {
        "headline": s.tokens_line,
        "total": s.tokens,
        "margin_share": round(s.token_margin, 6),
        "method": budget_method(budget, exact=has_tokenizer),
        "exact": has_tokenizer,
        "notes": list(getattr(budget, "notes", []) or []),
        "sample_size": getattr(budget, "sample_size", 0),
        "tokenizers": [
            {
                "name": row["name"],
                "total_tokens": row["total"],
                "margin_share": round(row["margin_share"], 6),
                "premium_vs_cheapest": round(row["premium"], 6),
                "cheapest": index == 0,
            }
            for index, row in enumerate(rows)
        ],
    }


def _atlas(result, s: ScanSummary, *, include_evidence: bool) -> dict[str, Any] | None:
    atlas = s.atlas
    if atlas is None:
        return None
    if not atlas.available:
        return {"available": False, "reason": atlas.unavailable_reason,
                "identity": atlas_identity(result)}
    return {
        "available": True,
        "identity": atlas_identity(result),
        "sampled_records": atlas.sampled,
        "placed_records": atlas.placed,
        "too_short_to_place": atlas.too_short,
        "subregions_total": atlas.regions_total,
        "subregions_touched": atlas.regions_touched,
        "effective_reach": round(atlas.effective, 4),
        "effective_reach_label": format_reach(atlas.effective),
        "shape": atlas.shape,
        "shape_detail": atlas.shape_line,
        "concentration": atlas.concentration,
        "off_map_records": atlas.off_count,
        "off_map_rate": round(atlas.off_rate, 6),
        "off_map_line": atlas.off_line,
        "off_map_detail": atlas.off_detail,
        "off_map_examples": off_map_examples(result, s, include_evidence=include_evidence),
        "grid_peak_density": round(atlas.grid_peak, 4),
        "subject_areas": [
            {
                "name": area.name,
                "records": area.records,
                "share": round(area.share, 6),
                "density": round(area.ratio, 4),
                "density_label": density_ratio(area.ratio),
                "reach": round(area.effective_reach, 4),
                "reach_label": format_reach(area.effective_reach),
                "subregions": len(area.cells),
                "subregions_unreached": area.unreached,
                "fully_reached": area.fully_reached,
                "cells": [
                    {
                        "region": cell.region,
                        "records": cell.records,
                        "density": round(cell.ratio, 4),
                        "caption": cell.caption,
                    }
                    for cell in area.cells
                ],
            }
            for area in atlas.grid
        ],
        "insights": [
            {
                "kind": insight.kind,
                "label": insight.label,
                "headline": insight.headline,
                "detail": insight.detail,
                "tone": insight.tone,
                "evidence": safe_snippet(insight.evidence, 240) if include_evidence else "",
            }
            for insight in atlas.insights
        ],
        "most_of": [_place(p, include_evidence=include_evidence) for p in atlas.places],
        "least_of": [
            _place(p, include_evidence=include_evidence) for p in atlas.thin_places
        ],
        "imbalances": [
            {
                "region": item.region,
                "area": item.area,
                "density": round(item.ratio, 4),
                "density_label": density_ratio(item.ratio),
                "records": item.records,
                "share": round(item.share, 6),
                "action": item.action,
                "yours": safe_snippet(item.yours, 240) if include_evidence else "",
            }
            for item in atlas.imbalances
        ],
    }


def _place(place, *, include_evidence: bool) -> dict[str, Any]:
    return {
        "region": place.region,
        "area": place.area,
        "records": place.records,
        "share": round(place.share, 6),
        "density": round(place.ratio, 4),
        "density_label": density_ratio(place.ratio),
        "cohesion": place.cohesion,
        "repetitive": place.repetitive,
        "caption": place.caption,
        "yours": safe_snippet(place.yours, 240) if include_evidence else "",
    }
