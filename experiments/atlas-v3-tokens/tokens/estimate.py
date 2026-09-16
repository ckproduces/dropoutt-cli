"""Stage 3: extrapolate the sample to the whole corpus, with uncertainty.

Population: the 163,452,464 records the map was fitted on, 212,786,662,386
UTF-8 bytes of text (both exact, from the build's checkpoint and per-record
source ids). Per source, the retained byte count is the cache's byte count
scaled by the share of its rows the build retained after dedup; those
per-source bytes are then calibrated within each axis so they sum exactly to
the axis totals the build recorded.

Estimator: a separate ratio estimator per source. r_i = (sum of tokens in the
source's sample) / (sum of its bytes); tokens_i = B_i * r_i. The variance of a
ratio estimator is var(t - r b) / (n * mean(b)^2); the total's variance is the
sum of B_i^2 var(r_i) over sources, and the 95% interval is +/- 1.96 standard
errors. Sampling within a source is uniform (reservoir), so this is the
standard stratified ratio estimate with sources as strata.

Cross-check: the build's own 500,000-record reservoir is a uniform sample of
the retained population but stores only 600-character heads; its tokens per
byte, applied to the exact byte totals, gives a second estimate whose gap to
the first measures how much heads differ from whole records.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .common import RESULTS, SCRATCH, manifest, population
from .panel import PANEL

COUNTS = SCRATCH / "counts"


def source_bytes(pop, man) -> dict[str, float]:
    """Retained UTF-8 bytes per source, calibrated to the exact axis totals."""
    raw = {}
    for slug, v in pop["sources"].items():
        m = man[slug]
        raw[slug] = m["logical_bytes"] * v["records"] / max(m["rows"], 1)
    by_axis = defaultdict(float)
    for slug, b in raw.items():
        by_axis[pop["sources"][slug]["axis"]] += b
    cal = {}
    for slug, b in raw.items():
        axis = pop["sources"][slug]["axis"]
        cal[slug] = b * pop["axes"][axis]["logical_bytes"] / by_axis[axis]
    return cal, {a: by_axis[a] / pop["axes"][a]["logical_bytes"] for a in by_axis}


def main():
    pop, man = population(), manifest()
    B, axis_scale = source_bytes(pop, man)
    keys = [k for k, *_ in PANEL]
    per_source = {}
    for slug in pop["sources"]:
        p = COUNTS / f"{slug}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        b = d["nbytes"].astype(np.float64)
        n = len(b)
        row = {"n": n, "sample_bytes": float(b.sum()), "sample_chars": float(d["nchars"].sum()),
               "mean_chars": float(d["nchars"].mean()), "tokens": {}}
        for k in keys:
            t = d[k].astype(np.float64)
            r = t.sum() / b.sum()
            resid = t - r * b
            var_r = resid.var(ddof=1) / (n * b.mean() ** 2)
            row["tokens"][k] = {"ratio": r, "se_ratio": float(np.sqrt(var_r)), "bytes_per_token": 1 / r,
                                "chars_per_token": float(d["nchars"].sum() / t.sum())}
        per_source[slug] = row
    missing = [s for s in pop["sources"] if s not in per_source]

    totals = {}
    by_axis = defaultdict(lambda: defaultdict(float))
    by_lang = defaultdict(lambda: defaultdict(float))
    var_axis = defaultdict(lambda: defaultdict(float))
    for k in keys:
        est = var = 0.0
        for slug, row in per_source.items():
            bi = B[slug]
            ti = bi * row["tokens"][k]["ratio"]
            vi = (bi ** 2) * row["tokens"][k]["se_ratio"] ** 2
            est += ti; var += vi
            ax = pop["sources"][slug]["axis"]; lg = man[slug]["requested"].get("lang") or "?"
            by_axis[ax][k] += ti; var_axis[ax][k] += vi
            by_lang[lg][k] += ti
        se = np.sqrt(var)
        totals[k] = {"tokens": est, "se": float(se), "ci95": [est - 1.96 * se, est + 1.96 * se],
                     "rel_se": float(se / est), "bytes_per_token": pop["logical_bytes"] / est}

    # cross-check on the build's reservoir heads
    cross = {}
    rp = COUNTS / "reservoir.npz"
    if rp.exists():
        d = np.load(rp, allow_pickle=True)
        axes = d["axis"]; b = d["nbytes"].astype(np.float64)
        for k in keys:
            t = d[k].astype(np.float64)
            est = 0.0
            for ax in pop["axes"]:
                m = axes == ax
                if m.any():
                    est += pop["axes"][ax]["logical_bytes"] * t[m].sum() / b[m].sum()
            cross[k] = {"tokens_from_heads": est, "ratio_to_primary": est / totals[k]["tokens"] if k in totals else None,
                        "heads_bytes_per_token": float(b.sum() / t.sum())}
        cross["_n"] = int(len(b)); cross["_mean_head_chars"] = float(d["nchars"].mean())

    out = {
        "population": {"records": pop["n"], "utf8_bytes": pop["logical_bytes"], "sources": len(pop["sources"]),
                       "note": "records the map was fitted on; text capped at 4,000 characters when fetched"},
        "sample": {"records": int(sum(r["n"] for r in per_source.values())), "bytes": float(sum(r["sample_bytes"] for r in per_source.values())),
                   "sources_sampled": len(per_source), "sources_missing": missing},
        "byte_calibration_by_axis": axis_scale,
        "tokenizers": {k: {"label": lab, "used_by": users} for k, lab, _repo, users in PANEL},
        "totals": totals,
        "by_axis": {ax: {k: {"tokens": by_axis[ax][k], "se": float(np.sqrt(var_axis[ax][k])), "bytes": pop["axes"][ax]["logical_bytes"], "records": pop["axes"][ax]["records"]} for k in keys} for ax in by_axis},
        "by_language_of_source": {lg: {k: by_lang[lg][k] for k in keys} for lg in by_lang},
        "cross_check_reservoir_heads": cross,
        "per_source": per_source,
    }
    (RESULTS / "estimate.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"sampled {out['sample']['records']:,} records, {out['sample']['bytes']/1e9:.2f} GB from {len(per_source)} sources; missing {len(missing)}")
    for k in keys:
        t = totals[k]
        print(f"{k:10s} {t['tokens']/1e9:8.2f} B tokens  ±{100*1.96*t['rel_se']:.2f}%  {t['bytes_per_token']:.2f} bytes/token" + (f"  heads-check {cross[k]['ratio_to_primary']:.3f}" if k in cross else ""))


if __name__ == "__main__":
    main()
