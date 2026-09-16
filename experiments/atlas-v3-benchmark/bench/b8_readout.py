"""B8. Does the density readout mean what it says?

For every area a corpus reaches, ``dropoutt atlas`` prints a density: the
corpus's share of that area divided by the reference corpus's share of the
same area. ``1.0x`` is printed as "matches the map". Two checks.

  a. Self-calibration. A 200,000-record sample of the build's own reservoir
     (a uniform sample of the records the map was fitted on) is placed the way
     a user's corpus is: detected language, the live cutoff, the CLI's own
     ``coverage()``. Every density should sit at 1.0x up to sampling noise,
     and the map's effective coverage should approach the whole map. The same
     statistics are computed for a multinomial draw from the map's stored
     reference shares, so sampling noise and real bias can be told apart.
     The reservoir also carries the build's axis and language per record, so
     the off-atlas rate on the map's own mix is broken down by both.
  b. What the readout says about typical web text: a held-out English web
     corpus (C4, 40,000 records) and an in-family sample (fineweb, 20,000).

Run after B4 so the web placements come from the cache.
"""
from __future__ import annotations

import json
import random

import numpy as np

from . import common as C
from . import data as D

N_SELF = 200_000
SEED = 41


def reservoir_rows(name: str):
    path = C.RESERVOIRS[name]
    rows = []
    with open(path, encoding="utf-8") as fh:
        fh.readline()  # {"size": ..., "seen": ...}
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows.append((r[1], r[2], r[3], r[4]))  # text, axis, lang, source
    return rows


def density_stats(counts: np.ndarray, ref_share: np.ndarray, label: str) -> dict:
    """Raw quotient densities over occupied cells, plus reference-mass-weighted views."""
    n = counts.sum()
    occ = counts > 0
    dens = (counts[occ] / n) / ref_share[occ]
    log2 = np.log2(dens)
    w = ref_share[occ] / ref_share[occ].sum()
    return {
        "label": label, "placed": int(n), "cells_occupied": int(occ.sum()),
        "median_density": float(np.median(dens)),
        "median_abs_log2": float(np.median(np.abs(log2))),
        "share_cells_within_0.5_2x": float(((dens >= 0.5) & (dens <= 2.0)).mean()),
        "share_cells_within_0.8_1.25x": float(((dens >= 0.8) & (dens <= 1.25)).mean()),
        "share_cells_above_2x": float((dens > 2.0).mean()),
        "share_cells_below_0.5x": float((dens < 0.5).mean()),
        "refmass_weighted_share_within_0.5_2x": float((w * ((dens >= 0.5) & (dens <= 2.0))).sum()),
        "unoccupied_reference_mass": float(ref_share[~occ].sum()),
    }


def self_calibration(name: str) -> dict:
    a = C.atlas(name)
    rows = reservoir_rows(name)
    rng = random.Random(SEED)
    rows = rng.sample(rows, min(N_SELF, len(rows)))
    texts = [r[0] for r in rows]
    axis = np.array([r[1] for r in rows]); lang = np.array([r[2] for r in rows])
    p = C.place(name, texts, tag=f"b8self-{name}")
    cov = a.coverage(p.best, p.l1, p.langs, scores=p.score, nearest=p.nearest,
                     embeddings=p.emb, lengths=[len(t) for t in texts])
    ref = np.asarray(a.region_size, dtype=np.float64); ref_share = ref / ref.sum()
    counts = np.bincount(p.best[p.placed], minlength=a.n_regions).astype(np.float64)
    real = density_stats(counts, ref_share, "reservoir sample, raw quotient")
    # the CLI's own shrunk estimate
    cli = np.array(list(cov["region_density"].values()), dtype=np.float64)
    real_cli = {"median_density": float(np.median(cli)),
                "share_cells_within_0.5_2x": float(((cli >= 0.5) & (cli <= 2.0)).mean()),
                "share_cells_within_0.8_1.25x": float(((cli >= 0.8) & (cli <= 1.25)).mean()),
                "prior_strength": cov["density_model"].get("prior_strength"),
                "unreached_density": cov["density_model"].get("unreached_density")}
    # multinomial expectation from the stored reference shares, same n
    g = np.random.default_rng(SEED)
    sims = [density_stats(g.multinomial(int(counts.sum()), ref_share).astype(np.float64), ref_share, "multinomial")
            for _ in range(10)]
    expected = {k: float(np.mean([s[k] for s in sims])) for k in sims[0] if isinstance(sims[0][k], (int, float))}
    # off-atlas on the map's own mix, by axis and language
    by_axis = {ax: {"n": int((axis == ax).sum()), "off_rate": float((~p.placed[axis == ax]).mean()),
                    "median_sim": float(np.median(p.score[axis == ax]))} for ax in sorted(set(axis.tolist()))}
    langs_sorted = sorted(set(lang.tolist()), key=lambda l: -(lang == l).sum())
    by_lang = {l: {"n": int((lang == l).sum()), "off_rate": float((~p.placed[lang == l]).mean())}
               for l in langs_sorted[:25]}
    # chi-square style: how far is the placed histogram from the reference shares, vs a multinomial draw?
    exp_counts = counts.sum() * ref_share
    chi_real = float((((counts - exp_counts) ** 2) / np.maximum(exp_counts, 1e-9)).sum())
    chi_sim = float(np.mean([(((g.multinomial(int(counts.sum()), ref_share) - exp_counts) ** 2) / np.maximum(exp_counts, 1e-9)).sum()
                             for _ in range(10)]))
    # which cells deviate most (real over-/under-density beyond noise)
    dens = (counts / counts.sum()) / ref_share
    z = (counts - exp_counts) / np.sqrt(np.maximum(exp_counts, 1e-9))
    worst_over = [(int(i), round(float(dens[i]), 2), round(float(z[i]), 1), a.region_labels[int(i)] if a.region_labels else "")
                  for i in np.argsort(-z)[:8]]
    worst_under = [(int(i), round(float(dens[i]), 2), round(float(z[i]), 1), a.region_labels[int(i)] if a.region_labels else "")
                   for i in np.argsort(z)[:8]]
    return {
        "n_sampled": len(texts), "placed": int(p.placed.sum()), "off_atlas_rate": p.off_rate,
        "cutoff": a.off_threshold, "off_by_axis": by_axis, "off_by_language_top25": by_lang,
        "regions_occupied": cov["regions_occupied"], "effective_regions": cov["effective_regions"],
        "effective_share": cov["effective_regions"] / cov["regions_total"],
        "raw": real, "cli": real_cli, "expected_multinomial": expected,
        "chi2_real": chi_real, "chi2_expected_multinomial": chi_sim, "chi2_ratio": chi_real / max(chi_sim, 1e-9),
        "dof": int(a.n_regions - 1),
        "most_overdense_cells": worst_over, "most_underdense_cells": worst_under,
        "detected_vs_build_language_agreement": float(np.mean([x == y for x, y in zip(p.langs, lang.tolist())])),
    }


def web_readout(name: str) -> dict:
    a = C.atlas(name)
    out = {}
    for label, slug, n, seed, tag in (("C4 English web (held out)", "allenai__c4__en", 40_000, 5, "b4base"),
                                      ("fineweb English (in family)", "HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", 20_000, 23, None)):
        if tag == "b4base":
            texts = D.v2cache(slug, 60_000, seed=seed)[:n]
            p = C.place(name, texts, tag=tag)
        else:
            texts = D.v2cache(slug, 100_000, seed=13)[:n]
            p = C.place(name, texts, tag=f"b4fw-{n}")
        cov = a.coverage(p.best, p.l1, p.langs, scores=p.score, nearest=p.nearest, embeddings=p.emb,
                         lengths=[len(t) for t in texts])
        ref = np.asarray(a.region_size, dtype=np.float64); ref_share = ref / ref.sum()
        counts = np.bincount(p.best[p.placed], minlength=a.n_regions).astype(np.float64)
        st = density_stats(counts, ref_share, label)
        cli = cov["region_density"]
        top = sorted(cli.items(), key=lambda kv: -kv[1])[:6]
        out[label] = {**st, "off_atlas_rate": p.off_rate, "effective_share": cov["effective_regions"] / cov["regions_total"],
                      "regions_occupied": cov["regions_occupied"],
                      "densest_cells_cli": [(int(c), d, a.region_labels[int(c)] if a.region_labels else "") for c, d in top],
                      "l1_share_top5": [(int(i), round(float(v), 3), a.l1_labels[int(i)] if a.l1_labels else "")
                                        for i, v in sorted(enumerate(C.histogram(p.l1[p.placed], a.n_l1)), key=lambda kv: -kv[1])[:5]]}
    return out


def main():
    res = {"products": {}}
    for name in ("atlas-v3", "atlas-v3-prev"):
        print("B8", name, flush=True)
        res["products"][name] = {"self_calibration": self_calibration(name), "web": web_readout(name)}
    C.save("b8_readout", res)
    for name, r in res["products"].items():
        s = r["self_calibration"]
        print(f"{name:14s} self: off {s['off_atlas_rate']:.4f} eff {s['effective_share']:.3f} "
              f"raw within 0.5-2x {s['raw']['share_cells_within_0.5_2x']:.3f} (multinomial {s['expected_multinomial']['share_cells_within_0.5_2x']:.3f}) "
              f"chi2 ratio {s['chi2_ratio']:.2f} lang agree {s['detected_vs_build_language_agreement']:.3f}")
        for label, w in r["web"].items():
            print(f"   {label}: off {w['off_atlas_rate']:.4f} eff {w['effective_share']:.3f} >2x {w['share_cells_above_2x']:.3f} <0.5x {w['share_cells_below_0.5x']:.3f}")


if __name__ == "__main__":
    main()
