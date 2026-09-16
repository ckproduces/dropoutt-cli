"""B4. Does the map detect and describe coverage?

  a. Injection sensitivity: a held-out source is mixed into a held-out web
     base (C4 English, 40,000 records) at 0.25% to 10%. The source's home
     cells (the cells holding 80% of a disjoint reference sample of it) are
     read from the base-only and mixed placements; the excess mass there
     estimates the injected share, and a Poisson z-score says whether the
     excess could be sampling noise. The smallest share detected at z >= 3
     with the estimate within a factor of two is the sensitivity floor.
  b. Coverage ladder: corpora of increasing breadth. Effective coverage
     (sum of min(1, density)) and occupied cells must rise monotonically.
  c. Comparison and test-retest: the CLI's own compare() on split halves of
     one corpus (should be near-identical), on a source and its translation
     (Turkish-Alpaca is alpaca in Turkish), and on unrelated pairs.
     Test-retest: histogram cosine between disjoint halves against sample
     size, which is the question "how many records do I need to place".
  d. Off-atlas calibration: where in-family similarity sits, where machine
     formats sit, and what the fixed 0.35 cutoff does to each.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import data as D

BASE_SLUG = "allenai__c4__en"
BASE_N = 40_000
SHARES = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10]
INJECT = [
    ("coastalcph__lex_glue__eurlex", "EU regulations (eurlex)"),
    ("bigcode__commitpackft__python_train_0000.parquet", "Python commits"),
    ("microsoft__orca-math-word-problems-200k", "Math word problems (orca-math)"),
    ("turkish-nlp-suite__InstrucTurca", "Turkish instructions (InstrucTurca)"),
    ("gretelai__synthetic_text_to_sql", "Text-to-SQL (gretel)"),
    ("qiaojin__PubMedQA__pqa_artificial", "Biomedical QA (PubMedQA)"),
    ("HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet", "Turkish web (fineweb-2 tr)"),
]
REF_N = 5000


def coverage_of(name, p: C.Placed, texts):
    a = C.atlas(name)
    return a.coverage(p.best, p.l1, p.langs, scores=p.score, nearest=p.nearest,
                      embeddings=p.emb, lengths=[len(t) for t in texts], texts=None)


def home_cells(cells: np.ndarray, n: int, mass: float = 0.8) -> np.ndarray:
    h = C.histogram(cells, n)
    order = np.argsort(-h)
    cum = np.cumsum(h[order])
    k = int(np.searchsorted(cum, mass) + 1)
    return order[:k]


def injection(name):
    a = C.atlas(name)
    base_texts = D.v2cache(BASE_SLUG, 60_000, seed=5)[:BASE_N]
    pb = C.place(name, base_texts, tag="b4base")
    out = {}
    for slug, label in INJECT:
        pool = D.v2cache(slug, 20_000, seed=7)
        ref, inj = pool[:REF_N], pool[REF_N:]
        pr = C.place(name, ref, tag=f"b4ref-{slug}")
        rows = []
        for level, n_bins in (("l2", a.n_regions), ("l1", a.n_l1)):
            cells_ref = pr.nearest if level == "l2" else pr.l1
            cells_base = pb.nearest if level == "l2" else pb.l1
            H = home_cells(cells_ref[pr.placed], n_bins)
            in_h = np.zeros(n_bins, bool); in_h[H] = True
            p_ref = float(in_h[cells_ref[pr.placed]].mean())
            p_base = float(in_h[cells_base[pb.placed]].mean())
            for s in SHARES:
                k = int(round(s * BASE_N / (1 - s)))
                if k > len(inj):
                    rows.append({"level": level, "share": s, "k": k, "skipped": "pool too small"}); continue
                pi = C.place(name, inj[:k], tag=f"b4inj-{slug}-{k}")
                cells_inj = pi.nearest if level == "l2" else pi.l1
                placed_mix = int(pb.placed.sum() + pi.placed.sum())
                c_mix = int(in_h[cells_base[pb.placed]].sum() + in_h[cells_inj[pi.placed]].sum())
                expected = placed_mix * p_base
                z = (c_mix - expected) / np.sqrt(max(expected, 1.0))
                p_mix = c_mix / placed_mix
                est = (p_mix - p_base) / max(p_ref - p_base, 1e-9)
                rows.append({"level": level, "share": s, "k": k, "home_cells": int(len(H)),
                             "p_home_ref": p_ref, "p_home_base": p_base, "p_home_mix": p_mix,
                             "z": float(z), "estimated_share": float(est), "ratio_est_true": float(est / s),
                             "detected": bool(z >= 3 and 0.5 <= est / s <= 2.0)})
        # what the CLI would print: density in the source's top home cell, base vs 10% mix
        cov_base = coverage_of(name, pb, base_texts)
        k10 = int(round(0.10 * BASE_N / 0.9)); k10 = min(k10, len(inj))
        pi = C.place(name, inj[:k10], tag=f"b4inj-{slug}-{k10}")
        mix = C.Placed(name, np.vstack([pb.emb, pi.emb]), np.concatenate([pb.best, pi.best]),
                       np.concatenate([pb.score, pi.score]), np.concatenate([pb.nearest, pi.nearest]),
                       np.concatenate([pb.l1, pi.l1]), np.vstack([pb.soft_cells, pi.soft_cells]),
                       np.vstack([pb.soft_w, pi.soft_w]), pb.langs + pi.langs)
        cov_mix = coverage_of(name, mix, base_texts + inj[:k10])
        top = int(np.bincount(pr.nearest[pr.placed], minlength=a.n_regions).argmax())
        out[label] = {
            "slug": slug, "rows": rows,
            "top_home_cell": top, "top_home_cell_name": a.region_labels[top] if a.region_labels else "",
            "top_home_cell_density_base": cov_base["region_density"].get(str(top), 0.0),
            "top_home_cell_density_mix10": cov_mix["region_density"].get(str(top), 0.0),
            "effective_regions_base": cov_base["effective_regions"], "effective_regions_mix10": cov_mix["effective_regions"],
            "regions_occupied_base": cov_base["regions_occupied"], "regions_occupied_mix10": cov_mix["regions_occupied"],
        }
        out[label]["min_detected_share_l2"] = min([r["share"] for r in rows if r["level"] == "l2" and r.get("detected")], default=None)
        out[label]["min_detected_share_l1"] = min([r["share"] for r in rows if r["level"] == "l1" and r.get("detected")], default=None)
    return out


LADDER = [
    ("geometry only", [("EleutherAI__hendrycks_math__geometry", 2000)]),
    ("+ all hendrycks math", [(f"EleutherAI__hendrycks_math__{s}", 1200) for s in
                              ("algebra", "counting_and_probability", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")]),
    ("+ code (4 languages)", [(f"bigcode__commitpackft__{l}_train_0000.parquet", 2000) for l in ("python", "java", "javascript", "go")]),
    ("+ chat (ultrachat)", [("HuggingFaceH4__ultrachat_200k", 8000)]),
    ("+ legal (3 lex_glue tasks)", [(f"coastalcph__lex_glue__{t}", 2700) for t in ("eurlex", "ledgar", "ecthr_a")]),
    ("+ science (peS2o, arXiv, PubMedQA)", [("BEE-spoke-data__peS2o-100k_en-xlong", 2700), ("CShorten__ML-ArXiv-Papers", 2700), ("qiaojin__PubMedQA__pqa_artificial", 2700)]),
    ("+ English web (C4)", [("allenai__c4__en", 8000)]),
    ("+ web in 8 languages (fineweb-2)", [(slug, 1000) for slug, l in D.LANG_PANEL_WEB if l != "en"]),
    ("+ Turkish instructions + 13 wikipedias", [("turkish-nlp-suite__InstrucTurca", 4000)] + [(slug, 600) for slug, l in D.LANG_PANEL_WIKI]),
]


def ladder(name):
    texts, out = [], []
    for step, parts in LADDER:
        for slug, n in parts:
            texts += D.v2cache(slug, n, seed=11)
        p = C.place(name, texts, tag=f"b4ladder-{len(texts)}")
        cov = coverage_of(name, p, texts)
        a = C.atlas(name)
        out.append({"step": step, "records": len(texts), "placed": cov["placed"], "off_atlas_rate": cov["off_atlas_rate"],
                    "regions_occupied": cov["regions_occupied"], "regions_total": cov["regions_total"],
                    "effective_regions": cov["effective_regions"], "effective_share": cov["effective_regions"] / cov["regions_total"],
                    "l1_occupied": int(len(set(p.l1[p.placed].tolist()))), "l1_total": a.n_l1,
                    "entropy_share": cov["region_entropy"] / cov["max_region_entropy"],
                    "unreached_density": cov["density_model"].get("unreached_density")})
    # reference-like corpus at growing sample sizes: does effective coverage saturate?
    fw = D.v2cache("HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", 100_000, seed=13)
    sat = []
    for n in (2_000, 5_000, 20_000, 50_000, 100_000):
        p = C.place(name, fw[:n], tag=f"b4fw-{n}")
        cov = coverage_of(name, p, fw[:n])
        sat.append({"n": n, "regions_occupied": cov["regions_occupied"], "effective_regions": cov["effective_regions"],
                    "effective_share": cov["effective_regions"] / cov["regions_total"], "off_atlas_rate": cov["off_atlas_rate"],
                    "prior_strength": cov["density_model"].get("prior_strength"), "unreached_density": cov["density_model"].get("unreached_density")})
    return {"ladder": out, "fineweb_saturation": sat}


PAIRS = [
    ("same corpus, two halves", ("HuggingFaceH4__ultrachat_200k", 0), ("HuggingFaceH4__ultrachat_200k", 1)),
    ("alpaca vs its Turkish translation", ("tatsu-lab__alpaca", 0), ("TFLai__Turkish-Alpaca", 0)),
    ("Turkish instructions vs English instructions", ("turkish-nlp-suite__InstrucTurca", 0), ("tatsu-lab__alpaca", 0)),
    ("hendrycks algebra vs intermediate algebra", ("EleutherAI__hendrycks_math__algebra", 0), ("EleutherAI__hendrycks_math__intermediate_algebra", 0)),
    ("Python commits vs Python text-to-code", ("bigcode__commitpackft__python_train_0000.parquet", 0), ("codeparrot__xlcost-text-to-code__Python-program-level_train_0000.parquet", 0)),
    ("eurlex vs ECtHR case law", ("coastalcph__lex_glue__eurlex", 0), ("coastalcph__lex_glue__ecthr_a", 0)),
    ("Turkish web vs English web", ("HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet", 0), ("HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", 0)),
    ("Turkish wikipedia vs English wikipedia", ("wikimedia__wikipedia__20231101.tr", 0), ("wikimedia__wikipedia__20231101.en", 0)),
    ("Python commits vs ultrachat", ("bigcode__commitpackft__python_train_0000.parquet", 0), ("HuggingFaceH4__ultrachat_200k", 0)),
    ("PubMedQA vs eurlex", ("qiaojin__PubMedQA__pqa_artificial", 0), ("coastalcph__lex_glue__eurlex", 0)),
]
PAIR_N = 5000


def _half(slug, which, n=PAIR_N):
    pool = D.v2cache(slug, 2 * n, seed=17)
    half = len(pool) // 2
    return pool[:half] if which == 0 else pool[half:]


def comparisons(name):
    from dropoutt.atlas.compare import compare
    out = []
    for label, (sa, ha), (sb, hb) in PAIRS:
        ta, tb = _half(sa, ha), _half(sb, hb)
        pa, pb = C.place(name, ta, tag=f"b4pair-{sa}-{ha}"), C.place(name, tb, tag=f"b4pair-{sb}-{hb}")
        ca, cb = coverage_of(name, pa, ta), coverage_of(name, pb, tb)
        r = compare(ca, cb)
        a = C.atlas(name)
        out.append({"pair": label, "n_a": len(ta), "n_b": len(tb), "similarity": r.similarity,
                    "shared": r.shared_mass, "new": r.added_mass, "new_reverse": compare(cb, ca).added_mass,
                    "l1_hist_cosine": C.cosine(C.histogram(pa.l1[pa.placed], a.n_l1), C.histogram(pb.l1[pb.placed], a.n_l1)),
                    "off_a": pa.off_rate, "off_b": pb.off_rate, "caveats": r.caveats})
    return out


def _expected_new(hist, n, sims=20, seed=0):
    """New-between-halves if both halves were independent draws from ``hist``.

    A frozen map places deterministically, so two disjoint halves of one corpus
    differ only by sampling. This is the share of one half's mass in cells the
    other half happened to miss, under pure multinomial sampling: the floor the
    readout cannot beat at this sample size.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(sims):
        a = rng.multinomial(n, hist); b = rng.multinomial(n, hist)
        ha = a / max(a.sum(), 1)
        vals.append(float(ha[b == 0].sum()))
    return float(np.mean(vals))


def test_retest(name):
    a = C.atlas(name)
    out = {}
    for slug in ("allenai__c4__en", "HuggingFaceH4__ultrachat_200k", "HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet"):
        pool = D.v2cache(slug, 40_000, seed=19)
        p = C.place(name, pool, tag=f"b4retest-{slug}")
        rows = []
        for n in (250, 500, 1000, 2000, 5000, 10000, 20000):
            if 2 * n > len(pool):
                break
            A, B = p.nearest[:n], p.nearest[n:2 * n]
            A1, B1 = p.l1[:n], p.l1[n:2 * n]
            hA, hB = C.histogram(A, a.n_regions), C.histogram(B, a.n_regions)
            gA, gB = C.histogram(A1, a.n_l1), C.histogram(B1, a.n_l1)
            occA, occB = set(np.nonzero(hA)[0].tolist()), set(np.nonzero(hB)[0].tolist())
            new_mass = float(sum(hA[c] for c in occA - occB))
            new_l1 = float(sum(gA[c] for c in set(np.nonzero(gA)[0].tolist()) - set(np.nonzero(gB)[0].tolist())))
            pooled = C.histogram(np.concatenate([A, B]), a.n_regions)
            rows.append({"n": n, "cos_l2": C.cosine(hA, hB), "cos_l1": C.cosine(gA, gB),
                         "new_between_halves_l2": new_mass, "new_between_halves_l1": new_l1,
                         "new_expected_from_sampling_l2": _expected_new(pooled, n),
                         "cells_a": len(occA)})
        key = D.short_name(slug)
        out[key] = rows
        out[key + "_min_n_new_l2_below_5pct"] = min([r["n"] for r in rows if r["new_between_halves_l2"] < 0.05], default=None)
        out[key + "_min_n_new_l1_below_5pct"] = min([r["n"] for r in rows if r["new_between_halves_l1"] < 0.05], default=None)
    return out


def off_atlas(name):
    a = C.atlas(name)
    fam = D.v2cache("HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", 20_000, seed=23) + \
          D.v2cache("wikimedia__wikipedia__20231101.en", 3000, seed=23) + \
          D.v2cache("HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet", 3000, seed=23)
    pf = C.place(name, fam, tag="b4fam")
    ood, kinds = D.ood_texts(200)
    po = C.place(name, ood, langs=["unknown"] * len(ood), tag="b4ood")
    kinds = np.array(kinds)
    pct = {f"p{q}": float(np.percentile(pf.score, q)) for q in (1, 2, 5, 10, 50)}
    per_kind = {k: {"off_rate_at_cutoff": float((~po.placed[kinds == k]).mean()), "median_sim": float(np.median(po.score[kinds == k]))}
                for k in sorted(set(kinds.tolist()))}
    # a cutoff at the in-family 2nd percentile, the rule the docs state: what would it reject?
    t2 = pct["p2"]
    sweep = []
    for t in sorted({0.35, round(a.off_threshold, 4), 0.40, 0.45, 0.50, round(t2, 4)}):
        sweep.append({"cutoff": float(t), "in_family_off": float((pf.score < t).mean()), "ood_rejected": float((po.score < t).mean()),
                      "ood_rejected_by_kind": {k: float((po.score[kinds == k] < t).mean()) for k in sorted(set(kinds.tolist()))}})
    return {"in_family_n": len(fam), "in_family_similarity_percentiles": pct, "cutoff_in_use": a.off_threshold,
            "cutoff_calibration": a.meta.get("off_atlas_calibration"),
            "in_family_off_rate": pf.off_rate, "ood_off_rate": po.off_rate, "ood_per_kind": per_kind, "cutoff_sweep": sweep}


def main():
    res = {"products": {}}
    for name in C.PRODUCTS:
        print("B4", name, flush=True)
        res["products"][name] = {"injection": injection(name), "ladder": ladder(name),
                                 "comparisons": comparisons(name), "test_retest": test_retest(name),
                                 "off_atlas": off_atlas(name)}
    C.save("b4_coverage", res)
    for name, r in res["products"].items():
        print(name, "min detected share L2:", {k: v["min_detected_share_l2"] for k, v in r["injection"].items()})
        print(name, "ladder eff share:", [round(x["effective_share"], 3) for x in r["ladder"]["ladder"]])
        print(name, "pairs:", [(x["pair"][:22], round(x["similarity"], 2), round(x["new"], 2)) for x in r["comparisons"]])
        print(name, "ood off:", round(r["off_atlas"]["ood_off_rate"], 3), "in-family p2:", round(r["off_atlas"]["in_family_similarity_percentiles"]["p2"], 3))


if __name__ == "__main__":
    main()
