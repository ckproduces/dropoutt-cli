"""B1. Does the map keep sources apart? Held-out source panel, 52 sources.

Records from 52 sources that were not v3 build inputs are placed; the cell a
record lands in is then read as a label and scored against the source it came
from at three granularities: source (52-way), axis (8-way: code, math, legal /
finance, scientific, instruction, dialogue, structured, web/news) and, inside
code, the programming language.

Chance is the majority class; the ceiling is a supervised probe on the same
projected vectors, which says how much of what the encoder knows the frozen
grid keeps.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import data as D

N_PER = 1000


def load_panel():
    texts, src, axis, lang = [], [], [], []
    for slug, ax, lg in D.SOURCE_PANEL:
        t = D.v2cache(slug, N_PER, seed=1)
        texts += t
        src += [D.short_name(slug)] * len(t)
        axis += [ax] * len(t)
        lang += [lg] * len(t)
    return texts, np.array(src), np.array(axis), np.array(lang)


def evaluate(name, texts, src, axis, lang, seed=0, hits_out=None):
    p = C.place(name, texts, tag="b1panel")
    a = C.atlas(name)
    tr, te = C.split_halves(len(texts), seed)
    l2 = p.nearest          # nearest cell regardless of cutoff (label reading)
    l1 = p.l1
    x = a.project(p.emb, p.langs)
    out = {"off_atlas_rate": p.off_rate, "cells_used_l2": int(len(set(l2.tolist()))),
           "cells_used_l1": int(len(set(l1.tolist())))}
    for gran, y in (("source", src), ("axis", axis)):
        maj = float((y == np.bincount(np.unique(y, return_inverse=True)[1]).argmax()).mean()) if False else None
        from collections import Counter
        chance = Counter(y[tr].tolist()).most_common(1)[0][1] / len(tr)
        hits2 = C.majority_map_hits(l2[tr], y[tr], l2[te], y[te])
        hits1 = C.majority_map_hits(l1[tr], y[tr], l1[te], y[te])
        ceil = C.linear_probe_accuracy(x[tr], y[tr], x[te], y[te])
        out[gran] = {
            "chance_majority": chance,
            "nmi_l2": C.nmi(y, l2), "ami_l2": C.ami(y, l2),
            "nmi_l1": C.nmi(y, l1), "ami_l1": C.ami(y, l1),
            "purity_l2": C.purity(l2, y), "purity_l1": C.purity(l1, y),
            "acc_l2": float(hits2.mean()), "acc_l2_ci95": C.bootstrap_ci(hits2),
            "acc_l1": float(hits1.mean()), "acc_l1_ci95": C.bootstrap_ci(hits1),
            "acc_soft_top5_l2": _soft_acc(p, y, tr, te),
            "ceiling_centroid": C.centroid_probe_accuracy(x[tr], y[tr], x[te], y[te]),
            "ceiling_linear": ceil,
            "retained_l2": C.retained(float(hits2.mean()), chance, ceil),
            "retained_l1": C.retained(float(hits1.mean()), chance, ceil),
            "n_test": int(len(te)),
        }
        if hits_out is not None:
            hits_out[(gran, "l2")] = hits2; hits_out[(gran, "l1")] = hits1
    # inside code: does the cell know the language?
    code = axis == "code"
    y = lang[code]; cl2 = l2[code]; cl1 = l1[code]
    ctr, cte = C.split_halves(int(code.sum()), seed)
    out["code_language"] = {
        "n": int(code.sum()), "n_classes": int(len(set(y.tolist()))),
        "chance_majority": float(max(np.unique(y, return_counts=True)[1]) / len(y)),
        "ami_l2": C.ami(y, cl2), "acc_l2": C.majority_map_accuracy(cl2[ctr], y[ctr], cl2[cte], y[cte]),
        "ceiling_centroid": C.centroid_probe_accuracy(x[code][ctr], y[ctr], x[code][cte], y[cte]),
    }
    # per-source: purity of the record's cell wrt its source, and top L1 name
    per = {}
    for s in sorted(set(src.tolist())):
        m = src == s
        cells, counts = np.unique(l1[m], return_counts=True)
        top = cells[counts.argmax()]
        share = counts.max() / m.sum()
        per[s] = {"n": int(m.sum()), "off_atlas": float((~p.placed[m]).mean()),
                  "top_l1": int(top), "top_l1_share": float(share),
                  "top_l1_name": a.l1_labels[int(top)] if a.l1_labels else "",
                  "l1_used": int(len(cells)), "l2_used": int(len(set(l2[m].tolist()))),
                  "top_l2": int(np.bincount(l2[m][l2[m] >= 0]).argmax()),
                  }
        rl = a.region_labels
        per[s]["top_l2_name"] = rl[per[s]["top_l2"]] if rl else ""
    out["per_source"] = per
    return out


def _soft_acc(p, y, tr, te):
    """Majority map on hard cells, scored if the mapped label of any top-5
    soft cell with weight >= 0.15 matches: does the soft answer contain it?"""
    from collections import Counter, defaultdict
    by = defaultdict(Counter)
    for c, l in zip(p.nearest[tr], y[tr]):
        by[c][l] += 1
    table = {c: cnt.most_common(1)[0][0] for c, cnt in by.items()}
    hits = 0
    for i in te:
        cells = p.soft_cells[i]; w = p.soft_w[i]
        cand = [table.get(int(c)) for c, ww in zip(cells, w) if c >= 0 and ww >= 0.15]
        if not cand:
            cand = [table.get(int(p.nearest[i]))]
        hits += y[i] in cand
    return hits / len(te)


def main():
    texts, src, axis, lang = load_panel()
    res = {"n_records": len(texts), "n_sources": int(len(set(src.tolist()))),
           "sources": sorted(set(src.tolist())), "products": {}, "paired_vs_main": {}}
    hits = {}
    for name in C.PRODUCTS:
        print("B1", name, flush=True)
        hits[name] = {}
        res["products"][name] = evaluate(name, texts, src, axis, lang, hits_out=hits[name])
    # Same records, same split: the difference between two maps is paired.
    for name in C.PRODUCTS:
        if name == C.MAIN:
            continue
        res["paired_vs_main"][name] = {f"{g}_{lvl}": C.paired_bootstrap_delta(hits[C.MAIN][(g, lvl)], hits[name][(g, lvl)])
                                       for (g, lvl) in hits[C.MAIN]}
    C.save("b1_sources", res)
    for name, r in res["products"].items():
        print(f"{name:14s} src acc L2 {r['source']['acc_l2']:.3f} L1 {r['source']['acc_l1']:.3f} "
              f"ceil {r['source']['ceiling_linear']:.3f} | axis acc L2 {r['axis']['acc_l2']:.3f} "
              f"L1 {r['axis']['acc_l1']:.3f} ceil {r['axis']['ceiling_linear']:.3f} | "
              f"AMI src L2 {r['source']['ami_l2']:.3f} | off {r['off_atlas_rate']:.3f}")


if __name__ == "__main__":
    main()
