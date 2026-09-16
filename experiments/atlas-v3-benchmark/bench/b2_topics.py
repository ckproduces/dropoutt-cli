"""B2. Topic categorization on labelled evaluation sets never used to build.

MMLU (57 subjects, English), MMLU-Pro (14 categories, English) and EXAMS-tr
(8 school subjects, Turkish). A question plus its options is placed; the cell
is scored against the subject the benchmark authors assigned.
"""
from __future__ import annotations

import numpy as np

from . import common as C
from . import data as D


def evaluate(name, texts, labels, tag, seed=0, hits_out=None):
    from collections import Counter
    y = np.array(labels)
    p = C.place(name, texts, tag=tag)
    a = C.atlas(name)
    tr, te = C.split_halves(len(texts), seed)
    l2, l1 = p.nearest, p.l1
    x = a.project(p.emb, p.langs)
    chance = Counter(y[tr].tolist()).most_common(1)[0][1] / len(tr)
    hits2 = C.majority_map_hits(l2[tr], y[tr], l2[te], y[te])
    hits1 = C.majority_map_hits(l1[tr], y[tr], l1[te], y[te])
    ceil = C.linear_probe_accuracy(x[tr], y[tr], x[te], y[te])
    if hits_out is not None:
        hits_out["l2"] = hits2; hits_out["l1"] = hits1
    out = {
        "n": len(texts), "n_classes": int(len(set(labels))), "off_atlas_rate": p.off_rate,
        "chance_majority": chance,
        "nmi_l2": C.nmi(y, l2), "ami_l2": C.ami(y, l2), "nmi_l1": C.nmi(y, l1), "ami_l1": C.ami(y, l1),
        "purity_l2": C.purity(l2, y), "purity_l1": C.purity(l1, y),
        "acc_l2": float(hits2.mean()), "acc_l2_ci95": C.bootstrap_ci(hits2),
        "acc_l1": float(hits1.mean()), "acc_l1_ci95": C.bootstrap_ci(hits1),
        "ceiling_centroid": C.centroid_probe_accuracy(x[tr], y[tr], x[te], y[te]),
        "ceiling_linear": ceil,
        "retained_l2": C.retained(float(hits2.mean()), chance, ceil),
        "retained_l1": C.retained(float(hits1.mean()), chance, ceil),
        "n_test": int(len(te)),
        "cells_used_l2": int(len(set(l2.tolist()))), "cells_used_l1": int(len(set(l1.tolist()))),
    }
    per = {}
    for s in sorted(set(labels)):
        m = y == s
        cells, counts = np.unique(l1[m], return_counts=True)
        order = np.argsort(-counts)[:3]
        per[s] = {
            "n": int(m.sum()),
            "top_l1": [(int(cells[i]), float(counts[i] / m.sum()),
                        a.l1_labels[int(cells[i])] if a.l1_labels else "") for i in order],
            "l1_used": int(len(cells)),
        }
        c2, n2 = np.unique(l2[m], return_counts=True)
        j = n2.argmax()
        rl = a.region_labels
        per[s]["top_l2"] = (int(c2[j]), float(n2[j] / m.sum()), rl[int(c2[j])] if rl else "")
    out["per_subject"] = per
    return out


def main():
    sets = {"mmlu": D.mmlu(), "mmlu_pro": D.mmlu_pro(), "exams_tr": D.exams_tr()}
    res = {"products": {}, "paired_vs_main": {}}
    hits = {}
    for name in C.PRODUCTS:
        res["products"][name] = {}
        hits[name] = {}
        for key, (texts, labels) in sets.items():
            print("B2", name, key, flush=True)
            hits[name][key] = {}
            res["products"][name][key] = evaluate(name, texts, labels, tag=f"b2-{key}", hits_out=hits[name][key])
    for name in C.PRODUCTS:
        if name == C.MAIN:
            continue
        res["paired_vs_main"][name] = {f"{key}_{lvl}": C.paired_bootstrap_delta(hits[C.MAIN][key][lvl], hits[name][key][lvl])
                                       for key in sets for lvl in ("l2", "l1")}
    C.save("b2_topics", res)
    for name, r in res["products"].items():
        for key in sets:
            q = r[key]
            print(f"{name:14s} {key:9s} acc L2 {q['acc_l2']:.3f} L1 {q['acc_l1']:.3f} ceil {q['ceiling_linear']:.3f} "
                  f"chance {q['chance_majority']:.3f} AMI L2 {q['ami_l2']:.3f} L1 {q['ami_l1']:.3f} off {q['off_atlas_rate']:.3f}")


if __name__ == "__main__":
    main()
