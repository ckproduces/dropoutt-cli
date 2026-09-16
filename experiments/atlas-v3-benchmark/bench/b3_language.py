"""B3. Is the map about subjects rather than languages?

Three readings.
  a. The designed probe file: 14 topics written in en/tr/ar/zh. Same topic
     across languages should share a cell; different topics in one language
     should not; the OOD block must come back off-atlas.
  b. WMT17 tr-en parallel sentences: a sentence and its translation should
     land in the same L1 region and, ideally, the same cell. Chance is the
     probability two independent draws from the marginal share a bin.
  c. Language clustering vitals on web and encyclopedic panels: NMI(cell,
     language) and the lift of the dominant language in a cell over its base
     rate. Lower is better; both survive a change in the English base rate.
For the current v3 the per-language centering is also switched off to show
what it contributes.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import common as C
from . import data as D


def probe_block(name, use_langs=True):
    pr = D.probes()
    texts = [x["text"] for x in pr["probes"]]
    topics = [x["topic"] for x in pr["probes"]]
    langs = [x["lang"] for x in pr["probes"]]
    p = C.place(name, texts, langs=langs, tag="b3probes", use_langs=use_langs)
    same_l1 = same_l2 = n_same = 0
    diff_l1 = diff_l2 = n_diff = 0
    for i, j in itertools.combinations(range(len(texts)), 2):
        if topics[i] == topics[j]:
            n_same += 1; same_l1 += p.l1[i] == p.l1[j]; same_l2 += p.nearest[i] == p.nearest[j]
        elif langs[i] == langs[j]:
            n_diff += 1; diff_l1 += p.l1[i] == p.l1[j]; diff_l2 += p.nearest[i] == p.nearest[j]
    ood_texts = [x["text"] for x in pr["ood"]]
    po = C.place(name, ood_texts, langs=["unknown"] * len(ood_texts), tag="b3ood", use_langs=use_langs)
    return {
        "same_topic_pairs": n_same, "same_topic_l1_agree": same_l1 / n_same, "same_topic_l2_agree": same_l2 / n_same,
        "same_topic_l1_agree_ci95": C.wilson(int(same_l1), n_same),
        "diff_topic_same_lang_pairs": n_diff, "diff_topic_l1_collide": diff_l1 / n_diff, "diff_topic_l2_collide": diff_l2 / n_diff,
        "probe_off_atlas": p.off_rate,
        "ood_off_atlas": po.off_rate, "ood_names": [x["name"] for x in pr["ood"]], "ood_scores": po.score.round(3).tolist(),
        "per_topic_l1": {t: sorted({int(p.l1[i]) for i in range(len(texts)) if topics[i] == t}) for t in sorted(set(topics))},
    }


def wmt_block(name, n=8000, use_langs=True):
    en, tr = D.wmt_pairs(n)
    pe = C.place(name, en, langs=["en"] * len(en), tag="b3wmt-en", use_langs=use_langs)
    pt = C.place(name, tr, langs=["tr"] * len(tr), tag="b3wmt-tr", use_langs=use_langs)
    a = C.atlas(name)
    both = pe.placed & pt.placed
    h1 = C.histogram(np.concatenate([pe.l1, pt.l1]), a.n_l1)
    h2 = C.histogram(np.concatenate([pe.nearest, pt.nearest]), a.n_regions)
    top5 = np.array([(pt.nearest[i] in set(pe.soft_cells[i].tolist())) or (pe.nearest[i] in set(pt.soft_cells[i].tolist()))
                     for i in range(len(en))])
    return {
        "pairs": len(en),
        "off_atlas_en": pe.off_rate, "off_atlas_tr": pt.off_rate,
        "l1_agree": float((pe.l1 == pt.l1).mean()), "l1_agree_chance": C.chance_agreement(h1),
        "l1_agree_ci95": C.wilson(int((pe.l1 == pt.l1).sum()), len(en)),
        "l2_agree": float((pe.nearest == pt.nearest).mean()), "l2_agree_chance": C.chance_agreement(h2),
        "l2_agree_ci95": C.wilson(int((pe.nearest == pt.nearest).sum()), len(en)),
        "l2_in_top5_soft": float(top5.mean()), "l2_in_top5_soft_ci95": C.wilson(int(top5.sum()), len(en)),
        "l1_agree_both_placed": float((pe.l1[both] == pt.l1[both]).mean()) if both.any() else None,
        "median_sim_en": float(np.median(pe.score)), "median_sim_tr": float(np.median(pt.score)),
        "hist_cosine_l1": C.cosine(C.histogram(pe.l1, a.n_l1), C.histogram(pt.l1, a.n_l1)),
        "hist_cosine_l2": C.cosine(C.histogram(pe.nearest, a.n_regions), C.histogram(pt.nearest, a.n_regions)),
    }


def lang_vitals(name, panel, n_per, tag, use_langs=True):
    texts, langs = [], []
    for slug, lg in panel:
        t = D.v2cache(slug, n_per, seed=3)
        texts += t; langs += [lg] * len(t)
    langs = np.array(langs)
    p = C.place(name, texts, tag=tag, use_langs=use_langs)
    a = C.atlas(name)
    out = {"n": len(texts), "n_langs": int(len(set(langs.tolist()))), "off_atlas_rate": p.off_rate}
    base = {l: float((langs == l).mean()) for l in set(langs.tolist())}
    for level, cells, total in (("l2", p.nearest, a.n_regions), ("l1", p.l1, a.n_l1)):
        out[f"nmi_{level}"] = C.nmi(langs, cells)
        out[f"ami_{level}"] = C.ami(langs, cells)
        # dominant-language lift: for each occupied cell, share of the top
        # language divided by that language's base rate; record-weighted mean.
        lifts, wts, pure = [], [], 0
        for c in np.unique(cells):
            m = cells == c
            ls, cnt = np.unique(langs[m], return_counts=True)
            k = cnt.argmax()
            share = cnt[k] / m.sum()
            lifts.append(share / base[ls[k]]); wts.append(m.sum())
            pure += m.sum() if share >= 0.9 else 0
        lifts, wts = np.array(lifts), np.array(wts)
        out[f"dominant_lift_{level}"] = float((lifts * wts).sum() / wts.sum())
        out[f"share_records_in_90pct_single_language_cells_{level}"] = float(pure / len(texts))
        out[f"cells_used_{level}"] = int(len(np.unique(cells)))
    per_lang = {}
    for l in sorted(base):
        m = langs == l
        per_lang[l] = {"off_atlas": float((~p.placed[m]).mean()), "median_sim": float(np.median(p.score[m])),
                       "l1_used": int(len(set(p.l1[m].tolist()))), "l2_used": int(len(set(p.nearest[m].tolist())))}
    out["per_language"] = per_lang
    # topical read: are the top L1 regions of each language the same regions?
    tops = {l: set(np.argsort(-C.histogram(p.l1[langs == l], a.n_l1))[:10].tolist()) for l in base}
    pairs = list(itertools.combinations(sorted(base), 2))
    out["mean_top10_l1_overlap_between_languages"] = float(np.mean([len(tops[x] & tops[y]) / 10 for x, y in pairs]))
    return out


def main():
    res = {"products": {}}
    for name in C.PRODUCTS:
        print("B3", name, flush=True)
        r = {"probes": probe_block(name), "wmt17": wmt_block(name),
             "web_langs": lang_vitals(name, D.LANG_PANEL_WEB, 4000, "b3web"),
             "wiki_langs": lang_vitals(name, D.LANG_PANEL_WIKI, 3000, "b3wiki")}
        res["products"][name] = r
    print("B3 atlas-v3 without per-language centering", flush=True)
    res["products"]["atlas-v3 (global mean)"] = {
        "probes": probe_block("atlas-v3", use_langs=False), "wmt17": wmt_block("atlas-v3", use_langs=False),
        "web_langs": lang_vitals("atlas-v3", D.LANG_PANEL_WEB, 4000, "b3web", use_langs=False),
        "wiki_langs": lang_vitals("atlas-v3", D.LANG_PANEL_WIKI, 3000, "b3wiki", use_langs=False)}
    C.save("b3_language", res)
    for name, r in res["products"].items():
        print(f"{name:22s} probes same-topic L1 {r['probes']['same_topic_l1_agree']:.2f} L2 {r['probes']['same_topic_l2_agree']:.2f} "
              f"diff-topic L1 collide {r['probes']['diff_topic_l1_collide']:.2f} ood off {r['probes']['ood_off_atlas']:.2f} | "
              f"wmt L1 {r['wmt17']['l1_agree']:.3f} (chance {r['wmt17']['l1_agree_chance']:.3f}) L2 {r['wmt17']['l2_agree']:.3f} top5 {r['wmt17']['l2_in_top5_soft']:.3f} | "
              f"web NMI L2 {r['web_langs']['nmi_l2']:.3f} lift {r['web_langs']['dominant_lift_l2']:.2f} | wiki NMI L2 {r['wiki_langs']['nmi_l2']:.3f} lift {r['wiki_langs']['dominant_lift_l2']:.2f}")


if __name__ == "__main__":
    main()
