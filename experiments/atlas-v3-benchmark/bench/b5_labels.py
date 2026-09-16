"""B5. Are the names useful to a researcher?

  a. Name alignment: each cell's hand-written name is embedded with the same
     encoder and ranked against every centroid. A name that describes its
     cell should rank its own centroid near the top. Done for L2 names
     against 4,096 cells and for L1 names against 256 regions, on the
     current and the previous v3; v2's auto-captions are the comparison.
  b. Exemplar self-consistency: the 4 exemplar texts stored per cell are
     placed back. They should land in their own cell or at least their
     own region.
  c. Name hygiene: duplicate names, language words, shared content words.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from . import common as C

LANG_WORDS = {"english", "turkish", "arabic", "chinese", "german", "french", "spanish", "russian", "japanese", "korean",
              "hindi", "persian", "italian", "portuguese", "dutch", "polish", "vietnamese", "indonesian", "thai", "greek",
              "hebrew", "swedish", "danish", "finnish", "czech", "hungarian", "romanian", "ukrainian", "bulgarian", "urdu",
              "bengali", "tamil", "telugu", "malay", "swahili", "azerbaijani", "georgian", "lithuanian", "latvian",
              "estonian", "slovak", "slovenian", "croatian", "serbian", "norwegian", "catalan", "basque", "galician"}


def _rank_of_own(name_vecs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    sims = name_vecs @ centroids.T
    own = np.diag(sims)
    return (sims > own[:, None]).sum(axis=1) + 1


def alignment(name):
    a = C.atlas(name)
    out = {}
    l2 = a.region_labels
    if l2 and len(l2) == a.n_regions:
        v = a.project(C.encode(name, l2))
        r = _rank_of_own(v, a.centroids)
        # within-family rank: own cell among its L1 siblings
        fam = []
        for i in range(a.n_regions):
            sib = np.nonzero(a.region_category == a.region_category[i])[0]
            s = v[i] @ a.centroids[sib].T
            fam.append(int((s > v[i] @ a.centroids[i]).sum() + 1))
        out["l2"] = {"n": a.n_regions, "median_rank": float(np.median(r)), "top1": float((r == 1).mean()),
                     "top10": float((r <= 10).mean()), "top100": float((r <= 100).mean()),
                     "median_rank_within_family": float(np.median(fam)), "top1_within_family": float((np.array(fam) == 1).mean()),
                     "source": a.meta.get("region_labels_source"), "worst": [(int(i), l2[int(i)], int(r[int(i)])) for i in np.argsort(-r)[:8]]}
    l1 = a.l1_labels
    if l1 and a.l1_centroids is not None and len(l1) == a.n_l1:
        v = a.project(C.encode(name, l1))
        r = _rank_of_own(v, a.l1_centroids)
        out["l1"] = {"n": a.n_l1, "median_rank": float(np.median(r)), "top1": float((r == 1).mean()), "top5": float((r <= 5).mean()),
                     "source": a.meta.get("l1_labels_source"), "worst": [(int(i), l1[int(i)], int(r[int(i)])) for i in np.argsort(-r)[:8]]}
    return out


def exemplars(name):
    import json
    d = np.load(C.PRODUCTS[name], allow_pickle=True)
    if "exemplar_texts" not in d.files:
        # The wheel ships the artifact with exemplars stripped; the build's own
        # copy keeps them. Use it only if its centroids are the shipped ones.
        if name != C.MAIN or not C.EXEMPLAR_ARTIFACT.exists():
            return None
        full = np.load(C.EXEMPLAR_ARTIFACT, allow_pickle=True)
        if "exemplar_texts" not in full.files or not np.array_equal(full["centroids"], d["centroids"]):
            return None
        d = full
    ex = d["exemplar_texts"]
    n, k = ex.shape
    texts = [str(t) for row in ex for t in row]
    owner = np.repeat(np.arange(n), k)
    a = C.atlas(name)
    p = C.place(name, texts, tag=f"b5ex-{name}")
    same_cell = p.nearest == owner
    same_l1 = p.l1 == a.region_category[owner]
    return {"n": len(texts), "chars": int(ex.dtype.itemsize // 4), "own_cell": float(same_cell.mean()), "own_l1": float(same_l1.mean()),
            "off_atlas": p.off_rate, "own_cell_in_top5_soft": float(np.mean([owner[i] in set(p.soft_cells[i].tolist()) for i in range(len(texts))]))}


def hygiene(name):
    a = C.atlas(name)
    out = {}
    for level, names in (("l2", a.region_labels), ("l1", a.l1_labels)):
        if not names:
            continue
        low = [n.lower() for n in names]
        dup = sum(c - 1 for c in Counter(low).values() if c > 1)
        words = [set(re.findall(r"[a-zà-ÿ']+", n)) for n in low]
        lang_hits = [names[i] for i, w in enumerate(words) if w & LANG_WORDS]
        vocab = Counter(w for ws in words for w in ws)
        out[level] = {"n": len(names), "duplicates": dup, "names_with_language_word": len(lang_hits),
                      "language_word_examples": lang_hits[:6], "distinct_words": len(vocab),
                      "mean_words": float(np.mean([len(n.split()) for n in names]))}
    return out


def main():
    res = {"products": {}}
    for name in C.PRODUCTS:
        print("B5", name, flush=True)
        res["products"][name] = {"alignment": alignment(name), "exemplars": exemplars(name), "hygiene": hygiene(name)}
    C.save("b5_labels", res)
    for name, r in res["products"].items():
        al = r["alignment"]
        print(name, "L2 align:", {k: v for k, v in al.get("l2", {}).items() if k not in ("worst", "source")}, "| L1:", {k: v for k, v in al.get("l1", {}).items() if k not in ("worst", "source")}, "| ex:", r["exemplars"])


if __name__ == "__main__":
    main()
