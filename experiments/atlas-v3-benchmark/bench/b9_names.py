"""B9. Do the names describe what actually lands there? Read from outside, and blind.

B5 checks a name against its cell with the atlas's own encoder, which built the
cell and so agrees with it by construction. This test uses a second opinion.

  a. External alignment. Every cell's name and twenty of its reservoir members
     (the B7 draw) are embedded with paraphrase-multilingual-MiniLM-L12-v2, an
     encoder the atlas never used. The name should be closer to its own cell's
     members than to the other 4,095 cells' members. Reported as the rank of the
     own cell, at L2 (4,096 names) and L1 (256 names), for the shipped map and
     the map it replaced.
  b. Blind sheet over held-out slices. For each of the 52 panel datasets (B1)
     and the 57 MMLU subjects (B2), the name of the cell that holds most of
     that slice's records on each v3 map. Items are shuffled and the map is
     hidden. A reader scores each 0 (wrong), 1 (partly) or 2 (describes it);
     ``--score`` joins the answers, reports each map's share with a Wilson
     interval and pairs the two maps on the same slices.
  c. Blind cell read (reduced). Forty atlas-v3 cells, five per coherence
     octile, ten members each spread from centre to edge, with the current name;
     the reader classifies the cell (subject / form / grab-bag) and grades the
     name (accurate / vague / soup / false-specific / wrong).

    python -m bench.b9_names            # a + write the two blind sheets
    python -m bench.b9_names --score    # join results/names_scores.json and
                                        # results/b7_judge/atlas-v3/mini_answers.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

import numpy as np

from . import b1_sources as B1
from . import b7_cells as B7
from . import common as C
from . import data as D

sys.path.insert(0, str(C.REPO / "tools"))
import minilm_numpy  # noqa: E402
from atlas_naming_worklist import spread_sample  # noqa: E402

SHEET = C.RESULTS / "names_blind.json"
KEY = C.RESULTS / "names_key.json"
SCORES = C.RESULTS / "names_scores.json"
MINI_DIR = C.RESULTS / "b7_judge" / "atlas-v3"
MINI_CELLS_PER_OCTILE = 5
MINI_MEMBERS = 10


def _member_vectors(name: str):
    """MiniLM vectors of the B7 members, from B7's cache when it ran first."""
    reservoir = str(C.RESERVOIRS[name])
    texts, langs, cells, picked, _ = B7._members(name, reservoir)
    n = len(picked)
    rows = [int(i) for c in range(n) for i in picked[c][:B7.MEMBERS]]
    owner = np.array([c for c in range(n) for _ in picked[c][:B7.MEMBERS]])
    cache = C.SCRATCH / f"b7-minilm-{name}-{B7._digest(reservoir + C.fingerprint(name))}.npz"
    if cache.exists():
        vectors = np.load(cache)["v"]
    else:
        vectors = minilm_numpy.encode_parallel([texts[i] for i in rows])
        np.savez(cache, v=vectors)
    return vectors, owner


def _rank_own(name_vecs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    sims = name_vecs @ targets.T
    own = np.diag(sims)
    return (sims > own[:, None]).sum(axis=1) + 1


def external_alignment(name: str) -> dict:
    a = C.atlas(name)
    vectors, owner = _member_vectors(name)
    n = a.n_regions
    cent = np.zeros((n, vectors.shape[1]), dtype=np.float32)
    for c in range(n):
        m = owner == c
        if m.any():
            v = vectors[m].mean(axis=0); cent[c] = v / (np.linalg.norm(v) + 1e-9)
    names = a.region_labels
    nv = minilm_numpy.encode(list(names))
    r = _rank_own(nv, cent)
    fam = []
    for i in range(n):
        sib = np.nonzero(a.region_category == a.region_category[i])[0]
        s = nv[i] @ cent[sib].T
        fam.append(int((s > nv[i] @ cent[i]).sum() + 1))
    fam = np.array(fam)
    # chance: a random name ranks uniformly, median n/2
    out = {"l2": {"n": n, "median_rank": float(np.median(r)), "top1": float((r == 1).mean()),
                  "top1_ci95": C.wilson(int((r == 1).sum()), n),
                  "top10": float((r <= 10).mean()), "top100": float((r <= 100).mean()),
                  "median_rank_within_family": float(np.median(fam)), "top1_within_family": float((fam == 1).mean()),
                  "chance_median_rank": n / 2,
                  "worst": [(int(i), names[int(i)], int(r[int(i)])) for i in np.argsort(-r)[:10]]}}
    # L1: members of all cells in the district
    l1n = a.l1_labels
    cat = np.asarray(a.region_category)
    c1 = np.zeros((a.n_l1, vectors.shape[1]), dtype=np.float32)
    for d in range(a.n_l1):
        m = np.isin(owner, np.nonzero(cat == d)[0])
        if m.any():
            v = vectors[m].mean(axis=0); c1[d] = v / (np.linalg.norm(v) + 1e-9)
    nv1 = minilm_numpy.encode(list(l1n))
    r1 = _rank_own(nv1, c1)
    out["l1"] = {"n": a.n_l1, "median_rank": float(np.median(r1)), "top1": float((r1 == 1).mean()),
                 "top1_ci95": C.wilson(int((r1 == 1).sum()), a.n_l1), "top5": float((r1 <= 5).mean()),
                 "chance_median_rank": a.n_l1 / 2,
                 "worst": [(int(i), l1n[int(i)], int(r1[int(i)])) for i in np.argsort(-r1)[:8]]}
    # kinds, when stamped: are form cells the ones whose names rank badly?
    kinds = a.meta.get("region_kinds")
    if kinds:
        kinds = np.array(kinds)
        out["l2"]["by_kind"] = {k: {"n": int((kinds == k).sum()), "median_rank": float(np.median(r[kinds == k])),
                                    "top10": float((r[kinds == k] <= 10).mean())} for k in sorted(set(kinds.tolist()))}
    return out


def _slices():
    """(kind, plain description, texts) for every held-out slice."""
    texts, src, axis, lang = B1.load_panel()
    out = []
    for s in sorted(set(src.tolist())):
        m = src == s
        ax = axis[m][0]
        out.append(("source", f"{s} ({ax.replace('_', ' / ')} dataset)", [t for t, k in zip(texts, m) if k]))
    mt, ml = D.mmlu()
    ml = np.array(ml)
    for s in sorted(set(ml.tolist())):
        out.append(("mmlu", f"MMLU exam questions: {s.replace('_', ' ')}", [t for t, k in zip(mt, ml == s) if k]))
    return out


def make_sheet() -> None:
    products = ("atlas-v3", "atlas-v3-prev")
    slices = _slices()
    # top cell per slice per product, from the cached B1/B2 placements
    tops = {}
    for name in products:
        a = C.atlas(name)
        texts, src, axis, lang = B1.load_panel()
        p = C.place(name, texts, tag="b1panel")
        mt, ml = D.mmlu()
        pm = C.place(name, mt, tag="b2-mmlu")
        i_src = 0
        for kind, desc, _ in slices:
            if kind == "source":
                s = sorted(set(src.tolist()))[i_src]; i_src += 1
                cells = p.nearest[src == s]
            else:
                subj = desc.split(": ", 1)[1].replace(" ", "_")
                cells = pm.nearest[np.array(ml) == subj]
            c = int(np.bincount(cells[cells >= 0]).argmax())
            share = float((cells == c).mean())
            tops[(name, desc)] = (c, a.region_labels[c], share, a.l1_labels[int(a.region_category[c])])
    items, key = [], {}
    rng = random.Random(3)
    ids = list(range(len(slices) * len(products))); rng.shuffle(ids)
    k = 0
    for kind, desc, _ in slices:
        for name in products:
            c, label, share, l1 = tops[(name, desc)]
            items.append({"id": ids[k], "slice": desc, "name": label})
            key[str(ids[k])] = {"product": name, "slice": desc, "kind": kind, "cell": c, "share_in_cell": share, "l1": l1}
            k += 1
    items.sort(key=lambda x: x["id"])
    SHEET.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    KEY.write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(f"wrote {len(items)} blind items to {SHEET}")


def make_mini_cells() -> None:
    name = "atlas-v3"
    reservoir = str(C.RESERVOIRS[name])
    texts, langs, cells, picked, similarity = B7._members(name, reservoir)
    coh = np.load(C.RESULTS / f"b7_coherence_{name}.npy")
    labels = C.atlas(name).region_labels
    kinds = C.atlas(name).meta.get("region_kinds") or [""] * len(labels)
    edges = np.nanpercentile(coh, np.linspace(0, 100, 9))
    rng = random.Random(B7.SEED + 1)
    chosen = []
    for k in range(8):
        pool = [c for c in range(len(coh)) if edges[k] <= coh[c] <= edges[k + 1]]
        chosen += rng.sample(pool, min(MINI_CELLS_PER_OCTILE, len(pool)))
    rng.shuffle(chosen)
    spread_rng = np.random.default_rng(B7.SEED + 1)
    MINI_DIR.mkdir(parents=True, exist_ok=True)
    lines, key = [], {}
    for j, cell in enumerate(chosen):
        tag = f"M{j:02d}"
        key[tag] = {"cell": int(cell), "coherence": float(coh[cell]), "kind": kinds[cell]}
        lines.append(f"=== {tag}\nCURRENT NAME: {labels[cell]}")
        members = picked[cell]
        sample = spread_sample(members, similarity[members], MINI_MEMBERS, spread_rng)
        for i in spread_rng.permutation(sample):
            lines.append("  - " + " ".join(texts[int(i)][:160].split()))
        lines.append("")
    (MINI_DIR / "mini_sheet.txt").write_text("\n".join(lines), encoding="utf-8")
    (MINI_DIR / "mini_key.json").write_text(json.dumps(key, indent=1))
    print(f"wrote {len(chosen)} cells to {MINI_DIR / 'mini_sheet.txt'}")


MIXED_WORDS = ("mixed", "unrelated", "no shared subject", "grab-bag", "miscellaneous", "assorted", "fragments")


def _is_mixed(a) -> np.ndarray:
    """Cells the map itself cannot describe as one subject: the stamped kind
    ``mixed`` where kinds exist, else a name that says so."""
    kinds = a.meta.get("region_kinds")
    if kinds:
        return np.array([k == "mixed" for k in kinds])
    return np.array([any(w in n.lower() for w in MIXED_WORDS) for n in a.region_labels])


def mixed_landing() -> dict:
    """Share of a user's records that land in cells the map calls mixed.

    A record placed in a well-named cell is described; a record placed in
    "Mixed pages with no shared subject" is only located. Measured on the
    held-out panel (B1), MMLU (B2), held-out English web (B4 base) and each
    map's own reservoir sample (B8), from the cached placements.
    """
    out = {}
    texts, src, axis, lang = B1.load_panel()
    mt, ml = D.mmlu()
    c4 = D.v2cache("allenai__c4__en", 60_000, seed=5)[:40_000]
    for name in ("atlas-v3", "atlas-v3-prev"):
        a = C.atlas(name)
        mixed = _is_mixed(a)
        ref = np.asarray(a.region_size, dtype=np.float64)
        r = {"mixed_cells": int(mixed.sum()), "mixed_cells_share": float(mixed.mean()),
             "reference_mass_in_mixed_cells": float(ref[mixed].sum() / ref.sum()),
             "how_identified": "stamped kind" if a.meta.get("region_kinds") else "name words"}
        for label, tx, tag in (("held-out panel", texts, "b1panel"), ("MMLU", mt, "b2-mmlu"), ("C4 English web", c4, "b4base")):
            pl = C.place(name, tx, tag=tag)
            hit = mixed[pl.nearest[pl.placed]]
            r[label] = {"share_in_mixed": float(hit.mean()), "ci95": C.wilson(int(hit.sum()), int(pl.placed.sum()))}
        # per held-out source: which datasets are mostly "located, not described"
        pl = C.place(name, texts, tag="b1panel")
        per = {}
        for s_ in sorted(set(src.tolist())):
            m = (src == s_) & pl.placed
            per[s_] = float(mixed[pl.nearest[m]].mean())
        r["held-out panel"]["sources_over_half_in_mixed"] = sorted([s_ for s_, v in per.items() if v > 0.5])
        r["held-out panel"]["per_source_top8"] = sorted(per.items(), key=lambda kv: -kv[1])[:8]
        out[name] = r
    return out


def score() -> dict:
    key = json.loads(KEY.read_text())
    scores = json.loads(SCORES.read_text())
    per = {}
    by_slice = {}
    for sid, meta in key.items():
        if sid not in scores:
            continue
        v = int(scores[sid])
        per.setdefault(meta["product"], []).append(v)
        by_slice.setdefault(meta["slice"], {})[meta["product"]] = v
    out = {"n_items_scored": sum(len(v) for v in per.values()), "products": {}}
    for name, vals in per.items():
        vals = np.array(vals)
        out["products"][name] = {"n": int(len(vals)), "mean_score_of_2": float(vals.mean()),
                                 "describes_share": float((vals == 2).mean()), "describes_ci95": C.wilson(int((vals == 2).sum()), len(vals)),
                                 "wrong_share": float((vals == 0).mean()), "wrong_ci95": C.wilson(int((vals == 0).sum()), len(vals)),
                                 "at_least_partly_share": float((vals >= 1).mean())}
    both = [(d["atlas-v3"], d["atlas-v3-prev"]) for d in by_slice.values() if len(d) == 2]
    if both:
        a = np.array([x for x, _ in both], dtype=float); b = np.array([y for _, y in both], dtype=float)
        out["paired_v3_minus_prev"] = C.paired_bootstrap_delta(a / 2, b / 2)
        out["paired_v3_minus_prev"]["v3_better"] = int((a > b).sum()); out["paired_v3_minus_prev"]["prev_better"] = int((a < b).sum())
        out["paired_v3_minus_prev"]["tie"] = int((a == b).sum())
    # by kind
    for kind in ("source", "mmlu"):
        for name in per:
            vals = np.array([int(scores[s]) for s, m in key.items() if s in scores and m["product"] == name and m["kind"] == kind])
            out["products"][name][f"describes_share_{kind}"] = float((vals == 2).mean()) if len(vals) else None
    mini = MINI_DIR / "mini_answers.json"
    if mini.exists():
        mk = json.loads((MINI_DIR / "mini_key.json").read_text())
        ans = json.loads(mini.read_text())
        content = Counter(x["content"] for x in ans); verdict = Counter(x["name_verdict"] for x in ans)
        n = len(ans)
        coh = np.array([mk[x["tag"]]["coherence"] for x in ans])
        grab = np.array([x["content"] == "grab-bag" for x in ans])
        out["mini_cells"] = {"judged": n, "content": dict(content), "name_verdict": dict(verdict),
                             "grab_bag_share": float(grab.mean()), "grab_bag_ci95": C.wilson(int(grab.sum()), n),
                             "accurate_share": verdict["accurate"] / n, "accurate_ci95": C.wilson(verdict["accurate"], n),
                             "soup_names": verdict["soup"],
                             "grab_bag_share_coherence_below_0.15": float(grab[coh < 0.15].mean()) if (coh < 0.15).any() else None,
                             "grab_bag_share_coherence_at_least_0.15": float(grab[coh >= 0.15].mean()) if (coh >= 0.15).any() else None,
                             "gates": {"grab_bag": float(grab.mean()) <= B7.GATE_GRAB_BAG, "no_soup": verdict["soup"] == 0,
                                       "accurate": verdict["accurate"] / n >= B7.GATE_ACCURATE}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--no-alignment", action="store_true")
    args = ap.parse_args()
    path = C.RESULTS / "b9_names.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if args.score:
        res["blind"] = score()
        print(json.dumps(res["blind"], indent=1))
    else:
        if not args.no_alignment:
            res["external_alignment"] = {}
            for name in ("atlas-v3", "atlas-v3-prev"):
                print("B9 alignment", name, flush=True)
                res["external_alignment"][name] = external_alignment(name)
                r = res["external_alignment"][name]
                print(f"{name:14s} L2 median rank {r['l2']['median_rank']:.0f} top1 {r['l2']['top1']:.3f} top10 {r['l2']['top10']:.3f} "
                      f"fam top1 {r['l2']['top1_within_family']:.3f} | L1 median {r['l1']['median_rank']:.0f} top1 {r['l1']['top1']:.3f}")
        res["mixed_landing"] = mixed_landing()
        for name, r in res["mixed_landing"].items():
            print(f"{name:14s} mixed cells {r['mixed_cells']} ({r['mixed_cells_share']:.1%}, ref mass {r['reference_mass_in_mixed_cells']:.1%}); "
                  f"records in mixed cells: panel {r['held-out panel']['share_in_mixed']:.1%}, MMLU {r['MMLU']['share_in_mixed']:.1%}, C4 {r['C4 English web']['share_in_mixed']:.1%}")
        make_sheet()
        make_mini_cells()
    C.save("b9_names", res)


if __name__ == "__main__":
    main()
