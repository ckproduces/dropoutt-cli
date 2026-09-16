"""Render REPORT-technical.md (the technical rendering) from results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

from . import common as C

R = C.RESULTS
ORDER = ["atlas-v3", "atlas-v3-prev", "atlas-v2", "atlas-v2-lite"]
SHORT = {"atlas-v3": "v3", "atlas-v3-prev": "v3-prev", "atlas-v2": "v2", "atlas-v2-lite": "v2-lite", "atlas-v3 (global mean)": "v3, no lang. centering"}


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    b1, b2, b3, b4, b5, b6 = (load(n) for n in ("b1_sources", "b2_topics", "b3_language", "b4_coverage", "b5_labels", "b6_perf_cli"))
    jkey = load("judge_key"); scores = load("judge_scores")
    L = []
    P = L.append
    P("# atlas-v3 benchmark report (technical)\n")
    P("Run 2026-09-12 on the release machine (14 cores, 48 GB). Four products placed the same texts through the "
      "exact `dropoutt atlas` path. **v3** is the shipped `atlas-v3.npz` (corpus 4203fc3a, 163.45M reference records, "
      "4,096 hand-named cells in 256 hand-named regions). **v3-prev** is the earlier v3 build (corpus e92d45dd, 95.8M "
      "records) it replaced. **v2** (296 cells / 128 regions) and **v2-lite** (65 cells / 32 regions, 64-d) are the "
      "products it supersedes. Method, data provenance and rerun instructions are in [README.md](README.md); every "
      "number below is in `results/*.json`. A plain-language rendering is [REPORT.md](REPORT.md). Rerun after the "
      "off-atlas cutoff was calibrated to 0.3538 (see section 7); the centroids did not change.\n")

    # ---- headline ----
    P("## 1. Headline\n")
    rows = []
    def row(label, fn, fmt=f):
        rows.append([label] + [fmt(fn(n)) for n in ORDER])
    row("Held-out sources: cell → source accuracy (52-way, chance 2.1%)", lambda n: b1["products"][n]["source"]["acc_l2"], pct)
    row("Held-out sources: cell → axis accuracy (8-way)", lambda n: b1["products"][n]["axis"]["acc_l2"], pct)
    row("MMLU: cell → subject accuracy (57-way, chance 11%)", lambda n: b2["products"][n]["mmlu"]["acc_l2"], pct)
    row("MMLU-Pro: cell → category accuracy (14-way, chance 11%)", lambda n: b2["products"][n]["mmlu_pro"]["acc_l2"], pct)
    row("EXAMS-tr: region → subject accuracy (8-way, chance 24%)", lambda n: b2["products"][n]["exams_tr"]["acc_l1"], pct)
    row("Probe topics: same topic, different language → same region", lambda n: b3["products"][n]["probes"]["same_topic_l1_agree"], pct)
    row("Probe topics: different topic, same language → same region (lower is better)", lambda n: b3["products"][n]["probes"]["diff_topic_l1_collide"], pct)
    row("WMT17 tr-en: sentence and its translation share a region", lambda n: b3["products"][n]["wmt17"]["l1_agree"], pct)
    row("Web in 9 languages: NMI(region, language) at L1 (lower is better)", lambda n: b3["products"][n]["web_langs"]["nmi_l1"])
    row("Injection: smallest share of eurlex detected in a web corpus", lambda n: b4["products"][n]["injection"]["EU regulations (eurlex)"]["min_detected_share_l2"], lambda x: "not detected" if x is None else pct(x, 2))
    row("Injection: smallest share of Turkish instructions detected", lambda n: b4["products"][n]["injection"]["Turkish instructions (InstrucTurca)"]["min_detected_share_l2"], lambda x: "not detected" if x is None else pct(x, 2))
    row("Off-atlas rate on 52 held-out prose/code sources", lambda n: b1["products"][n]["off_atlas_rate"], pct)
    row("Off-atlas rate on 1,400 synthetic machine-format records (higher is better; cutoff cannot fix this)", lambda n: b4["products"][n]["off_atlas"]["ood_off_rate"], pct)
    row("compare(): New between two halves of one corpus at 5,000 records (should be ~0)", lambda n: [x for x in b4["products"][n]["comparisons"] if x["pair"].startswith("same corpus")][0]["new"], pct)
    row("Cell names: own centroid ranks first among all cells", lambda n: b5["products"][n]["alignment"].get("l2", {}).get("top1"), pct)
    row("Region names: own centroid ranks first among regions", lambda n: b5["products"][n]["alignment"].get("l1", {}).get("top1"), pct)
    row("Exemplar texts placed back land in their own cell", lambda n: (b5["products"][n]["exemplars"] or {}).get("own_cell"), pct)
    P(table(["metric"] + [SHORT[n] for n in ORDER], rows))
    P("")
    P("**Reading it.** v3 is the best map on every categorisation and detection measure, and its hand names are the "
      "most aligned with their cells. Two things did not improve with the map and are properties of the code around it, "
      "not of the geometry: the off-atlas cutoff, now calibrated at 0.3538, cannot keep machine-format records off the map "
      "on this encoder; and `compare()` reads a fifth of a corpus as *new* against its own other half, because "
      "4,096 cells are sparsely occupied at ordinary sample sizes. Both are quantified in sections 6 and 7.\n")

    # ---- setup ----
    P("## 2. What was placed\n")
    P("- **52 held-out sources** (1,000 records each, ≥ 80 characters, the CLI's placement floor) from the local v2 build "
      "cache; none is a v3 build input. Axes: code (13, eleven programming languages), math (9), legal/finance (7), "
      "scientific (4), instruction/chat (9, two Turkish), dialogue/QA (3), text-to-SQL (4), web (C4 en, C4 multilingual), "
      "Turkish news (1).")
    P("- **MMLU** test (13,756 questions with options, 57 subjects), **MMLU-Pro** test (12,007, 14 categories), "
      "**EXAMS** crosslingual Turkish (1,561 school-exam questions, 8 subjects).")
    P("- **WMT17 tr-en**: 8,000 sentence pairs with both sides ≥ 80 characters; the repo's **probe file** "
      "(14 topics × en/tr/ar/zh, plus 4 out-of-distribution blobs).")
    P("- **Language panels** (in-build family, same datasets as v3 inputs): fineweb / fineweb-2 in 9 languages (4,000 "
      "each) and Wikipedia in 13 languages (3,000 each).")
    P("- **Coverage**: C4 English (40,000, held out) as the injection base; seven held-out sources injected at "
      "0.25–10%; a nine-step ladder of growing breadth; ten corpus pairs through the CLI's own `compare()`; 1,400 synthetic "
      "machine-format records.\n")
    P("Ceilings are a logistic-regression probe and a nearest-class-mean classifier trained on the same projected "
      "vectors (half the records), which say how much of what the encoder knows the frozen grid keeps. Accuracies are "
      "majority-vote maps from cell to label fitted on one half and scored on the other. AMI is adjusted mutual "
      "information, which corrects for the finite-sample inflation a 4,096-way partition gets for free.\n")

    # ---- B1 ----
    P("## 3. Source categorisation (B1)\n")
    hdr = ["product", "off-atlas", "cells used", "source acc (cell)", "source acc (region)", "source acc, soft top-5", "source AMI (cell)", "ceiling: centroid", "ceiling: linear", "axis acc (cell)", "axis acc (region)", "axis ceiling", "code language acc (cell)", "code lang. ceiling"]
    rows = []
    for n in ORDER:
        r = b1["products"][n]; s, a, c = r["source"], r["axis"], r["code_language"]
        rows.append([SHORT[n], pct(r["off_atlas_rate"]), r["cells_used_l2"], pct(s["acc_l2"]), pct(s["acc_l1"]), pct(s["acc_soft_top5_l2"]), f(s["ami_l2"]), pct(s["ceiling_centroid"]), pct(s["ceiling_linear"]), pct(a["acc_l2"]), pct(a["acc_l1"]), pct(a["ceiling_linear"]), pct(c["acc_l2"]), pct(c["ceiling_centroid"])])
    P(table(hdr, rows)); P("")
    P("Chance is 2.1% for sources and 33% (majority class: instruction) for axes. Reading a v3 cell as a source label "
      "recovers 47% of what a supervised probe gets (30.6 of 64.9 points); reading the soft top-5 recovers 79%. "
      "Programming language is not a subject and the map does not separate it (34% over a 23% majority class), which is "
      "the intended behaviour of a subject map. The v2 AMI is slightly higher than v3's because AMI penalises the "
      "4,096-way partition for the many cells that hold a handful of records; every accuracy and purity measure, which "
      "is what a reader of the report sees, favours v3.\n")
    P("Where the held-out sources landed on v3 (largest region and its share of the source):\n")
    ps = b1["products"]["atlas-v3"]["per_source"]
    rows = [[s, pct(v["off_atlas"]), v["l1_used"], pct(v["top_l1_share"], 0), v["top_l1_name"], v["top_l2_name"]] for s, v in sorted(ps.items(), key=lambda kv: -kv[1]["top_l1_share"])]
    P(table(["source", "off-atlas", "regions reached", "top region share", "top region", "top cell"], rows)); P("")
    P("The narrow sources (competition math, EU law, code, SCOTUS) land 70–90% in one aptly named region. Mixed "
      "instruction sets (alpaca, dolly, ultrachat, oasst1) spread over 190–210 regions with no region above 6%, which is "
      "what a general-purpose instruction mixture looks like on a subject map. Two oddities are explained by the data, "
      "not the map: xlcost Python is code rendered as `NEW_LINE INDENT` token streams, a template rather than prose, "
      "and squad's top region is whichever Wikipedia topic its passages happen to sample most.\n")

    # ---- B2 ----
    P("## 4. Topic categorisation on labelled benchmarks (B2)\n")
    hdr = ["product", "set", "off-atlas", "cell → label acc", "region → label acc", "chance", "ceiling (linear)", "AMI cell", "AMI region", "purity cell"]
    rows = []
    for key, label in (("mmlu", "MMLU (57)"), ("mmlu_pro", "MMLU-Pro (14)"), ("exams_tr", "EXAMS-tr (8)")):
        for n in ORDER:
            q = b2["products"][n][key]
            rows.append([SHORT[n], label, pct(q["off_atlas_rate"]), pct(q["acc_l2"]), pct(q["acc_l1"]), pct(q["chance_majority"]), pct(q["ceiling_linear"]), f(q["ami_l2"]), f(q["ami_l1"]), pct(q["purity_l2"])])
    P(table(hdr, rows)); P("")
    P("On MMLU a v3 cell predicts the subject 3.6× better than chance and reaches 62% of the supervised ceiling; on the "
      "Turkish exam questions the 256 regions do better than the cells (63% against 57%), which is the expected shape "
      "for 1,561 short questions over 4,096 cells. v2 reads the Turkish set at cell level slightly better (62%) while "
      "leaving 26% of its questions off the map; v3 places 97%.\n")
    P("Selected MMLU subjects and the v3 region that holds most of their questions:\n")
    per = b2["products"]["atlas-v3"]["mmlu"]["per_subject"]
    pick = ["astronomy", "virology", "nutrition", "international_law", "high_school_macroeconomics", "college_biology", "computer_security", "marketing", "world_religions", "electrical_engineering", "moral_scenarios", "abstract_algebra", "high_school_us_history", "human_aging", "logical_fallacies"]
    rows = [[s, per[s]["n"], pct(per[s]["top_l1"][0][1], 0), per[s]["top_l1"][0][2], pct(per[s]["top_l1"][1][1], 0), per[s]["top_l1"][1][2]] for s in pick]
    P(table(["MMLU subject", "n", "share", "top region", "share", "second region"], rows)); P("")
    P("`moral_scenarios` (895 items, 78% in *Personal feelings, grief and confessional writing*) is the one large "
      "subject the map reads by register rather than by subject: the questions are first-person vignettes about "
      "everyday wrongdoing. `human_aging` and `logical_fallacies` spread thinly and their top regions are only loosely "
      "related. Full per-subject placements for all three sets are in `results/b2_topics.json`.\n")

    # ---- B3 ----
    P("## 5. Subjects, not languages (B3)\n")
    P("### Designed probes and parallel sentences\n")
    hdr = ["product", "same topic, other language → same region", "→ same cell", "different topic, same language → same region", "OOD blobs off-atlas", "WMT pairs share region", "chance", "share cell", "chance", "cell in partner's soft top-5", "off-atlas en / tr"]
    rows = []
    for n in ORDER + ["atlas-v3 (global mean)"]:
        r = b3["products"][n]; p, w = r["probes"], r["wmt17"]
        rows.append([SHORT[n], pct(p["same_topic_l1_agree"], 0), pct(p["same_topic_l2_agree"], 0), pct(p["diff_topic_l1_collide"], 0), pct(p["ood_off_atlas"], 0), pct(w["l1_agree"]), pct(w["l1_agree_chance"]), pct(w["l2_agree"]), pct(w["l2_agree_chance"]), pct(w["l2_in_top5_soft"]), f"{pct(w['off_atlas_en'])} / {pct(w['off_atlas_tr'])}"])
    P(table(hdr, rows)); P("")
    P("On the 14 designed topics v3 puts the same topic in the same region for 49% of cross-language pairs and never "
      "collides two different topics written in one language; the previous v3 build managed 29% and 22%. Switching v3's "
      "per-language centering off costs 7 points on the probes and raises language NMI on the web panel from 0.09 to "
      "0.17, so the centering is doing real work. On 8,000 WMT17 news sentence pairs 40% land in the same region "
      "(chance 2%) and 62% have the translation's cell inside the sentence's soft top-5; that number is nearly the same "
      "on every product, which says it is a property of the encoder rather than of the map. The four OOD blobs are "
      "placed by every product; see section 8.\n")
    P("### Language clustering vitals\n")
    hdr = ["product", "panel", "cells", "NMI(cell, lang)", "AMI(cell, lang)", "NMI(region, lang)", "dominant-language lift, cell", "lift, region", "records in ≥90%-one-language cells", "top-10 regions shared between languages", "off-atlas"]
    rows = []
    for n in ORDER + ["atlas-v3 (global mean)"]:
        for key, label in (("web_langs", "web ×9"), ("wiki_langs", "wikipedia ×13")):
            r = b3["products"][n][key]
            rows.append([SHORT[n], label, C.atlas(n if n in C.PRODUCTS else "atlas-v3").n_regions, f(r["nmi_l2"]), f(r["ami_l2"]), f(r["nmi_l1"]), f(r["dominant_lift_l2"], 2), f(r["dominant_lift_l1"], 2), pct(r["share_records_in_90pct_single_language_cells_l2"], 0), pct(r["mean_top10_l1_overlap_between_languages"], 0), pct(r["off_atlas_rate"])])
    P(table(hdr, rows)); P("")
    P("These numbers are not comparable across different cell counts (a finer partition always carries more mutual "
      "information with anything), so read them within a product or between v3 and v3-prev, which share k. Against the "
      "previous build, v3 halves the language information in its regions (NMI 0.09 vs 0.17 on web, 0.12 vs 0.22 on "
      "Wikipedia) and halves the share of records sitting in single-language cells (12% vs 24%). Per-language off-atlas "
      "rates on v3 are all under 0.6%; on v2 Chinese web text was 13% off-atlas and Turkish 10%.\n")

    # ---- B4 ----
    P("## 6. Coverage detection (B4)\n")
    P("### Injection sensitivity\n")
    P("A held-out source is mixed into 40,000 C4 English records. Its *home cells* are the cells holding 80% of a "
      "separate 5,000-record sample of it. The excess mass in those cells estimates the injected share; a Poisson "
      "z-score says whether the excess could be noise. Detected means z ≥ 3 and the estimate within a factor of two.\n")
    inj = {n: b4["products"][n]["injection"] for n in ORDER}
    labels = list(inj["atlas-v3"].keys())
    rows = []
    for lab in labels:
        rows.append([lab] + [("not detected" if inj[n][lab]["min_detected_share_l2"] is None else pct(inj[n][lab]["min_detected_share_l2"], 2)) for n in ORDER] + [inj["atlas-v3"][lab]["rows"][0]["home_cells"], pct(inj["atlas-v3"][lab]["rows"][0]["p_home_base"], 2)])
    P(table(["injected source", "v3", "v3-prev", "v2", "v2-lite", "v3 home cells", "base already there"], rows)); P("")
    rows = []
    for lab in labels:
        for q in inj["atlas-v3"][lab]["rows"]:
            if q["level"] == "l2" and "z" in q:
                rows.append([lab, pct(q["share"], 2), q["k"], f(q["z"], 1), pct(q["estimated_share"], 2), f(q["ratio_est_true"], 2), "yes" if q["detected"] else "no"])
    P("v3 detail (fine cells):\n")
    P(table(["source", "true share", "records", "z", "estimated share", "estimate / true", "detected"], rows)); P("")
    P("Sources that own a few cells (EU law: 8 cells, biomedical abstracts: 64) are detected at 0.25% with the share "
      "recovered to within 3%. Sources that are themselves mixtures (Turkish instructions, Turkish web: ~900 home cells "
      "that C4 already occupies at 21–22%) need 2% and are under-estimated by a quarter, because their home cells are "
      "also web cells. v2 needs 2–10× the share for the same sources and v2-lite never detects EU law at all. What the "
      "CLI prints follows: at 10% eurlex the top home cell reads 72× the map's density against 0.3× in the base.\n")
    P("### Coverage ladder\n")
    hdr = ["step", "records", "v3 occupied", "v3 effective", "v3 regions", "v3-prev effective", "v2 effective", "v2-lite effective"]
    rows = []
    lad = {n: b4["products"][n]["ladder"]["ladder"] for n in ORDER}
    for i, q in enumerate(lad["atlas-v3"]):
        rows.append([q["step"], q["records"], f"{q['regions_occupied']}/{q['regions_total']}", f"{q['effective_regions']:.0f} ({pct(q['effective_share'], 0)})", f"{q['l1_occupied']}/{q['l1_total']}"] + [pct(lad[n][i]["effective_share"], 0) for n in ("atlas-v3-prev", "atlas-v2", "atlas-v2-lite")])
    P(table(hdr, rows)); P("")
    P("Effective coverage (sum of min(1, density) over cells) rises with breadth on every product and v3 spreads the "
      "ladder over the widest range (1% to 56%), which is what makes a narrow corpus legible as narrow: geometry alone "
      "reads 1% on v3 and 16% on v2-lite. The one non-monotone step (+ legal, 30.5% → 29.7%) is dilution, since adding "
      "records to a few dense cells lowers every other cell's share.\n")
    sat = {n: b4["products"][n]["ladder"]["fineweb_saturation"] for n in ORDER}
    rows = [[q["n"]] + [f"{sat[n][i]['regions_occupied']} / {pct(sat[n][i]['effective_share'], 0)}" for n in ORDER] for i, q in enumerate(sat["atlas-v3"])]
    P("Reference-like corpus (fineweb English) at growing sample sizes, occupied cells / effective share:\n")
    P(table(["records"] + [SHORT[n] for n in ORDER], rows)); P("")
    P("On v3 the effective share of a web corpus saturates at 64% from 20,000 records; the remaining third of the map is "
      "not English web (code, law, multilingual prose, training prompts), which is the intended reading. The 500,000 "
      "default sample is more than the number needs; 20,000 gets within a point.\n")
    P("### The CLI's own comparison\n")
    rows = []
    for i, q in enumerate(b4["products"]["atlas-v3"]["comparisons"]):
        rows.append([q["pair"]] + [f"{f(b4['products'][n]['comparisons'][i]['similarity'], 2)} / {pct(b4['products'][n]['comparisons'][i]['new'], 0)}" for n in ORDER] + [f(q["l1_hist_cosine"], 2)])
    P(table(["pair (5,000 records each side)", "v3 sim / New", "v3-prev", "v2", "v2-lite", "v3 region-level cosine"], rows)); P("")
    P("Directionally every product gets the pairs right: unrelated corpora (EU law vs case law, biomedical QA vs EU "
      "law) read as 0 similar and 94–98% new on v3; sibling math sets read 0.94 similar; alpaca and its Turkish "
      "translation read 0.66 similar with 81% shared mass, and 0.89 at region level. The absolute *New* figure on v3 is "
      "not trustworthy at this sample size: two halves of the same corpus read 19% new. Test-retest below shows why.\n")
    P("### Test-retest against sample size\n")
    rt = {n: b4["products"][n]["test_retest"] for n in ORDER}
    ds = list(rt["atlas-v3"].keys())
    hdr = ["records per half"] + [f"{SHORT[n]} New / cos(cell) / cos(region)" for n in ORDER]
    for d in ds:
        rows = []
        for i, q in enumerate(rt["atlas-v3"][d]):
            rows.append([q["n"]] + [f"{pct(rt[n][d][i]['new_between_halves_l2'], 0)} / {f(rt[n][d][i]['cos_l2'], 2)} / {f(rt[n][d][i]['cos_l1'], 2)}" for n in ORDER])
        P(f"{d}:\n"); P(table(hdr, rows)); P("")
    P("Two disjoint halves of one corpus should read as the same corpus. On v2 they do from 1,000 records; on v3's "
      "4,096 cells the fine histogram needs 10,000 records per side to get within 8% new and a cosine of 0.83–0.96, "
      "while the 256-region histogram is at 0.95–0.99 from 1,000 records. The map is fine enough that a fingerprint of "
      "a few thousand records is a sparse sample of it, and `compare()` reads sampling gaps as novelty. The region "
      "level is the stable read at those sizes.\n")

    # ---- off-atlas ----
    P("## 7. Off-atlas calibration (B4d)\n")
    hdr = ["product", "cutoff in use", "in-family p1 / p2 / p5 / p50 similarity", "in-family off at cutoff", "machine formats rejected at cutoff", "at in-family p2: in-family off / machine rejected", "at 0.50: in-family off / machine rejected"]
    rows = []
    for n in ORDER:
        o = b4["products"][n]["off_atlas"]; pc = o["in_family_similarity_percentiles"]
        at_p2 = o["cutoff_sweep"][-1]; at_50 = [s for s in o["cutoff_sweep"] if abs(s["cutoff"] - 0.5) < 1e-6][0]
        rows.append([SHORT[n], f(o["cutoff_in_use"], 2), f"{f(pc['p1'])} / {f(pc['p2'])} / {f(pc['p5'])} / {f(pc['p50'])}", pct(o["in_family_off_rate"], 2), pct(o["ood_off_rate"]), f"{pct(at_p2['in_family_off'])} / {pct(at_p2['ood_rejected'])}", f"{pct(at_50['in_family_off'])} / {pct(at_50['ood_rejected'])}"])
    P(table(hdr, rows)); P("")
    P("When the first run was made the v3 artifact carried no `off_atlas_threshold` and the loader's 0.35 default "
      "applied. It has since been calibrated by `tools/calibrate_off_atlas_v3.py` to **0.3538**: the 2nd percentile of "
      "nearest-cell cosine over the build's own language-and-axis-balanced 2,000,000-row calibration draw (per-axis p2 "
      "runs from 0.312 for code and 0.334 for training prompts to 0.421 for scientific text), stamped with an "
      "`off_atlas_calibration` block and guarded by a test. The 26,000-record in-family sample used here is English-web "
      "heavy, which is why its own p2 is 0.425; the balanced draw is deliberately not that. The v2 default of 0.35 sits "
      "*above* v2's own p2 of 0.309, which is why v2 sends 12–18% of ordinary held-out prose off the map. At 0.3538 v3 "
      "places 99.8% of in-family text and 100% of 1,400 base64 / hex / DNA / minified-JS / CSV / random records; at 0.425 "
      "it would place 98% and still reject only 10% of the machine formats, because on this encoder those formats sit "
      "inside the prose similarity band (median 0.43–0.75 against 0.65 for prose). The cutoff cannot be made to reject "
      "them without discarding prose; the surface-share diagnosis in the off-atlas description (whitespace and "
      "non-letter shares) is what separates a template from a subject, and it only runs on records the cutoff already "
      "removed. Remaining recommendation: run the surface test on placed records too, or report machine-format share "
      "separately.\n")

    # ---- B5 ----
    P("## 8. Names (B5)\n")
    hdr = ["product", "cell names", "own cell ranks 1st / top-10 / top-100", "median rank of own cell", "1st among siblings", "region names", "own region ranks 1st / top-5", "exemplars back in own cell / region", "exemplar in soft top-5"]
    rows = []
    for n in ORDER:
        al = b5["products"][n]["alignment"]; ex = b5["products"][n]["exemplars"] or {}
        l2 = al.get("l2", {}); l1 = al.get("l1", {})
        rows.append([SHORT[n], f"{l2.get('n', '–')} ({'hand' if 'curated' in str(l2.get('source')) else 'auto'})", f"{pct(l2.get('top1'), 0)} / {pct(l2.get('top10'), 0)} / {pct(l2.get('top100'), 0)}", f(l2.get("median_rank"), 0), pct(l2.get("top1_within_family"), 0), f"{l1.get('n', '–')} (hand)", f"{pct(l1.get('top1'), 0)} / {pct(l1.get('top5'), 0)}", f"{pct(ex.get('own_cell'), 0)} / {pct(ex.get('own_l1'), 0)}", pct(ex.get("own_cell_in_top5_soft"), 0)])
    P(table(hdr, rows)); P("")
    hy = b5['products']['atlas-v3']['hygiene']
    P("A name embedded with the atlas encoder ranks its own centroid first among 4,096 for 31% of v3 cells and in the "
      "top ten for 65% (median rank 4), against 25% / 57% / 6 on the previous build (v2's 47% is over 296 candidates, an "
      "easier rank, and its captions are the automatic contrastive terms); among a region's siblings 44% of names pick "
      "out their own cell. Region names rank their own centroid first 73% of the time. The stored exemplar texts, "
      "320-character heads of the records nearest each centroid, come back to their own cell 53% of the time and to "
      "their own region 72% (previous build: 30% / 43%), so the map's own witnesses are consistent with it.\n")
    P(f"Hygiene: {hy['l2']['duplicates']} duplicate cell names and {hy['l1']['duplicates']} duplicate region names; "
      f"{hy['l2']['distinct_words']:,} distinct words across the 4,096 cell names. {hy['l2']['names_with_language_word']} cell "
      "names contain a language word: 14 are language-as-subject (Arabic and Russian poetry, Slovenian literary prose, Greek "
      "New Testament manuscripts, English language learning), 3 are place names or false matches (*Czech Republic*, *nail "
      "polish*), and 4 use a nationality adjective where the naming rules ask for the place (*Japanese baseball*, *Japanese "
      "television dramas*, *Slovenian supreme court*, *German place-name lists*). Two region names do the same "
      f"({', '.join(repr(x) for x in hy['l1']['language_word_examples'])}). Those six are the only naming-rule slips found.\n")
    if jkey and scores:
        P("### Blind judge\n")
        P("For each held-out source and each MMLU / MMLU-Pro / EXAMS subject, the name of the region (and, for v3, the "
          "cell) holding most of its records was scored blind by a language model on a 0–2 scale: 2 clearly describes the "
          "data, 1 partially or too generically, 0 unrelated. Items were shuffled and the product hidden.\n")
        from collections import defaultdict
        agg = defaultdict(list)
        for jid, it in enumerate(jkey):
            s = scores.get(str(jid))
            if s is None: continue
            agg[(it["product"], it["level"], it["kind"])].append(s); agg[(it["product"], it["level"], "all")].append(s)
        rows = []
        for (prod, level) in (("atlas-v3", "L1"), ("atlas-v3", "L2"), ("atlas-v2", "L1")):
            r = [f"{SHORT[prod]} {'region' if level == 'L1' else 'cell'} names"]
            for kind in ("source", "mmlu", "mmlu_pro", "exams_tr", "all"):
                v = agg.get((prod, level, kind), [])
                r.append(f"{sum(v) / len(v):.2f} ({pct(sum(1 for x in v if x == 2) / len(v), 0)} clear)" if v else "–")
            rows.append(r)
        P(table(["names", "52 sources", "MMLU 57", "MMLU-Pro 14", "EXAMS-tr 8", "all 131"], rows)); P("")
        P("Mean score and the share of items whose top name clearly describes them. Instruction mixtures cap the "
          "sources column: a corpus that spreads over 200 regions has no single describing region by construction.\n")
    else:
        P("*(Blind judge scores pending; rerun `make_report` once `results/judge_scores.json` exists.)*\n")

    # ---- B6 ----
    P("## 9. Cost and the command (B6)\n")
    t = b6["throughput"]
    rows = [["records", f"{t['records']:,}"], ["mean characters", f"{t['mean_chars']:.0f}"], ["language detection", f"{t['langid_s']:.1f} s"],
            ["encode (tokenise + SIF pool)", f"{t['encode_s']:.1f} s ({t['encode_rec_per_s']:,.0f} rec/s)"], ["assign to 4,096 cells", f"{t['assign_s']:.1f} s ({t['assign_rec_per_s']:,.0f} rec/s)"],
            ["coverage report", f"{t['coverage_s']:.1f} s"], ["peak RSS during the pass", f"{t['peak_rss_mb_after']:,.0f} MB (harness holds all texts and vectors in memory)"],
            ["off-atlas / occupied / effective", f"{pct(t['off_atlas_rate'])} / {t['regions_occupied']} / {t['effective_regions']:.0f}"]]
    P("v3 at its default 500,000-record sample (web text in 5 languages):\n")
    P(table(["stage", "measured"], rows)); P("")
    P("Encoding runs at 8,200 records/s on 1,500-character records; the profile note's 17,500/s was measured on "
      "shorter text. The whole pass is under two minutes, and the 0.9 s coverage step is negligible.\n")
    rows = [[k, f"{b6['corpora'][k.split('/')[0]]['records']:,}", v["exit"], f"{v['wall_s']:.1f} s", ", ".join(f"{a} {b / 1024:.0f} KB" for a, b in sorted(v["written"].items()) if a != "terminal.txt")] for k, v in b6["cli"].items()]
    P("`dropoutt atlas` end to end (`--offline --no-open`, outputs under `runs/`):\n")
    P(table(["corpus / product", "records", "exit", "wall", "artifacts"], rows)); P("")
    P("What the terminal said, trimmed (full text in each run's `terminal.txt`):\n")
    P("- **narrow-geometry / v3**: *Effective coverage 56 of 4096 (56 subregions) (specialised)*; *Mathematics papers, "
      "proofs and theorems 13/18 reach, 92.2% share*; *228 of the map's 256 subject areas never reached*; *31% of your "
      "data sits in a single place on the map*.")
    P("- **mixed-four-sources / v3**: the four sources come back as the four top areas, named: *Commission regulations on "
      "export refunds and prices 21.2%*, *Source code, functions and programming questions 11.0%*, *Money arithmetic word "
      "problems 6.5%*, *Software installation guides and shell setup 5.2%*; ultrachat is the long tail (*Recipes*, "
      "*Routines, sleep and time word problems*). 34 of 12,000 records off the map.")
    P("- **mixed-four-sources / v2**: the same corpus reads *Web servers, SEO and software troubleshooting 12.7%* with "
      "cells captioned `javascript, html, import, class` and `euro, only, back, were`; 1,895 records (15.8%) off the map.")
    P("- **multilingual-web / v3**: eight fineweb-2 languages read as *broad* (2,400 of 4,096 effective) with subject "
      "areas rather than languages on top: *Hotels, rooms and guest reviews*, *Football leagues, cups and match "
      "previews*, *Video games, consoles and gameplay*, *Heads of state and national political news*. One area, "
      "*Religious texts, archive scans and mixed stubs*, is a catch-all whose cells are named as such.\n")

    # ---- findings ----
    P("## 10. Findings\n")
    P("1. **v3 is a better coordinate system than v2 on held-out data.** Source separation +7 points at cell level, "
      "axis separation +3, MMLU subject purity +4, MMLU-Pro +5; every held-out prose source places at ≥ 96% where v2 "
      "dropped 12–26% of several sets off the map.")
    P("2. **v3 is a subject map, and the second build fixed the first's language clustering.** Same-topic cross-language "
      "agreement 49% vs 29%, zero same-language topic collisions vs 22%, half the language information in regions, "
      "12% instead of 24% of web records in single-language cells. Per-language centering accounts for part of that "
      "(7 points on the probes, NMI 0.09 vs 0.17 without it).")
    P("3. **Coverage detection is sharp.** A source with its own cells is found at 0.25% of a 40,000-record web corpus "
      "with its share recovered to within 3%; mixtures need 2%. The reported density in the top home cell moves from "
      "0.3× to 72× at 10% eurlex, so the terminal reads it without statistics.")
    P("4. **The hand names hold up.** 31% of cell names and 73% of region names rank their own centroid first with the "
      "same encoder; exemplars return to their own region 72% of the time; MMLU subjects land under regions a reader "
      "would pick for them (astronomy, virology, nutrition, international law, marketing, computer security).")
    P("5. **Fixed since the first run: the off-atlas cutoff.** The artifact shipped without `off_atlas_threshold`; it now "
      "carries 0.3538 from the balanced calibration draw, with provenance and a test. Machine formats are still placed at "
      "100%, and no cutoff changes that on this encoder; the surface-share test has to run on placed records to catch "
      "templates.")
    P("6. **Defect: `compare()` over 4,096 cells mistakes sampling gaps for novelty.** 19% New between halves of one "
      "corpus at 5,000 records, 8% at 10,000 (v2: 0%). Shared/New should be computed at region level, or corrected for "
      "the sampling-only expectation, before v3 fingerprints are diffed.")
    P("7. **Resolution has a sampling cost the defaults already cover.** Fine-cell histograms stabilise at ~10,000 "
      "records; the 500,000 default is generous, and 20,000 records already saturate effective coverage for web text. "
      "For corpora under a few thousand records the region-level read is the reliable one, and the report could say so.\n")
    P("## 11. Caveats\n")
    P("- The held-out panel comes from the v2 build cache, so it is held out of **v3** but partly in-build for v2 and "
      "v2-lite, which if anything flatters those two.")
    P("- The language panels (fineweb, fineweb-2, Wikipedia) are the same datasets as v3 inputs, though not necessarily "
      "the same records. They are used only for the language vitals and coverage saturation, never for accuracy.")
    P("- Labels for the source panel are dataset of origin, which is a proxy for subject; a map that separates two "
      "math sets is not better for it. The axis-level numbers are the fairer read.")
    P("- Mutual-information measures are not comparable across products with different cell counts; accuracies and "
      "purities are what the tables compare, and AMI is shown for completeness.")
    P("- One blind LLM judge, one pass; treat the judge table as a check on face validity, not as a measurement.")
    (C.EXP / "REPORT-technical.md").write_text("\n".join(L))
    print("wrote", C.EXP / "REPORT-technical.md", len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
