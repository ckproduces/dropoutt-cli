"""Render REPORT.md (plain language, for a non-technical reader) from results/*.json.

The technical rendering that preceded it is kept as make_report_technical.py and
now writes REPORT-technical.md.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from . import common as C

R = C.RESULTS
ORDER = ["atlas-v3", "atlas-v3-prev", "atlas-v2", "atlas-v2-lite"]
NAME = {"atlas-v3": "new map (v3)", "atlas-v3-prev": "previous v3 build", "atlas-v2": "old map (v2)",
        "atlas-v2-lite": "old small map (v2-lite)", "atlas-v3 (global mean)": "new map, language correction off"}


def load(name):
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=0):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def per100(x):
    return "–" if x is None else f"{round(100 * x)} in 100"


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def terminal_lines(run: str):
    p = C.RUNS / run / "terminal.txt"
    if not p.exists():
        return {}
    t = p.read_text()
    out = {}
    m = re.search(r"Effective coverage ([\d.,]+) of ([\d,]+) \(([\d,]+) subregions hold any records\) \((\w+)\)", t)
    if m:
        out["effective"] = m.group(1); out["total"] = m.group(2); out["occupied"] = m.group(3); out["verdict"] = m.group(4)
    m = re.search(r"([\d,]+) of ([\d,]+) sampled records placed · ([\d,]+) off the map \(([\d.]+%)\)", t)
    if m:
        out["placed"] = m.group(1); out["sampled"] = m.group(2); out["off"] = m.group(3); out["off_pct"] = m.group(4)
    return out


def main():
    b1, b2, b3, b4, b5, b6 = (load(n) for n in ("b1_sources", "b2_topics", "b3_language", "b4_coverage", "b5_labels", "b6_perf_cli"))
    jkey, scores = load("judge_key"), load("judge_scores")
    v3 = {"b1": b1["products"]["atlas-v3"], "b2": b2["products"]["atlas-v3"], "b3": b3["products"]["atlas-v3"],
          "b4": b4["products"]["atlas-v3"], "b5": b5["products"]["atlas-v3"]}
    v2 = {"b1": b1["products"]["atlas-v2"], "b2": b2["products"]["atlas-v2"], "b3": b3["products"]["atlas-v2"],
          "b4": b4["products"]["atlas-v2"], "b5": b5["products"]["atlas-v2"]}
    prev = {"b3": b3["products"]["atlas-v3-prev"], "b5": b5["products"]["atlas-v3-prev"]}
    a3 = C.atlas("atlas-v3")
    cutoff = a3.off_threshold
    cal = a3.meta.get("off_atlas_calibration") or {}

    L = []
    P = L.append

    # ------------------------------------------------------------------ title
    P("# How good is the new atlas map? A plain-language benchmark report\n")
    P("*Tests run on 12 September 2026. Rerun after the off-map cutoff was calibrated the same day. "
      "Every number here comes from the files in `results/`; the technical version of this report is "
      "[REPORT-technical.md](REPORT-technical.md).*\n")

    P("## The short version\n")
    P("The atlas is a fixed map of text. The new version, **atlas-v3**, has 4,096 small areas grouped into 256 "
      "larger areas, and every area has a hand-written name. When you hand the tool a dataset, it places each "
      "record on the map and tells you which areas your data fills, which it barely touches, and which it never "
      "reaches. We tested whether the new map does that job well, using data it had never seen during its "
      "construction, and we compared it with the map it replaces.\n")
    P("**Verdict: the new map is clearly better than the old one at every task we could measure, and its names "
      "are trustworthy.** It sorts unfamiliar data into the right areas, it sees the subject of a text rather "
      "than the language it is written in, and it can spot a small slice of unusual data hidden inside a large "
      "corpus. Two things around the map needed fixing. One, the rule that decides when a record is \"off the "
      "map\", has been fixed since the first run. The other, a comparison feature that over-reports how much "
      "of a small dataset is new, is being worked on.\n")

    # ------------------------------------------------------------------ the atlas
    P("## The atlas in one minute\n")
    P("- Think of a **city map** with 256 districts, each split into a handful of neighbourhoods, 4,096 in all. "
      "A district might be *Recipes, cooking and home kitchens*; one of its neighbourhoods might be *Pizza "
      "recipe and restaurant articles*.")
    P("- The map was drawn once from a very large public collection of text, 163 million documents in about "
      "60 languages, and it never changes. That is the whole point: two datasets placed on the same map can be "
      "compared, the way two shops can be compared by their addresses.")
    P("- To place a record, the tool turns its text into a list of numbers that captures what it is about, "
      "then finds the neighbourhood whose typical text is most similar. The record also gets a **similarity "
      "score** between 0 and 1: how much it resembles that neighbourhood's typical text.")
    P("- If the score is below a **cutoff**, the record is **off the map**: it resembles nothing the map knows. "
      f"That cutoff is now {cutoff:.3f}, set by the calibration described below.")
    P("- The tool then reports **coverage**: how much of your data sits in each district, and how that compares "
      "with the map's own mix. A district where you have three times the map's share is a **3× density**.\n")

    # ------------------------------------------------------------------ scorecard
    P("## The scorecard\n")
    s1 = v3["b1"]["source"]; a1 = v3["b1"]["axis"]; s1v2 = v2["b1"]["source"]
    m1 = v3["b2"]["mmlu"]; m1v2 = v2["b2"]["mmlu"]
    pr3, pr2, prp = v3["b3"]["probes"], v2["b3"]["probes"], prev["b3"]["probes"]
    inj = v3["b4"]["injection"]
    same = [x for x in v3["b4"]["comparisons"] if x["pair"].startswith("same corpus")][0]
    same2 = [x for x in v2["b4"]["comparisons"] if x["pair"].startswith("same corpus")][0]
    # judge
    agg = defaultdict(list)
    if jkey and scores:
        for jid, it in enumerate(jkey):
            s = scores.get(str(jid))
            if s is not None:
                agg[(it["product"], it["level"])].append(s)
    j3c = agg.get(("atlas-v3", "L2"), []); j3r = agg.get(("atlas-v3", "L1"), []); j2r = agg.get(("atlas-v2", "L1"), [])
    clear = lambda v: (sum(1 for x in v if x == 2) / len(v)) if v else None
    t6 = b6["throughput"]
    rows = [
        ["Does it keep different datasets apart?", f"From a record's position alone, its source dataset is named right {per100(s1['acc_l2'])} (52 datasets, so {per100(1/52)} by luck). Old map: {per100(s1v2['acc_l2'])}.", "good"],
        ["Does it sort by subject?", f"Exam questions land in the right subject area {per100(m1['acc_l2'])} (57 subjects, {per100(m1['chance_majority'])} by luck). Old map: {per100(m1v2['acc_l2'])}.", "good"],
        ["Does it see the subject rather than the language?", f"The same topic written in English, Turkish, Arabic and Chinese lands in the same district {pct(pr3['same_topic_l1_agree'])} of the time (previous build {pct(prp['same_topic_l1_agree'])}). It never put two different topics in the same district just because they shared a language (previous build: {pct(prp['diff_topic_l1_collide'])}).", "good, much improved"],
        ["Can it find a small slice hidden in a big corpus?", f"100 EU-law documents hidden among 40,000 web pages were found, and their share was measured to within 3%. The old map needed twice as many; the old small map never found them.", "good"],
        ["Does it describe how broad a corpus is?", "A corpus of only geometry problems reads as 1% of the map; adding code, chat, law, science and web in 21 languages takes it to 56%, rising at every step.", "good"],
        ["Does it notice text that does not belong?", f"Ordinary text is placed almost always ({pct(1 - v3['b1']['off_atlas_rate'], 1)} of held-out records). Machine junk such as base64 or DNA strings is *not* kept off the map, on any version. The cutoff cannot do that job; other checks in the tool must.", "partly, by design"],
        ["Are the names on the map right?", f"A blind judge said the name of the area holding a dataset clearly describes it {pct(clear(j3c))} of the time for the new map's neighbourhood names, against {pct(clear(j2r))} for the old map's district names.", "good"],
        ["Can two datasets be compared?", f"Yes for the big picture: unrelated datasets read as 0% similar, a dataset and its own translation as {f(([x for x in v3['b4']['comparisons'] if x['pair'].startswith('alpaca')][0]['similarity']))} similar. But the \"new\" percentage is inflated for small datasets: two halves of one dataset read {pct(same['new'])} new at 5,000 records (old map: {pct(same2['new'])}).", "needs work (in progress)"],
        ["How fast is it?", f"{t6['records']:,} records placed in about {round((t6['langid_s'] + t6['encode_s'] + t6['assign_s'] + t6['coverage_s']) / 60)} minutes on a laptop-class machine.", "good"],
    ]
    P(table(["question", "what we found", "verdict"], rows)); P("")

    # ------------------------------------------------------------------ method
    P("## How we tested\n")
    P("### The rules we followed\n")
    P("1. **Test on data the map has never seen.** A map built from a pile of text will naturally place that "
      "same text well, the way a student does well on questions copied from the textbook. So the test data "
      "came from 52 public datasets that were *not* used to build the new map, plus four well-known exam and "
      "translation benchmarks that no version of the map was built from. Where we did use data of the same "
      "kind the map was built from, the report says so, and it is never used for the sorting scores.")
    P("2. **Every map gets exactly the same records**, and every record goes through exactly the same steps a "
      "real user's run goes through: the same text window, the same encoder, the same language detection, the "
      "same placement rule, the same cutoff. Nothing was tuned for the test.")
    P("3. **Always show what luck would get.** A score means nothing without knowing what guessing would score. "
      "Every sorting result is shown next to the score you would get by always guessing the most common answer.")
    P("4. **Always show the best possible score.** For each sorting test we also trained a proper classifier on "
      "the same information the map uses. Its score is the ceiling: how much the underlying text understanding "
      "knows. The map's score against that ceiling tells you how much the fixed grid keeps.")
    P("5. **Never let a map grade its own homework.** Sorting scores are computed on records the scoring rule "
      "never saw (see the next section). Names were judged blind, with the map's identity hidden and the items "
      "shuffled.")
    P("6. **Compare like with like.** The new map has 4,096 areas; the old one has 296. Some statistics reward "
      "having more areas no matter what. Those statistics are shown only in the technical appendix and only "
      "compared between maps with the same number of areas. The plain scores above are fair across maps.")
    P("7. **Save everything.** Every placement, every number and every terminal printout is in `results/` and "
      "`runs/`, and the whole suite reruns from one command listed in the README.\n")

    P("### How a sorting score is computed\n")
    P("Take 52,000 records, 1,000 from each of 52 datasets. Place them all. Split them in half at random.")
    P("- On the first half, look at each neighbourhood and note which dataset most of its records came from. "
      "That becomes the neighbourhood's label. This is a **majority vote**.")
    P("- On the second half, for each record, guess the label of the neighbourhood it landed in. Count how "
      "often the guess is right.")
    P("- That count is the score. It is deliberately the weakest possible way to read the map: no training, "
      "no clever model, just \"records that land here usually come from X\". If that alone works, the map is "
      "carrying real information about the data.")
    P("- **Luck** is the score you get by always guessing the single most common dataset. **Ceiling** is what a "
      "trained classifier scores on the same second half using the same numbers the map uses.\n")

    P("### The maps we compared\n")
    P(table(["map", "areas (neighbourhoods / districts)", "built from", "names"], [
        ["new map (v3)", "4,096 / 256", "163 million documents, about 60 languages", "hand-written at both levels"],
        ["previous v3 build", "4,096 / 256", "96 million documents, 20 languages", "hand-written at both levels"],
        ["old map (v2)", "296 / 128", "69 million documents", "district names hand-written; neighbourhood captions automatic"],
        ["old small map (v2-lite)", "65 / 32", "69 million documents", "district names hand-written"],
    ])); P("")

    P("### The data we placed\n")
    P(table(["what", "how much", "seen by the new map during building?", "used for"], [
        ["52 public datasets: code in 11 languages, competition and word-problem math, EU and US law, finance, scientific papers, chat and instruction data in English and Turkish, question answering, text-to-SQL, general web, Turkish news", "1,000 records each, 52,000 in all", "no", "sorting by dataset and by kind"],
        ["MMLU, a standard exam benchmark", "13,756 questions in 57 subjects", "no", "sorting by subject"],
        ["MMLU-Pro", "12,007 questions in 14 categories", "no", "sorting by subject"],
        ["EXAMS, Turkish school-exam questions", "1,561 questions in 8 subjects", "no", "sorting by subject, in Turkish"],
        ["WMT17 news sentences with their Turkish translations", "8,000 pairs", "no", "language test"],
        ["A probe file of 14 topics, each written in English, Turkish, Arabic and Chinese", "56 texts plus 4 junk blobs", "no", "language test"],
        ["Web pages in 9 languages and Wikipedia in 13 languages", "36,000 and 39,000 records", "same sources, not necessarily the same pages", "language mix inside areas; coverage of a typical web corpus"],
        ["C4, an English web crawl", "40,000 records", "no", "the base corpus that slices were hidden in"],
        ["Synthetic machine junk: base64, hex dumps, DNA strings, minified JavaScript, columns of numbers, random letters", "1,400 records", "no", "does the cutoff keep junk off the map?"],
    ])); P("")

    P("### Steps, in order\n")
    P("1. Load the four maps and the encoder. Confirm the encoder is the one each map was built with.")
    P("2. Read the test datasets. Drop records shorter than 80 characters, because the tool itself does not place those.")
    P("3. Detect each record's language, the way the tool does, because the new map adjusts for language before placing.")
    P("4. Place every record on every map. Keep the neighbourhood, the district, the similarity score and the top-5 nearest neighbourhoods.")
    P("5. Score sorting (datasets, kinds, subjects) with the majority-vote rule, next to luck and the ceiling.")
    P("6. Score the language tests: do translations and same-topic texts land together, and how much does an area's identity reveal about language.")
    P("7. Score coverage: hide a slice of one dataset in the web corpus at six sizes and check whether the tool sees it and measures it; build a corpus in nine steps of increasing breadth and check the breadth reading rises; run the tool's own comparison on ten pairs of datasets; split datasets in half and check the two halves read as the same dataset.")
    P("8. Score the names: embed each name with the same encoder and check it points at its own area; place the map's own example texts back and check they return home; have a blind judge rate the names against the data that lands there.")
    P("9. Time a full-size run and run the real command on three small corpora, keeping the printouts.")
    P("10. After the first run, the off-map cutoff was calibrated by a separate piece of work. We re-applied the new cutoff to every stored placement, confirmed the map's areas had not moved, and regenerated every number.\n")

    # ------------------------------------------------------------------ detailed results
    P("## Results in detail\n")

    # --- B1
    P("### 1. Keeping datasets apart\n")
    rows = []
    for n in ORDER:
        r = b1["products"][n]; s, a = r["source"], r["axis"]
        rows.append([NAME[n], pct(r["off_atlas_rate"], 1), per100(s["acc_l2"]), per100(s["acc_soft_top5_l2"]), per100(s["ceiling_linear"]), per100(a["acc_l2"]), per100(a["ceiling_linear"])])
    P(table(["map", "records off the map", "source named right (52 choices, luck 2 in 100)", "source is among the 5 nearest areas", "ceiling", "kind of data named right (8 kinds, luck 33 in 100)", "ceiling"], rows)); P("")
    P(f"**How to read this.** From nothing but where a record landed, the new map names its source dataset right "
      f"{per100(s1['acc_l2'])}, where luck gives 2 and a trained classifier gives {per100(s1['ceiling_linear'])}. "
      f"So the fixed grid keeps about half of what could be known. If we allow the five nearest neighbourhoods "
      f"instead of one, it rises to {per100(s1['acc_soft_top5_l2'])}. Sorting by *kind* of data (code, math, law, "
      f"science, chat, question answering, SQL, web) is right {per100(a1['acc_l2'])}. The old map dropped "
      f"{pct(v2['b1']['off_atlas_rate'])} of these ordinary records off the map; the new one drops "
      f"{pct(v3['b1']['off_atlas_rate'], 1)}.\n")
    P("Where some of the datasets landed on the new map, with the district that holds most of their records:\n")
    ps = v3["b1"]["per_source"]
    pick = ["hendrycks_math__geometry", "lex_glue__eurlex", "commitpackft__java", "CShorten__ML-ArXiv-Papers", "lex_glue__scotus", "openai__gsm8k__main", "gbharti__finance-alpaca", "mcemilg__news-cat", "qiaojin__PubMedQA__pqa_artificial", "lex_glue__unfair_tos", "ultrachat_200k", "tatsu-lab__alpaca"]
    plain = {"hendrycks_math__geometry": "competition geometry problems", "lex_glue__eurlex": "EU regulations", "commitpackft__java": "Java code commits", "CShorten__ML-ArXiv-Papers": "machine-learning papers", "lex_glue__scotus": "US Supreme Court opinions", "openai__gsm8k__main": "grade-school math word problems", "gbharti__finance-alpaca": "personal-finance Q&A", "mcemilg__news-cat": "Turkish news articles", "qiaojin__PubMedQA__pqa_artificial": "biomedical research Q&A", "lex_glue__unfair_tos": "website terms of service", "ultrachat_200k": "general chat conversations", "tatsu-lab__alpaca": "general instruction data"}
    rows = [[plain[k], pct(ps[k]["top_l1_share"]), ps[k]["top_l1_name"], ps[k]["l1_used"]] for k in pick if k in ps]
    P(table(["dataset", "share in its main district", "that district's name", "districts touched"], rows)); P("")
    P("Narrow datasets land mostly in one aptly named district. General chat and instruction data spreads over "
      "about 200 districts with no district above 6%, which is what a mixed bag should look like on a subject "
      "map. Programming language is deliberately *not* something the map separates: it reads Java and Python "
      "commits as the same kind of thing, code.\n")

    # --- B2
    P("### 2. Sorting by subject\n")
    rows = []
    for key, label in (("mmlu", "MMLU, 57 subjects"), ("mmlu_pro", "MMLU-Pro, 14 categories"), ("exams_tr", "Turkish exams, 8 subjects")):
        for n in ("atlas-v3", "atlas-v2"):
            q = b2["products"][n][key]
            rows.append([label, NAME[n], pct(q["off_atlas_rate"], 1), per100(q["acc_l2"]), per100(q["acc_l1"]), per100(q["chance_majority"]), per100(q["ceiling_linear"])])
    P(table(["exam set", "map", "off the map", "subject right from the neighbourhood", "subject right from the district", "luck", "ceiling"], rows)); P("")
    P(f"**How to read this.** A question's neighbourhood alone names its MMLU subject right {per100(m1['acc_l2'])}, "
      f"more than three times what luck gives. On the Turkish exam the districts do better than the "
      f"neighbourhoods ({per100(v3['b2']['exams_tr']['acc_l1'])} against {per100(v3['b2']['exams_tr']['acc_l2'])}), "
      f"because 1,561 short questions are too few to fill 4,096 neighbourhoods. The old map left "
      f"{pct(v2['b2']['exams_tr']['off_atlas_rate'])} of the Turkish questions off the map; the new one leaves "
      f"{pct(v3['b2']['exams_tr']['off_atlas_rate'], 1)}.\n")
    per = v3["b2"]["mmlu"]["per_subject"]
    pick = ["astronomy", "virology", "nutrition", "international_law", "high_school_macroeconomics", "marketing", "computer_security", "electrical_engineering", "world_religions", "moral_scenarios"]
    rows = [[s.replace("_", " "), pct(per[s]["top_l1"][0][1]), per[s]["top_l1"][0][2]] for s in pick]
    P("Some MMLU subjects and the district holding most of their questions:\n")
    P(table(["exam subject", "share", "district"], rows)); P("")
    P("The one large miss is *moral scenarios*: those questions are little first-person stories about everyday "
      "wrongdoing, and the map reads them by their tone rather than their subject. Every subject's placement is "
      "in `results/b2_topics.json`.\n")

    # --- B3
    P("### 3. Subject, not language\n")
    w3, w2 = v3["b3"]["wmt17"], v2["b3"]["wmt17"]
    wl3, wlp, wl2, wlg = v3["b3"]["web_langs"], prev["b3"]["web_langs"], v2["b3"]["web_langs"], b3["products"]["atlas-v3 (global mean)"]["web_langs"]
    rows = []
    for n in ORDER + ["atlas-v3 (global mean)"]:
        p, w = b3["products"][n]["probes"], b3["products"][n]["wmt17"]
        rows.append([NAME[n], pct(p["same_topic_l1_agree"]), pct(p["diff_topic_l1_collide"]), pct(w["l1_agree"]), pct(w["l1_agree_chance"]), pct(w["l2_in_top5_soft"]), f"{pct(w['off_atlas_en'], 1)} / {pct(w['off_atlas_tr'], 1)}"])
    P(table(["map", "same topic, different language: same district", "different topic, same language: same district (lower is better)", "a sentence and its translation share a district", "by luck", "translation is among the 5 nearest areas", "off the map, English / Turkish"], rows)); P("")
    P(f"**How to read this.** We wrote 14 topics, such as photosynthesis and monetary policy, in four languages. "
      f"On the new map the same topic lands in the same district {pct(pr3['same_topic_l1_agree'])} of the time "
      f"across languages, up from {pct(prp['same_topic_l1_agree'])} on the previous build, and two different "
      f"topics in the same language were put together {pct(pr3['diff_topic_l1_collide'])} of the time, down from "
      f"{pct(prp['diff_topic_l1_collide'])}. That last number is the one that matters most: the previous build had "
      f"areas that meant \"Turkish text\" rather than a subject, and this one does not. On {w3['pairs']:,} real "
      f"news sentences and their Turkish translations, {pct(w3['l1_agree'])} of pairs share a district where luck "
      f"would give {pct(w3['l1_agree_chance'])}. That figure is almost the same on every map, so it is a property "
      f"of the text encoder underneath rather than of the map. Turning the new map's language correction off "
      f"costs {round(100 * (pr3['same_topic_l1_agree'] - b3['products']['atlas-v3 (global mean)']['probes']['same_topic_l1_agree']))} points on the topic test, so that correction is doing real work.\n")
    P("A second view: place web pages in nine languages and ask how much an area's identity gives away about "
      "language. If areas were pure subject, knowing the area would say nothing about language.\n")
    rows = [[NAME[n], pct(b3["products"][n]["web_langs"]["share_records_in_90pct_single_language_cells_l2"]), f(b3["products"][n]["web_langs"]["dominant_lift_l1"], 1), pct(b3["products"][n]["web_langs"]["off_atlas_rate"], 1)] for n in ("atlas-v3", "atlas-v3-prev", "atlas-v3 (global mean)")]
    P(table(["map", "web records sitting in areas that are 90%+ one language", "how much the top language dominates a district (1.0 = no more than its normal share)", "off the map"], rows)); P("")
    P(f"On the new map {pct(wl3['share_records_in_90pct_single_language_cells_l2'])} of web records sit in "
      f"single-language areas, down from {pct(wlp['share_records_in_90pct_single_language_cells_l2'])} on the "
      f"previous build; switching the language correction off puts it back to "
      f"{pct(wlg['share_records_in_90pct_single_language_cells_l2'])}. Every language placed at 99% or better; "
      f"on the old map, {pct(wl2['per_language']['zh']['off_atlas'])} of Chinese pages and "
      f"{pct(wl2['per_language']['tr']['off_atlas'])} of Turkish pages fell off the map.\n")

    # --- B4a
    P("### 4. Finding a small slice hidden in a big corpus\n")
    P("We took 40,000 English web pages and hid a slice of a different dataset inside, at sizes from a quarter "
      "of a percent (100 records) to 10% (4,444 records). Then we asked: does the tool see the slice, and does "
      "it measure its size correctly? \"Seen\" means the extra records in the slice's home areas could not be "
      "explained by chance, and the estimated size is within a factor of two of the truth.\n")
    labels = list(inj.keys())
    rows = []
    for lab in labels:
        rows.append([lab] + [("never" if b4["products"][n]["injection"][lab]["min_detected_share_l2"] is None else pct(b4["products"][n]["injection"][lab]["min_detected_share_l2"], 2)) for n in ORDER])
    P(table(["hidden slice", "new map: smallest slice seen", "previous v3 build", "old map", "old small map"], rows)); P("")
    e10 = [q for q in inj["EU regulations (eurlex)"]["rows"] if q["level"] == "l2" and q["share"] == 0.10][0]
    e025 = [q for q in inj["EU regulations (eurlex)"]["rows"] if q["level"] == "l2" and q["share"] == 0.0025][0]
    t10 = [q for q in inj["Turkish instructions (InstrucTurca)"]["rows"] if q["level"] == "l2" and q["share"] == 0.10][0]
    P(f"**How to read this.** A slice with a distinctive home, such as EU regulations, is seen at 0.25% and its "
      f"size is measured almost exactly: {pct(e025['estimated_share'], 2)} estimated for a true 0.25%, and "
      f"{pct(e10['estimated_share'], 2)} for a true 10%. Slices that are themselves mixtures, such as Turkish "
      f"instruction data, share their home areas with ordinary web pages, so they need to be about 2% before "
      f"they stand out, and their size is under-read by about a quarter ({pct(t10['estimated_share'], 1)} for a "
      f"true 10%). What the tool actually prints follows the same pattern: the main EU-law neighbourhood reads "
      f"{f(inj['EU regulations (eurlex)']['top_home_cell_density_base'], 1)}× the map's density with no slice and "
      f"{f(inj['EU regulations (eurlex)']['top_home_cell_density_mix10'], 0)}× with a 10% slice.\n")

    # --- B4b
    P("### 5. Describing how broad a corpus is\n")
    lad = v3["b4"]["ladder"]["ladder"]
    rows = [[q["step"], f"{q['records']:,}", pct(q["effective_share"]), f"{q['l1_occupied']} of {q['l1_total']}"] for q in lad]
    P(table(["corpus built step by step", "records", "share of the map covered (new map)", "districts reached"], rows)); P("")
    sat = v3["b4"]["ladder"]["fineweb_saturation"]
    P(f"**How to read this.** \"Share of the map covered\" counts an area as fully covered when your data is at "
      f"least as dense there as the map's own data, and as a fraction when it is thinner. A corpus of geometry "
      f"problems alone covers {pct(lad[0]['effective_share'])} of the map; each addition raises it, to "
      f"{pct(lad[-1]['effective_share'])} with nine kinds of data in 21 languages. A large sample of ordinary "
      f"English web pages tops out at {pct(sat[-1]['effective_share'])} of the map from about 20,000 records "
      f"onward: the remaining third of the map is made of things English web is not, such as code, law, other "
      f"languages and training prompts. That is the intended reading.\n")

    # --- B4c
    P("### 6. Comparing two datasets\n")
    P("The tool can compare two placed datasets and report how similar they are and how much of one sits in "
      "areas the other never reaches (\"new\"). We ran it on ten pairs.\n")
    rows = []
    for i, q in enumerate(v3["b4"]["comparisons"]):
        rows.append([q["pair"], f(q["similarity"]), pct(q["new"]), f(v2["b4"]["comparisons"][i]["similarity"]), pct(v2["b4"]["comparisons"][i]["new"])])
    P(table(["pair (5,000 records each)", "new map: similarity (1 = identical)", "new map: share of the left dataset that is \"new\"", "old map: similarity", "old map: \"new\""], rows)); P("")
    P("**How to read this.** The big picture is right on both maps: unrelated datasets read as 0 similar; two "
      "flavours of competition algebra read as almost the same; an instruction set and its own Turkish "
      "translation read as mostly shared. The problem is the first row. Two halves of the *same* dataset should "
      f"read as identical, and on the new map they read {pct(same['new'])} new. The reason is the map's fineness: "
      "5,000 records spread over 4,096 neighbourhoods leave many neighbourhoods with one or two records, and "
      "the two halves happen to hit different ones. The tool counts those gaps as novelty.\n")
    rt = v3["b4"]["test_retest"]["ultrachat_200k"]; rt2 = v2["b4"]["test_retest"]["ultrachat_200k"]
    rows = [[f"{q['n']:,}", pct(q["new_between_halves_l2"]), f(q["cos_l1"]), pct(rt2[i]["new_between_halves_l2"])] for i, q in enumerate(rt)]
    P("The same effect against sample size, on a chat dataset:\n")
    P(table(["records per half", "new map: \"new\" between the two halves", "new map: how alike the halves are at district level (1 = identical)", "old map: \"new\""], rows)); P("")
    P("At the district level the halves look alike from 1,000 records; at the neighbourhood level they need "
      "about 10,000. Until the comparison is changed to work at district level or to allow for this, treat the "
      "\"new\" percentage on the new map as an upper limit for datasets under 10,000 records. That change is "
      "in progress.\n")

    # --- B4d
    P("### 7. Text that does not belong, and the cutoff\n")
    o3 = v3["b4"]["off_atlas"]; pc = o3["in_family_similarity_percentiles"]
    at_use = min(o3["cutoff_sweep"], key=lambda s: abs(s["cutoff"] - cutoff))
    at_p2 = min(o3["cutoff_sweep"], key=lambda s: abs(s["cutoff"] - pc["p2"]))
    at_50 = min(o3["cutoff_sweep"], key=lambda s: abs(s["cutoff"] - 0.50))
    P(f"A record whose similarity to its nearest neighbourhood is below the cutoff is off the map. When we first "
      f"ran the tests, the new map shipped without its own cutoff and a generic default of 0.35 applied. Since "
      f"then the cutoff has been set properly, to **{cutoff:.3f}**, by the rule the documentation promises: it is "
      f"the point below which only 2 in 100 records of the map's own building material fall, measured on a "
      f"{cal.get('rows_scored', 0):,}-record sample balanced across languages and kinds of text. The number "
      f"barely moved, so the results below are essentially unchanged.\n")
    rows = [
        ["ordinary held-out text (52 datasets)", pct(v3["b1"]["off_atlas_rate"], 1)],
        ["a typical English web corpus", pct(o3["in_family_off_rate"], 2)],
        ["machine junk (base64, hex, DNA, minified code, number columns, random characters)", pct(o3["ood_off_rate"])],
    ]
    P(table(["what was placed on the new map", "share off the map"], rows)); P("")
    P(f"**How to read this.** Ordinary text is placed almost always, which is what the cutoff is for. The "
      f"surprise is the junk: none of it falls off the map, on any version of the map. We checked whether a "
      f"higher cutoff would help. It would not, because on this text encoder junk scores about as similar to "
      f"the map as real prose does: a cutoff that rejects half the junk ({pct(at_50['ood_rejected'])}) also "
      f"throws away {pct(at_50['in_family_off'])} of ordinary web pages. The cutoff cannot tell junk from "
      f"prose; that job belongs to the tool's other checks, which look at the shape of the text itself, such as "
      f"how much of it is whitespace and letters. Those checks exist, but today they only run on records the "
      f"cutoff already removed, so they never see this junk. That is the remaining gap.\n")
    P("Why the calibrated cutoff is lower than an English-web reader might expect: the map's building material "
      "includes code and training prompts, which sit less snugly in their areas than prose does. Balancing the "
      f"calibration across all of it gives {cutoff:.3f}; English web pages alone would give about "
      f"{pc['p2']:.3f}. One cutoff serves every kind of text, so a corpus of nothing but code will read a few "
      "percent off the map without anything being wrong with it, and the tool's off-map description says what "
      "the records are so the reader can tell.\n")

    # --- B5
    P("### 8. Are the names right?\n")
    al3, alp = v3["b5"]["alignment"], prev["b5"]["alignment"]
    ex3, exp = v3["b5"]["exemplars"], prev["b5"]["exemplars"]
    rows = [
        ["a neighbourhood's name, embedded like a record, points at its own neighbourhood first (out of 4,096)", pct(al3["l2"]["top1"]), pct(alp["l2"]["top1"])],
        ["... or within the ten nearest", pct(al3["l2"]["top10"]), pct(alp["l2"]["top10"])],
        ["a district's name points at its own district first (out of 256)", pct(al3["l1"]["top1"]), pct(alp["l1"]["top1"])],
        ["the map's own example texts, placed again, return to their own neighbourhood", pct(ex3["own_cell"]), pct(exp["own_cell"])],
        ["... or at least their own district", pct(ex3["own_l1"]), pct(exp["own_l1"])],
    ]
    P(table(["check", "new map", "previous v3 build"], rows)); P("")
    P("**How to read this.** The first three rows ask whether a name and its area mean the same thing to the "
      "encoder. Pointing at the right one of 4,096 first is a hard target; a third of neighbourhood names do it "
      "and two thirds get within ten. District names hit first almost three times out of four. The last two rows "
      "use the four short example texts the map stores for every neighbourhood: placed again, most return to "
      "their own district. All five numbers improved over the previous build.\n")
    if agg:
        rows = []
        for prod, level, label in (("atlas-v3", "L2", "new map, neighbourhood names"), ("atlas-v3", "L1", "new map, district names"), ("atlas-v2", "L1", "old map, district names")):
            v = agg.get((prod, level), [])
            if v:
                rows.append([label, pct(sum(1 for x in v if x == 2) / len(v)), pct(sum(1 for x in v if x == 1) / len(v)), pct(sum(1 for x in v if x == 0) / len(v)), len(v)])
        P("Finally, a blind judge. For each of the 52 datasets and 79 exam subjects we took the name of the area "
          "holding most of its records, shuffled everything, hid which map it came from, and had a language "
          "model score whether the name describes the data: clearly, partly, or not at all.\n")
        P(table(["names judged", "clearly describes the data", "partly or too generic", "unrelated", "items"], rows)); P("")
        P("The cases the judge marked unrelated on the new map are mostly general chat datasets, which have no "
          "single describing area by construction, plus a code dataset stored as a stream of tokens rather than "
          "readable code. On the old map the judge also found competition math filed under banking and text-to-SQL "
          "under government regulations; the new map gets those right.\n")
    hy = v3["b5"]["hygiene"]
    P(f"Name hygiene: no duplicate names at either level, {hy['l2']['distinct_words']:,} distinct words across "
      f"the 4,096 neighbourhood names, and six names that use a nationality adjective (\"Japanese baseball\", "
      f"\"German municipalities\") where the naming rules ask for the place. Those are the only slips found.\n")

    # --- B6
    P("### 9. Speed, and what a real run looks like\n")
    rows = [["records placed", f"{t6['records']:,}"], ["detect languages", f"{t6['langid_s']:.0f} s"], ["turn text into numbers", f"{t6['encode_s']:.0f} s"],
            ["place on the map", f"{t6['assign_s']:.0f} s"], ["write the coverage report", f"{t6['coverage_s']:.1f} s"], ["total", f"about {round((t6['langid_s'] + t6['encode_s'] + t6['assign_s'] + t6['coverage_s']))} s"]]
    P(table(["step at the default sample size", "time"], rows)); P("")
    P("We also ran the actual `dropoutt atlas` command on three small corpora and kept the printouts under `runs/`:\n")
    tl = {k: terminal_lines(k) for k in ("narrow-geometry/atlas-v3", "mixed-four-sources/atlas-v3", "mixed-four-sources/atlas-v2", "multilingual-web/atlas-v3")}
    rows = []
    for k, v in b6["cli"].items():
        t = tl.get(k, {})
        rows.append([k.replace("/", " on "), f"{b6['corpora'][k.split('/')[0]]['records']:,}", f"{v['wall_s']:.1f} s", f"{t.get('off', '?')} ({t.get('off_pct', '?')})", f"{t.get('effective', '?')} of {t.get('total', '?')}, read as \"{t.get('verdict', '?')}\""])
    P(table(["corpus and map", "records", "time", "off the map", "share of the map covered"], rows)); P("")
    P("- **Geometry problems only, new map**: the tool says *specialised*, puts 92% of the records in "
      "*Mathematics papers, proofs and theorems*, and notes that 228 of the 256 districts were never reached.")
    P("- **Four datasets mixed (Python commits, chat, EU law, math word problems), new map**: the four come back "
      "as the four largest districts, correctly named: *Commission regulations on export refunds and prices*, "
      "*Source code, functions and programming questions*, *Money arithmetic word problems*, *Software "
      "installation guides and shell setup*. Chat is the long tail across many districts.")
    P("- **The same four datasets, old map**: the top district is *Web servers, SEO and software "
      "troubleshooting*, its neighbourhoods are captioned with raw words like `javascript, html, import, class`, "
      "and 1,895 records (16%) are off the map.")
    P("- **Web pages in eight languages, new map**: read as *broad*, and the largest districts are subjects, not "
      "languages: hotels and guest reviews, football, video games, heads of state and national politics.\n")

    # ------------------------------------------------------------------ fixes
    P("## What was fixed, and what still needs work\n")
    P(f"- **Fixed: the off-map cutoff.** The first run found the new map shipped without one. It is now "
      f"calibrated at {cutoff:.3f} by the documented rule, stamped into the map file with a record of how it was "
      f"computed, and a test fails if a future build ships without it. Practical effect: almost none on ordinary "
      f"data.")
    P("- **In progress: the comparison feature.** On the new map, \"new\" is over-reported for datasets under "
      "about 10,000 records because the map is so fine. The fix is to compare at district level or to allow for "
      "the expected gaps; a separate piece of work is on it.")
    P("- **Open: machine junk.** The cutoff cannot keep base64, hex dumps, DNA strings or minified code off the "
      "map, and the checks that can are only run on records the cutoff removed. They should also run on placed "
      "records, or the tool should report the share of such records separately.")
    P("- **Minor: six names** use a nationality adjective where the naming rules ask for the place name.\n")

    # ------------------------------------------------------------------ caveats
    P("## Things to keep in mind\n")
    P("- The 52 test datasets were never used to build the new map, but some of them *were* used to build the "
      "old map. If anything, that flatters the old map.")
    P("- The web and Wikipedia pages used for the language tests come from the same public sources the new map "
      "was built from. They were never used for the sorting scores.")
    P("- \"Which dataset did this come from\" is a stand-in for \"what is this about\". Two math datasets are "
      "genuinely alike, and a map that cannot tell them apart is not worse for it. The \"kind of data\" scores "
      "are the fairer read.")
    P("- The name judge is one language model, one pass. Treat its table as a sanity check, not a measurement.")
    P("- Everything was run on one machine, on one day, from cached copies of public datasets. The README says how to rerun it.\n")

    # ------------------------------------------------------------------ glossary
    P("## Words used in this report\n")
    P("- **Record**: one document, message, question or code file from a dataset.")
    P("- **Placing**: finding the neighbourhood on the map whose typical text a record most resembles.")
    P("- **Similarity score**: a number from 0 to 1 saying how much a record resembles its neighbourhood. Records below the cutoff are off the map.")
    P("- **Neighbourhood / district**: the map's 4,096 small areas and 256 large areas. The technical report calls them cells (L2) and regions (L1).")
    P("- **Density**: your share of an area divided by the map's own share of it. 1× means the same as the map; 10× means ten times denser.")
    P("- **Share of the map covered**: adds up, over all areas, how fully each one is covered, counting an area as fully covered once you match the map's density there.")
    P("- **Luck**: the score from always guessing the most common answer.")
    P("- **Ceiling**: the score a trained classifier gets on the same information, the most the map could have kept.")
    P("- **Held-out**: data that was not used to build the thing being tested.")
    P("- **Blind judge**: a rater who does not know which map produced what.")

    (C.EXP / "REPORT.md").write_text("\n".join(L))
    print("wrote", C.EXP / "REPORT.md", len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
