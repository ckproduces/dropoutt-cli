"""Print the tables the report quotes, straight from results/*.json.

    python -m bench.summary            # everything
    python -m bench.summary b1 b4      # some sections

Every number in REPORT.md should be reproducible from this printout.
"""
from __future__ import annotations

import json
import sys

from . import common as C


def load(name):
    p = C.RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def ci(v, d=1):
    return "-" if not v or v[0] is None else f"[{100 * v[0]:.{d}f}, {100 * v[1]:.{d}f}]"


def num(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def b1():
    d = load("b1_sources")
    if not d:
        return
    print(f"\n## B1 held-out sources: {d['n_sources']} datasets, {d['n_records']} records, test half\n")
    print("| map | off map | source right (L2) | 95% CI | luck | ceiling | kept share | kind right (L2) | 95% CI | luck | ceiling | kept share | code language right |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")
    for n, p in d["products"].items():
        s, a, c = p["source"], p["axis"], p["code_language"]
        print(f"| {n} | {pct(p['off_atlas_rate'])} | {pct(s['acc_l2'])} | {ci(s['acc_l2_ci95'])} | {pct(s['chance_majority'])} | {pct(s['ceiling_linear'])} | {pct(s['retained_l2'])} "
              f"| {pct(a['acc_l2'])} | {ci(a['acc_l2_ci95'])} | {pct(a['chance_majority'])} | {pct(a['ceiling_linear'])} | {pct(a['retained_l2'])} | {pct(c['acc_l2'])} (luck {pct(c['chance_majority'])}, ceiling {pct(c['ceiling_centroid'])}) |")
    print("\nPaired difference, atlas-v3 minus other map, same records (points; 95% CI; *=CI excludes 0):")
    for n, q in d["paired_vs_main"].items():
        print(f"- vs {n}: " + "; ".join(f"{k} {100 * v['delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]{'*' if v['significant'] else ''}" for k, v in q.items()))
    print("\nL1 (district) reading, source / kind:")
    for n, p in d["products"].items():
        print(f"- {n}: source {pct(p['source']['acc_l1'])} (kept {pct(p['source']['retained_l1'])}), kind {pct(p['axis']['acc_l1'])} (kept {pct(p['axis']['retained_l1'])}); cells used {p['cells_used_l2']} / districts {p['cells_used_l1']}")
    main = d["products"][C.MAIN]["per_source"]
    print("\nWhere held-out datasets land on atlas-v3 (share in main district, its name, districts touched):")
    for s in sorted(main, key=lambda k: -main[k]["top_l1_share"])[:12]:
        q = main[s]
        print(f"- {s}: {pct(q['top_l1_share'], 0)} in \"{q['top_l1_name']}\" ({q['l1_used']} districts; top cell \"{q['top_l2_name']}\")")
    print("...least concentrated:")
    for s in sorted(main, key=lambda k: main[k]["top_l1_share"])[:6]:
        q = main[s]
        print(f"- {s}: {pct(q['top_l1_share'], 0)} in \"{q['top_l1_name']}\" ({q['l1_used']} districts)")


def b2():
    d = load("b2_topics")
    if not d:
        return
    print("\n## B2 exam subjects\n")
    print("| map | set | questions | subjects | off map | subject right (L2) | 95% CI | luck | ceiling | kept share | subject right (L1) | kept share (L1) |")
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for n, p in d["products"].items():
        for k, q in p.items():
            print(f"| {n} | {k} | {q['n']} | {q['n_classes']} | {pct(q['off_atlas_rate'])} | {pct(q['acc_l2'])} | {ci(q['acc_l2_ci95'])} | {pct(q['chance_majority'])} | {pct(q['ceiling_linear'])} | {pct(q['retained_l2'])} | {pct(q['acc_l1'])} | {pct(q['retained_l1'])} |")
    print("\nPaired difference, atlas-v3 minus other map (points; 95% CI; *=CI excludes 0):")
    for n, q in d["paired_vs_main"].items():
        print(f"- vs {n}: " + "; ".join(f"{k} {100 * v['delta']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]{'*' if v['significant'] else ''}" for k, v in q.items()))
    m = d["products"][C.MAIN]["mmlu"]["per_subject"]
    print("\nMMLU subjects on atlas-v3, best and worst concentrated (share in top district, name):")
    order = sorted(m, key=lambda s: -m[s]["top_l1"][0][1])
    for s in order[:6] + order[-6:]:
        t = m[s]["top_l1"][0]
        print(f"- {s}: {pct(t[1], 0)} in \"{t[2]}\"; top cell \"{m[s]['top_l2'][2]}\"")


def b3():
    d = load("b3_language")
    if not d:
        return
    print("\n## B3 language\n")
    print("| map | probe: same topic, other language, same district | 95% CI | different topics, same language, same district | junk probes off map | WMT pair same district | 95% CI | luck | same cell | luck | cell in other's top 5 | off map en / tr | histogram similarity L1 / L2 |")
    print("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|")
    for n, p in d["products"].items():
        pr, w = p["probes"], p["wmt17"]
        print(f"| {n} | {pct(pr['same_topic_l1_agree'])} | {ci(pr.get('same_topic_l1_agree_ci95'))} | {pct(pr['diff_topic_l1_collide'])} | {pct(pr['ood_off_atlas'], 0)} | {pct(w['l1_agree'])} | {ci(w.get('l1_agree_ci95'))} | {pct(w['l1_agree_chance'])} | {pct(w['l2_agree'])} | {pct(w['l2_agree_chance'])} | {pct(w['l2_in_top5_soft'])} | {pct(w['off_atlas_en'])} / {pct(w['off_atlas_tr'])} | {num(w['hist_cosine_l1'], 2)} / {num(w['hist_cosine_l2'], 2)} |")
    print("\n| map | web 9 languages: language-cell NMI L2 / L1 | dominant-language lift L2 | records in >=90% single-language cells | top-10 district overlap between languages | off map | wiki 13 languages: NMI L2 / L1 | lift L2 | single-language share | off map |")
    print("|---|---|---:|---:|---:|---:|---|---:|---:|---:|")
    for n, p in d["products"].items():
        a, b = p["web_langs"], p["wiki_langs"]
        print(f"| {n} | {num(a['nmi_l2'])} / {num(a['nmi_l1'])} | {num(a['dominant_lift_l2'], 2)} | {pct(a['share_records_in_90pct_single_language_cells_l2'])} | {pct(a['mean_top10_l1_overlap_between_languages'])} | {pct(a['off_atlas_rate'])} | {num(b['nmi_l2'])} / {num(b['nmi_l1'])} | {num(b['dominant_lift_l2'], 2)} | {pct(b['share_records_in_90pct_single_language_cells_l2'])} | {pct(b['off_atlas_rate'])} |")
    print("\nOff map by language, Wikipedia panel, atlas-v3:", {k: pct(v["off_atlas"]) for k, v in d["products"][C.MAIN]["wiki_langs"]["per_language"].items()})
    print("Junk probe scores on atlas-v3 (cutoff", d["products"][C.MAIN]["probes"].get("cutoff", "see B4"), "):", dict(zip(d["products"][C.MAIN]["probes"]["ood_names"], d["products"][C.MAIN]["probes"]["ood_scores"])))


def b4():
    d = load("b4_coverage")
    if not d:
        return
    print("\n## B4 coverage\n")
    print("Injection: smallest share found (z>=3 and estimate within 2x), L2 / L1")
    print("| map | " + " | ".join(d["products"][C.MAIN]["injection"].keys()) + " |")
    print("|---|" + "---:|" * len(d["products"][C.MAIN]["injection"]))
    for n, p in d["products"].items():
        print(f"| {n} | " + " | ".join(f"{pct(v['min_detected_share_l2'], 2)} / {pct(v['min_detected_share_l1'], 2)}" for v in p["injection"].values()) + " |")
    print("\nInjection detail, atlas-v3, L2 rows (share, z, estimate/true):")
    for label, v in d["products"][C.MAIN]["injection"].items():
        rows = [r for r in v["rows"] if r["level"] == "l2" and "z" in r]
        print(f"- {label} (home cells {rows[0]['home_cells']}, base mass there {pct(rows[0]['p_home_base'], 2)}): " + "; ".join(f"{pct(r['share'], 2)}: z {r['z']:.1f}, est x{r['ratio_est_true']:.2f}" for r in rows))
    print("\nLadder (effective coverage share / districts reached / off map):")
    for n, p in d["products"].items():
        print(f"- {n}: " + "; ".join(f"{s['step']} {pct(s['effective_share'])} ({s['l1_occupied']} districts, off {pct(s['off_atlas_rate'])})" for s in p["ladder"]["ladder"]))
    print("\nfineweb saturation (n: effective share, cells occupied, off map):")
    for n, p in d["products"].items():
        print(f"- {n}: " + "; ".join(f"{s['n']}: {pct(s['effective_share'])}, {s['regions_occupied']}, off {pct(s['off_atlas_rate'])}" for s in p["ladder"]["fineweb_saturation"]))
    print("\nComparisons (similarity / shared / new, new reversed, L1 histogram cosine):")
    print("| pair | " + " | ".join(d["products"].keys()) + " |")
    print("|---|" + "---|" * len(d["products"]))
    pairs = [x["pair"] for x in d["products"][C.MAIN]["comparisons"]]
    for i, pair in enumerate(pairs):
        print(f"| {pair} | " + " | ".join(f"{p['comparisons'][i]['similarity']:.2f} / {pct(p['comparisons'][i]['shared'], 0)} / {pct(p['comparisons'][i]['new'], 0)} (rev {pct(p['comparisons'][i]['new_reverse'], 0)}; L1 cos {p['comparisons'][i]['l1_hist_cosine']:.2f})" for p in d["products"].values()) + " |")
    print("\nTest-retest (two disjoint halves of one corpus): n -> new L2 (sampling floor) / new L1 / cosine L2")
    for n, p in d["products"].items():
        for slug, rows in p["test_retest"].items():
            if isinstance(rows, list):
                print(f"- {n} {slug}: " + "; ".join(f"{r['n']}: {pct(r['new_between_halves_l2'])} (floor {pct(r.get('new_expected_from_sampling_l2'))}) / {pct(r['new_between_halves_l1'])} / {r['cos_l2']:.3f}" for r in rows))
        print(f"  min n for new < 5%: " + ", ".join(f"{k.replace('_min_n_new_', ' ')}: {v}" for k, v in p["test_retest"].items() if not isinstance(v, list)))
    print("\nOff map:")
    for n, p in d["products"].items():
        o = p["off_atlas"]
        print(f"- {n}: cutoff {o['cutoff_in_use']}, in-family prose percentiles {{{', '.join(f'{k} {v:.3f}' for k, v in o['in_family_similarity_percentiles'].items())}}}, in-family off {pct(o['in_family_off_rate'], 2)}, junk off {pct(o['ood_off_rate'])}; junk by kind: " + ", ".join(f"{k} {pct(v['off_rate_at_cutoff'], 0)} (median {v['median_sim']:.2f})" for k, v in o["ood_per_kind"].items()))


def b5():
    d = load("b5_labels")
    if not d:
        return
    print("\n## B5 names with the atlas's own encoder\n")
    for n, p in d["products"].items():
        al = p["alignment"]
        l2 = al.get("l2", {}); l1 = al.get("l1", {})
        print(f"- {n}: L2 names {l2.get('n')}: own cell median rank {l2.get('median_rank')}, top1 {pct(l2.get('top1'))}, top10 {pct(l2.get('top10'))}, within-district top1 {pct(l2.get('top1_within_family'))} | L1: median {l1.get('median_rank')}, top1 {pct(l1.get('top1'))}, top5 {pct(l1.get('top5'))} | exemplars {p['exemplars']} | hygiene {p['hygiene']}")


def b6():
    d = load("b6_perf_cli")
    if not d:
        return
    t = d["throughput"]
    print("\n## B6 speed\n")
    print(f"- {t['records']} records, mean {t['mean_chars']:.0f} chars: language detection {t['langid_s']:.0f}s, encoding {t['encode_s']:.0f}s ({t['encode_rec_per_s']:.0f}/s), placement {t['assign_s']:.0f}s, coverage {t['coverage_s']:.1f}s; peak RSS {t['peak_rss_mb_after']:.0f} MB; off map {pct(t['off_atlas_rate'])}, cells reached {t['regions_occupied']}, effective {t['effective_regions']:.0f}")
    for k, v in d["cli"].items():
        print(f"- CLI {k}: exit {v['exit']}, {v['wall_s']:.0f}s, files {list(v['written'])}")


def b7():
    d = load("b7_cells")
    if not d:
        return
    print("\n## B7 cell coherence (MiniLM, 20 members per cell)\n")
    import numpy as np
    for n, p in d.items():
        c = p.get("coherence", {})
        k = round(c.get("share_below_0.10", 0) * c.get("cells", 0))
        w = C.wilson(int(k), int(c.get("cells", 0))) if c else [None, None]
        extra = ""
        f = C.RESULTS / f"b7_coherence_{n}.npy"
        if f.exists():
            coh = np.load(f)
            qs = np.nanpercentile(coh, [10, 25, 50, 75, 90])
            extra = f"; percentiles p10 {qs[0]:.3f} p25 {qs[1]:.3f} p50 {qs[2]:.3f} p75 {qs[3]:.3f} p90 {qs[4]:.3f}; cells < 0.05 {int(np.nansum(coh < 0.05))}"
        print(f"- {n}: median {num(c.get('median'))}, cells < 0.10 {pct(c.get('share_below_0.10'), 2)} = {k} cells {ci(w, 2)} (gate 1.5%: {'pass' if c.get('gate_below_0.10') else 'FAIL'}), < 0.15 {pct(c.get('share_below_0.15'))}, reservoir share in cells < 0.10 {pct(c.get('reservoir_share_in_cells_below_0.10'), 2)}{extra}")
    a = C.RESULTS / "b7_coherence_atlas-v3.npy"; b = C.RESULTS / "b7_coherence_atlas-v3-prev.npy"
    if a.exists() and b.exists():
        x, y = np.load(a), np.load(b)
        rng = np.random.default_rng(0)
        d10 = [np.nanmean(rng.choice(x, len(x)) < 0.10) - np.nanmean(rng.choice(y, len(y)) < 0.10) for _ in range(1000)]
        dmed = [np.nanmedian(rng.choice(x, len(x))) - np.nanmedian(rng.choice(y, len(y))) for _ in range(1000)]
        print(f"- new minus previous: cells < 0.10 {100 * (np.nanmean(x < 0.10) - np.nanmean(y < 0.10)):+.2f} points, 95% CI [{100 * np.percentile(d10, 2.5):+.2f}, {100 * np.percentile(d10, 97.5):+.2f}]; median coherence {np.nanmedian(x) - np.nanmedian(y):+.3f} [{np.percentile(dmed, 2.5):+.3f}, {np.percentile(dmed, 97.5):+.3f}]")


def b8():
    d = load("b8_readout")
    if not d:
        return
    print("\n## B8 density readout\n")
    for n, p in d["products"].items():
        s = p["self_calibration"]
        print(f"- {n} own reservoir ({s['n_sampled']} records): off map {pct(s['off_atlas_rate'], 2)}; effective coverage {pct(s['effective_share'])}, cells reached {s['regions_occupied']}; "
              f"raw density within 0.5-2x {pct(s['raw']['share_cells_within_0.5_2x'])} (pure sampling would give {pct(s['expected_multinomial']['share_cells_within_0.5_2x'])}), within 0.8-1.25x {pct(s['raw']['share_cells_within_0.8_1.25x'])} (sampling {pct(s['expected_multinomial']['share_cells_within_0.8_1.25x'])}); "
              f"CLI shrunk within 0.5-2x {pct(s['cli']['share_cells_within_0.5_2x'])}, median {s['cli']['median_density']:.2f}, prior {s['cli']['prior_strength']}; chi2 ratio real/sampling {s['chi2_ratio']:.2f}; detected language = build language {pct(s['detected_vs_build_language_agreement'])}")
        print(f"  off map by axis: " + ", ".join(f"{k} {pct(v['off_rate'], 1)} (n {v['n']})" for k, v in s["off_by_axis"].items()))
        print(f"  off map by language (top): " + ", ".join(f"{k} {pct(v['off_rate'], 1)}" for k, v in list(s["off_by_language_top25"].items())[:15]))
        print(f"  most over-dense: {s['most_overdense_cells'][:4]}")
        print(f"  most under-dense: {s['most_underdense_cells'][:4]}")
        for label, w in p["web"].items():
            print(f"  {label}: off {pct(w['off_atlas_rate'], 2)}, effective {pct(w['effective_share'])}, cells {w['regions_occupied']}, >2x {pct(w['share_cells_above_2x'])}, <0.5x {pct(w['share_cells_below_0.5x'])}; densest {w['densest_cells_cli'][:3]}; top districts {w['l1_share_top5'][:3]}")


def b9():
    d = load("b9_names")
    if not d:
        return
    print("\n## B9 names from outside\n")
    for n, p in d.get("external_alignment", {}).items():
        l2, l1 = p["l2"], p["l1"]
        print(f"- {n}: L2 own-cell median rank {l2['median_rank']:.0f} of {l2['n']} (random {l2['chance_median_rank']:.0f}), top1 {pct(l2['top1'])} {ci(l2['top1_ci95'])}, top10 {pct(l2['top10'])}, top100 {pct(l2['top100'])}, within-district top1 {pct(l2['top1_within_family'])}; "
              f"L1 median {l1['median_rank']:.0f} of {l1['n']}, top1 {pct(l1['top1'])}, top5 {pct(l1['top5'])}")
        if "by_kind" in l2:
            print(f"  by kind: {l2['by_kind']}")
        print(f"  worst L2: {l2['worst'][:5]}")
    for n, r in d.get("mixed_landing", {}).items():
        print(f"- {n} mixed cells: {r['mixed_cells']} ({pct(r['mixed_cells_share'])} of cells, {pct(r['reference_mass_in_mixed_cells'])} of reference mass, by {r['how_identified']}); "
              f"records landing in them: held-out panel {pct(r['held-out panel']['share_in_mixed'])} {ci(r['held-out panel']['ci95'])}, MMLU {pct(r['MMLU']['share_in_mixed'])}, C4 web {pct(r['C4 English web']['share_in_mixed'])}; "
              f"datasets mostly in mixed cells: {r['held-out panel']['sources_over_half_in_mixed']}; top: {[(k, round(v, 2)) for k, v in r['held-out panel']['per_source_top8'][:5]]}")
    b = d.get("blind")
    if b:
        print(f"- blind slices: {b['n_items_scored']} items")
        for n, q in b["products"].items():
            print(f"  {n}: describes {pct(q['describes_share'])} {ci(q['describes_ci95'])}, at least partly {pct(q['at_least_partly_share'])}, wrong {pct(q['wrong_share'])} {ci(q['wrong_ci95'])}; sources {pct(q.get('describes_share_source'))}, MMLU {pct(q.get('describes_share_mmlu'))}")
        if "paired_v3_minus_prev" in b:
            print(f"  paired (v3 minus prev, score/2): {b['paired_v3_minus_prev']}")
        if "mini_cells" in b:
            print(f"  cell read: {b['mini_cells']}")


SECTIONS = {"b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6, "b7": b7, "b8": b8, "b9": b9}

if __name__ == "__main__":
    for k in (sys.argv[1:] or SECTIONS):
        SECTIONS[k]()
