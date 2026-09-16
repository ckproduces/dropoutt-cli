"""B6. Cost, and the command as a researcher runs it.

  a. Throughput at the v3 default sample (500,000 records): encode and
     assign wall time, records per second, peak resident memory.
  b. `dropoutt atlas` end to end on three corpora written as JSONL folders:
     a narrow one (hendrycks geometry + precalculus), a mixed one (Python
     commits, ultrachat, eurlex, orca-math in equal parts) and a multilingual
     web one (fineweb-2, 8 languages). Wall time, artifacts written, and
     the terminal report captured for the write-up.
"""
from __future__ import annotations

import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import data as D


def throughput(name="atlas-v3", n=500_000):
    texts = []
    for slug, k in (("HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", 200_000),
                    ("HuggingFaceFW__fineweb-edu__sample_10BT_000_00000.parquet", 120_000),
                    ("allenai__c4__en", 60_000), ("HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet", 45_000),
                    ("HuggingFaceFW__fineweb-2__data_deu_Latn_train_000_00000.parquet", 25_000),
                    ("HuggingFaceFW__fineweb-2__data_fra_Latn_train_000_00000.parquet", 25_000),
                    ("HuggingFaceFW__fineweb-2__data_spa_Latn_train_000_00000.parquet", 25_000)):
        texts += D.v2cache(slug, k, seed=29)
    texts = texts[:n]
    a = C.atlas(name); C.embedder(name)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t = time.time(); langs = C.detect_langs(texts); t_lang = time.time() - t
    t = time.time(); emb = C.encode(name, texts); t_enc = time.time() - t
    t = time.time(); r = a.assign_all(emb, langs); t_asg = time.time() - t
    t = time.time(); cov = a.coverage(r.best, r.categories, langs, scores=r.score, nearest=r.nearest, embeddings=emb,
                                      lengths=[len(x) for x in texts]); t_cov = time.time() - t
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"product": name, "records": len(texts), "mean_chars": float(np.mean([len(x) for x in texts])),
            "langid_s": t_lang, "encode_s": t_enc, "encode_rec_per_s": len(texts) / t_enc,
            "assign_s": t_asg, "assign_rec_per_s": len(texts) / t_asg, "coverage_s": t_cov,
            "peak_rss_mb_before": rss0 / 2**20, "peak_rss_mb_after": rss1 / 2**20,
            "off_atlas_rate": cov["off_atlas_rate"], "regions_occupied": cov["regions_occupied"], "effective_regions": cov["effective_regions"]}


CORPORA = {
    "narrow-geometry": [("EleutherAI__hendrycks_math__geometry", 870), ("EleutherAI__hendrycks_math__precalculus", 746)],
    "mixed-four-sources": [("bigcode__commitpackft__python_train_0000.parquet", 3000), ("HuggingFaceH4__ultrachat_200k", 3000),
                           ("coastalcph__lex_glue__eurlex", 3000), ("microsoft__orca-math-word-problems-200k", 3000)],
    "multilingual-web": [(slug, 2000) for slug, l in D.LANG_PANEL_WEB if l != "en"],
}


def write_corpora(root: Path):
    out = {}
    for cname, parts in CORPORA.items():
        d = root / cname
        d.mkdir(parents=True, exist_ok=True)
        total = 0
        for slug, n in parts:
            texts = D.v2cache(slug, n, seed=31)
            with open(d / f"{D.short_name(slug)}.jsonl", "w") as fh:
                for t in texts:
                    fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            total += len(texts)
        out[cname] = {"dir": str(d), "records": total, "files": len(parts)}
    return out


def run_cli(corpus_dir: Path, out_dir: Path, model: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "dropoutt", "atlas", str(corpus_dir), "--model", model, "--out", str(out_dir), "--no-open", "--offline"]
    t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(C.REPO), env={**__import__("os").environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "110"})
    dt = time.time() - t
    (out_dir / "terminal.txt").write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr.strip() else ""))
    written = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
    return {"cmd": " ".join(cmd), "exit": proc.returncode, "wall_s": dt, "written": written}


def main():
    res = {}
    print("B6 throughput", flush=True)
    res["throughput"] = throughput()
    root = C.SCRATCH / "cli-corpora"
    res["corpora"] = write_corpora(root)
    res["cli"] = {}
    for cname in CORPORA:
        for model in (["atlas-v3", "atlas-v2"] if cname == "mixed-four-sources" else ["atlas-v3"]):
            print("B6 cli", cname, model, flush=True)
            res["cli"][f"{cname}/{model}"] = run_cli(root / cname, C.RUNS / cname / model, model)
    C.save("b6_perf_cli", res)
    print(json.dumps(res["throughput"], indent=1))
    for k, v in res["cli"].items():
        print(k, "exit", v["exit"], "wall", round(v["wall_s"], 1), "s", v["written"])


if __name__ == "__main__":
    main()
