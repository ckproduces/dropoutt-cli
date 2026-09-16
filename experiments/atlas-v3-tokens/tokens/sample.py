"""Stage 1: a stratified, within-source reservoir sample of the corpus cache.

Every one of the 244 source files is scanned once and k_i of its records are
kept by reservoir sampling (Algorithm L), so each kept record is a uniform
random draw from its source. k_i is proportional to the source's share of the
map's 163,452,464 retained records, with a floor of 1,500. The kept records are
the full cached text (capped at 4,000 characters when fetched), not the
600-character heads the build's own reservoir stores.
"""
from __future__ import annotations

import gzip
import json
import math
import multiprocessing as mp
import random
import sys
import time

from .common import CACHE, SAMPLES, SEED, allocation


def sample_one(args):
    slug, k, seed = args
    path = CACHE / slug / "records.jsonl.gz"
    out = SAMPLES / f"{slug}.jsonl"
    meta_path = SAMPLES / f"{slug}.meta.json"
    if meta_path.exists():
        return slug, json.loads(meta_path.read_text())
    rng = random.Random(f"{seed}:{slug}")
    kept: list[tuple[int, bytes]] = []
    t0 = time.time()
    i = 0
    # Algorithm L (Li 1994): fill the reservoir, then skip ahead geometrically.
    w = math.exp(math.log(rng.random()) / k)
    next_i = k + int(math.log(rng.random()) / math.log(1 - w))
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if i < k:
                kept.append((i, line))
            elif i == next_i:
                kept[rng.randrange(k)] = (i, line)
                w *= math.exp(math.log(rng.random()) / k)
                next_i = i + 1 + int(math.log(rng.random()) / math.log(1 - w))
            i += 1
    kept.sort()
    n_written = 0
    with open(out, "w", encoding="utf-8") as fo:
        for line_no, raw in kept:
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            text = rec.get("text") or ""
            fo.write(json.dumps({"i": line_no, "text": text}, ensure_ascii=False) + "\n")
            n_written += 1
    meta = {"slug": slug, "rows_seen": i, "k": k, "kept": n_written, "seconds": round(time.time() - t0, 1)}
    meta_path.write_text(json.dumps(meta))
    return slug, meta


def main():
    alloc = allocation()
    from .common import population
    pop = population()["sources"]
    # largest first so the pool stays busy at the end
    jobs = sorted(((slug, k, SEED) for slug, k in alloc.items()), key=lambda j: -pop[j[0]]["records"])
    print(f"{len(jobs)} sources, {sum(alloc.values()):,} records to keep", flush=True)
    done = 0; t0 = time.time()
    with mp.Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as pool:
        for slug, meta in pool.imap_unordered(sample_one, jobs):
            done += 1
            print(f"[{done:3d}/{len(jobs)}] {int(time.time()-t0):5d}s rows {meta['rows_seen']:>10,} kept {meta['kept']:>7,} {meta['seconds']:>7.1f}s {slug}", flush=True)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
