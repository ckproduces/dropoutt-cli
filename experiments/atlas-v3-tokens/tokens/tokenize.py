"""Stage 2: count tokens of every sampled record under every tokenizer.

Writes counts/<slug>.npz per source: utf-8 bytes and characters per record,
and one int32 column per tokenizer with that record's token count (no special
tokens added). Also does the same for the build's 500,000-record reservoir of
600-character heads, as a cross-check on a uniform sample of the retained
population.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from .common import SAMPLES, SCRATCH, allocation
from .panel import load_all

COUNTS = SCRATCH / "counts"
COUNTS.mkdir(exist_ok=True)


def count_texts(toks, texts):
    nbytes = np.array([len(t.encode("utf-8")) for t in texts], dtype=np.int64)
    nchars = np.array([len(t) for t in texts], dtype=np.int64)
    cols = {}
    for key, _label, _users, tok in toks:
        enc = getattr(tok, "encode_batch_fast", None) or tok.encode_batch
        counts = np.zeros(len(texts), dtype=np.int32)
        for start in range(0, len(texts), 20000):
            part = texts[start:start + 20000]
            counts[start:start + len(part)] = [len(e.ids) for e in enc(part, add_special_tokens=False)]
        cols[key] = counts
    return nbytes, nchars, cols


def do_sources(toks):
    alloc = allocation()
    todo = [s for s in alloc if (SAMPLES / f"{s}.meta.json").exists() and not (COUNTS / f"{s}.npz").exists()]
    t0 = time.time()
    for n, slug in enumerate(todo, 1):
        texts = [json.loads(l)["text"] for l in open(SAMPLES / f"{slug}.jsonl", encoding="utf-8")]
        nbytes, nchars, cols = count_texts(toks, texts)
        np.savez(COUNTS / f"{slug}.npz", nbytes=nbytes, nchars=nchars, **cols)
        print(f"[{n}/{len(todo)}] {int(time.time()-t0)}s {len(texts):>7,} recs {nbytes.sum()/1e6:7.1f} MB {slug}", flush=True)
    return len(todo)


def do_reservoir(toks):
    out = COUNTS / "reservoir.npz"
    if out.exists():
        return
    texts, srcs, axes, langs = [], [], [], []
    with open(SCRATCH / "reservoir.jsonl", encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            _i, text, axis, lang, src = json.loads(line)
            texts.append(text); srcs.append(src); axes.append(axis); langs.append(lang)
    t0 = time.time()
    nbytes, nchars, cols = count_texts(toks, texts)
    np.savez(out, nbytes=nbytes, nchars=nchars, src=np.array(srcs), axis=np.array(axes), lang=np.array(langs), **cols)
    print(f"reservoir: {len(texts):,} heads, {nbytes.sum()/1e6:.1f} MB, {int(time.time()-t0)}s", flush=True)


def main():
    toks = load_all()
    print("loaded", len(toks), "tokenizers", flush=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("reservoir", "all"):
        do_reservoir(toks)
    if what in ("sources", "all"):
        do_sources(toks)


if __name__ == "__main__":
    main()
