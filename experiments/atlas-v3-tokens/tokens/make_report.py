"""Render REPORT.md (plain language) and a technical appendix from results/estimate.json."""
from __future__ import annotations

import json

from .common import RESULTS, EXP, manifest, population
from .panel import PANEL

AXIS_LABEL = {"web": "web pages", "training": "training and instruction data", "encyclopedic": "encyclopedias (Wikipedia, Wikisource)",
              "legal_government": "law and government", "scientific": "scientific papers", "books": "books",
              "forum": "forums and Q&A", "code": "source code", "educational": "educational material"}


def b(x, d=1):
    return f"{x / 1e9:,.{d}f} billion"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    e = json.loads((RESULTS / "estimate.json").read_text())
    pop, man = population(), manifest()
    keys = [k for k, *_ in PANEL]
    labels = {k: v["label"] for k, v in e["tokenizers"].items()}
    users = {k: v["used_by"] for k, v in e["tokenizers"].items()}
    T = e["totals"]; X = e["cross_check_reservoir_heads"]
    L = []; P = L.append

    P("# How many tokens are in the atlas-v3 training corpus?\n")
    P("*Measured on 12 September 2026 from a stratified sample of the corpus stored on the build volume. "
      "All numbers are in `results/estimate.json`; the technical notes are at the end of this page.*\n")

    P("## The short version\n")
    o = T["o200k"]; ll = T["llama3"]; q = T["qwen3"]
    P(f"The corpus the new map was built from holds **{pop['n']:,} records** and **{pop['logical_bytes']/1e9:,.1f} GB of text**. "
      f"How many tokens that is depends on whose tokenizer you count with, because every model family cuts text into "
      f"pieces differently. Counted the way OpenAI's current models do it, it is **about {o['tokens']/1e9:,.0f} billion tokens**; "
      f"counted the way Llama 3 does it, **about {ll['tokens']/1e9:,.0f} billion**; the way Qwen does it, **about {q['tokens']/1e9:,.0f} billion**. "
      f"Older tokenizers with small vocabularies, such as GPT-2's or Mistral 7B's, make it look much larger, up to "
      f"**{max(T[k]['tokens'] for k in keys)/1e9:,.0f} billion**.\n")
    P(f"Each estimate is accurate to within about ±{100*1.96*max(T[k]['rel_se'] for k in keys):.1f}% from sampling alone. "
      f"A second, independent read from the build's own 500,000-record sample lands within "
      f"{100*max(abs(X[k]['ratio_to_primary']-1) for k in keys if k in T):.1f}% of the first for every tokenizer.\n")

    P("## Token counts by tokenizer\n")
    rows = []
    for k in sorted(keys, key=lambda k: T[k]["tokens"]):
        t = T[k]
        rows.append([labels[k], users[k], f"{t['tokens']/1e9:,.1f} billion", f"±{100*1.96*t['rel_se']:.2f}%", f"{t['bytes_per_token']:.2f}",
                     f"{X[k]['ratio_to_primary']:.3f}" if k in X else "–"])
    P(table(["tokenizer", "used by", "tokens in the corpus", "95% margin", "bytes of text per token", "cross-check ratio"], rows)); P("")
    P("**How to read this.** \"Tokens in the corpus\" is the estimated total if you ran that tokenizer over every record. "
      "The margin is the statistical uncertainty from having sampled rather than tokenized everything; 1.00 in the "
      "cross-check column would mean the second method agreed exactly. \"Bytes per token\" says how much text one token "
      "covers on average: a bigger vocabulary means bigger pieces and fewer tokens. The corpus is 60% non-English by "
      "bytes, which is where tokenizers differ most: a tokenizer built mostly for English spends several tokens on a "
      "word that a multilingual one covers in one.\n")

    P("## Where the tokens are\n")
    ax = e["by_axis"]
    rows = []
    for a in sorted(ax, key=lambda a: -ax[a]["o200k"]["tokens"]):
        r = ax[a]
        rows.append([AXIS_LABEL.get(a, a), f"{r['o200k']['records']:,}", f"{r['o200k']['bytes']/1e9:,.1f} GB",
                     f"{r['o200k']['tokens']/1e9:,.1f} billion", f"{r['llama3']['tokens']/1e9:,.1f} billion", f"{r['qwen3']['tokens']/1e9:,.1f} billion",
                     f"{r['o200k']['bytes']/r['o200k']['tokens']:.2f}"])
    P(table(["kind of text", "records", "text", "tokens (o200k)", "tokens (Llama 3)", "tokens (Qwen)", "bytes per o200k token"], rows)); P("")
    lg = e["by_language_of_source"]
    top = sorted(lg, key=lambda l: -lg[l]["o200k"])[:14]
    rows = [[l, f"{lg[l]['o200k']/1e9:,.1f} billion", f"{lg[l]['llama3']/1e9:,.1f} billion", f"{lg[l]['qwen3']/1e9:,.1f} billion", f"{lg[l]['gemma3']/1e9:,.1f} billion"] for l in top]
    rest = [l for l in lg if l not in top]
    rows.append(["all other languages (" + str(len(rest)) + ")", *[f"{sum(lg[l][k] for l in rest)/1e9:,.1f} billion" for k in ("o200k", "llama3", "qwen3", "gemma3")]])
    P("By the language of the source (a source is one language, so this is exact at source level):\n")
    P(table(["language", "o200k", "Llama 3", "Qwen", "Gemma 3"], rows)); P("")

    P("## How the estimate was made\n")
    s = e["sample"]
    P(f"1. **What was counted.** The population is the {pop['n']:,} records that the map was fitted on, exactly as the "
      f"build recorded them: {pop['logical_bytes']/1e9:,.1f} GB of UTF-8 text across {pop['sources']} sources. Records were "
      f"capped at 4,000 characters when the corpus was fetched, so that cap applies here too.")
    P(f"2. **The sample.** Every one of the {pop['sources']} source files on the volume was read from start to end once, and "
      f"a uniform random sample of its records was kept (reservoir sampling), in proportion to the source's share of the "
      f"corpus with a floor of 1,500 records so that no source is guessed from its neighbours. That gave "
      f"{s['records']:,} records and {s['bytes']/1e9:,.2f} GB of text, {100*s['bytes']/pop['logical_bytes']:.2f}% of the corpus.")
    P("3. **Tokenizing.** Each sampled record was run through all 15 tokenizers, loaded from their official Hugging Face "
      "files, counting tokens only (no chat template, no special tokens).")
    P("4. **Extrapolating.** For each source, tokens per byte in its sample times the source's exact number of bytes in the "
      "corpus gives that source's tokens; summing over sources gives the total. Per-source byte counts come from the "
      "corpus manifest, adjusted for the small share of rows the build dropped as duplicates and calibrated so they sum "
      "exactly to the byte totals the build recorded for each kind of text.")
    P("5. **Uncertainty.** The margin is the standard statistical error of a ratio estimate within each source, combined "
      "across sources. It covers sampling only; the tokenizers themselves are deterministic.")
    P(f"6. **Cross-check.** The build itself kept a uniform random sample of 500,000 records, but only their first 600 "
      f"characters. Tokens per byte from those heads, applied to the exact byte totals, gives an independent estimate; the "
      f"ratio of that estimate to the main one is the last column of the first table. Heads are slightly denser in "
      f"tokens than whole records (titles, boilerplate, numbers), which is why the ratios sit a little above 1.\n")

    P("## Things to keep in mind\n")
    P("- These are counts of the text as stored for the build, with the 4,000-character cap. The original documents are longer; the map never saw the rest.")
    P("- A token count is not a training budget. Models that train on this corpus would also add separators between documents and drop or repeat parts of it.")
    P("- Two tokenizers with the same vocabulary size can still count differently; what matters is the vocabulary and the pre-tokenisation rules, and both are taken from the official files.")
    P("- Llama 4's tokenizer is gated on the Hub and could not be included.\n")

    P("## Technical appendix\n")
    P(f"- Population: `checkpoint.json` n = {pop['n']:,}, logical = {pop['logical_bytes']:,} bytes; per-record source ids from `src.u16`, "
      f"axis ids from `axis.u8`; per-axis byte totals from `logical_by_axis`.")
    P(f"- Sampling: Algorithm L reservoir sampling per file, seed {20260912}, k_i = max(1500, round(600000 × N_i / N)). Allocation and rows seen per source are in `results/sampling.json`.")
    P("- Per-source bytes: B_i = manifest logical_bytes_i × N_i / rows_i, then rescaled within each axis to match the exact axis total. Rescale factors: "
      + ", ".join(f"{a} {1/v:.4f}" for a, v in e["byte_calibration_by_axis"].items()) + ".")
    P("- Estimator: T̂ = Σ_i B_i r̂_i with r̂_i = Σt/Σb over the source's sample; Var(r̂_i) = var(t − r̂ b)/(n_i · b̄²); Var(T̂) = Σ_i B_i² Var(r̂_i); 95% interval ±1.96 SE.")
    P("- Cross-check: T̂_heads = Σ_axis B_axis × (Σt/Σb)_axis over the 500,000 reservoir heads (uniform over the retained population, 600-character heads).")
    P("- Tokenizers: `tokenizers.Tokenizer.from_file` on each repo's `tokenizer.json`, `encode_batch_fast`, `add_special_tokens=False`. o200k is read from the gpt-oss tokenizer file (o200k_harmony), which encodes ordinary text identically to o200k_base; cl100k from the Xenova/gpt-4 re-export.")
    P("- Per-source ratios, standard errors and sample sizes for every tokenizer are in `results/estimate.json` under `per_source`.")
    (EXP / "REPORT.md").write_text("\n".join(L))
    print("wrote", EXP / "REPORT.md")


if __name__ == "__main__":
    main()
