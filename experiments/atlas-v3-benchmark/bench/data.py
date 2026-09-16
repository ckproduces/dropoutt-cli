"""Loaders for the evaluation corpora.

Two pools. The local atlas-corpus cache (~/.cache/dropoutt/atlas-corpus) holds
the sources fetched for the v2 build; 90 of its 105 sources are not in the v3
build plan (tools/atlas-data/atlas-v3-fetch-list.json), so they are held out of
the map under test. The Hugging Face datasets cache holds labelled evaluation
sets (MMLU, MMLU-Pro, EXAMS-tr, WMT17 tr-en) that were never build inputs.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import pickle
import random
from pathlib import Path

from .common import MIN_PLACE_CHARS, REPO, SCRATCH

V2CACHE = Path(os.path.expanduser("~/.cache/dropoutt/atlas-corpus"))
HFDS = Path(os.path.expanduser("~/.cache/huggingface/datasets"))

#: Sources in the v3 build plan (same HF dataset, possibly other shards).
IN_BUILD_FAMILY = {"wikimedia__wikipedia", "HuggingFaceFW__fineweb", "HuggingFaceFW__fineweb-edu",
                   "HuggingFaceFW__fineweb-2"}


def in_build_family(slug: str) -> bool:
    return any(slug.startswith(p) for p in IN_BUILD_FAMILY)


def _cache(key: str):
    return SCRATCH / f"texts-{key}.pkl"


def v2cache(slug: str, n: int, seed: int = 0, min_chars: int = MIN_PLACE_CHARS,
            max_chars: int = 20000) -> list[str]:
    """Random sample of n placeable texts from one cached source (or all)."""
    key = f"{slug}-{n}-{seed}-{min_chars}"
    p = _cache(key)
    if p.exists():
        return pickle.loads(p.read_bytes())
    path = V2CACHE / slug / "records.jsonl.gz"
    texts = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line).get("text") or ""
            if len(t) >= min_chars:
                texts.append(t[:max_chars])
    rng = random.Random(seed)
    if len(texts) > n:
        texts = rng.sample(texts, n)
    else:
        rng.shuffle(texts)
    p.write_bytes(pickle.dumps(texts))
    return texts


def v2cache_rows(slug: str) -> int:
    meta = json.loads((V2CACHE / slug / "meta.json").read_text())
    return int(meta.get("rows") or 0)


def _arrow(pattern: str):
    from datasets import Dataset
    fs = sorted(glob.glob(str(HFDS / pattern)))
    if not fs:
        raise FileNotFoundError(pattern)
    return Dataset.from_file(fs[0])


def mmlu() -> tuple[list[str], list[str]]:
    """MMLU test: question + choices, labelled by its 57 subjects."""
    ds = _arrow("cais___mmlu/all/*/*/mmlu-test.arrow")
    texts, labels = [], []
    for r in ds:
        opts = "\n".join(f"- {c}" for c in r["choices"])
        t = f"{r['question']}\n{opts}"
        if len(t) >= MIN_PLACE_CHARS:
            texts.append(t); labels.append(r["subject"])
    return texts, labels


def mmlu_pro() -> tuple[list[str], list[str]]:
    ds = _arrow("TIGER-Lab___mmlu-pro/default/*/*/mmlu-pro-test.arrow")
    texts, labels = [], []
    for r in ds:
        opts = "\n".join(f"- {c}" for c in r["options"])
        t = f"{r['question']}\n{opts}"
        if len(t) >= MIN_PLACE_CHARS:
            texts.append(t); labels.append(r["category"])
    return texts, labels


def exams_tr() -> tuple[list[str], list[str]]:
    """EXAMS crosslingual Turkish: school-exam questions labelled by subject."""
    ds = _arrow("exams/crosslingual_tr/*/*/exams-train.arrow")
    texts, labels = [], []
    for r in ds:
        q = r["question"]
        opts = "\n".join(f"- {c}" for c in q["choices"]["text"])
        t = f"{q['stem']}\n{opts}"
        subj = str(r["info"].get("subject"))
        if len(t) >= MIN_PLACE_CHARS and subj not in ("None", ""):
            texts.append(t); labels.append(subj)
    return texts, labels


def wmt_pairs(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """WMT17 tr-en sentence pairs with both sides placeable (>= 80 chars)."""
    key = f"wmt17-{n}-{seed}"
    p = _cache(key)
    if p.exists():
        return pickle.loads(p.read_bytes())
    en, tr = [], []
    for split in ("test", "validation", "train"):
        ds = _arrow(f"wmt17/tr-en/*/*/wmt17-{split}.arrow")
        for r in ds:
            x = r["translation"]
            if len(x["en"]) >= MIN_PLACE_CHARS and len(x["tr"]) >= MIN_PLACE_CHARS:
                en.append(x["en"]); tr.append(x["tr"])
            if len(en) >= n * 3:
                break
        if len(en) >= n * 3:
            break
    rng = random.Random(seed)
    idx = rng.sample(range(len(en)), min(n, len(en)))
    out = ([en[i] for i in idx], [tr[i] for i in idx])
    p.write_bytes(pickle.dumps(out))
    return out


def probes() -> dict:
    return json.loads((REPO / "tools" / "atlas_probes.json").read_text())


# ---- synthetic out-of-distribution content ------------------------------

def ood_texts(n_per: int = 200, seed: int = 0) -> tuple[list[str], list[str]]:
    """Machine formats that must not be placed as confident neighbours."""
    import base64
    import string
    rng = random.Random(seed)
    texts, kinds = [], []
    for _ in range(n_per):
        raw = bytes(rng.getrandbits(8) for _ in range(180))
        texts.append(base64.b64encode(raw).decode()); kinds.append("base64")
        texts.append(" ".join(f"0x{rng.getrandbits(32):08x}" for _ in range(28))); kinds.append("hex_log")
        texts.append("".join(rng.choice("ACGT") for _ in range(260))); kinds.append("dna")
        ident = lambda: "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(1, 3)))
        texts.append(";".join(f"var {ident()}={ident()}({ident()},{rng.randint(0,99)})" for _ in range(22))); kinds.append("minified_js")
        texts.append(",".join(f"{rng.random()*1000:.3f}" for _ in range(40))); kinds.append("csv_numbers")
        texts.append("".join(chr(rng.randint(0x4E00, 0x9FFF)) if rng.random() < .5 else chr(rng.randint(0x400, 0x4FF)) for _ in range(120))); kinds.append("random_unicode")
        texts.append("".join(rng.choice(string.ascii_letters + "   ") for _ in range(240))); kinds.append("random_letters")
    return texts, kinds


# ---- the held-out source panel ------------------------------------------

#: (slug, axis, language) — 1,000 records each where the source has them.
SOURCE_PANEL = [
    # code
    ("bigcode__commitpackft__python_train_0000.parquet", "code", "code:python"),
    ("bigcode__commitpackft__java_train_0000.parquet", "code", "code:java"),
    ("bigcode__commitpackft__javascript_train_0000.parquet", "code", "code:javascript"),
    ("bigcode__commitpackft__go_train_0000.parquet", "code", "code:go"),
    ("bigcode__commitpackft__rust_train_0000.parquet", "code", "code:rust"),
    ("bigcode__commitpackft__c_train_0000.parquet", "code", "code:c"),
    ("bigcode__commitpackft__php_train_0000.parquet", "code", "code:php"),
    ("bigcode__commitpackft__ruby_train_0000.parquet", "code", "code:ruby"),
    ("bigcode__commitpackft__shell_train_0000.parquet", "code", "code:shell"),
    ("codeparrot__xlcost-text-to-code__Python-program-level_train_0000.parquet", "code", "code:python"),
    ("codecomplete__starcoderdata_0.003", "code", "code:multi"),
    ("codeparrot__codeparrot-clean-valid", "code", "code:python"),
    ("iamtarun__python_code_instructions_18k_alpaca", "code", "en"),
    # math
    ("EleutherAI__hendrycks_math__algebra", "math", "en"),
    ("EleutherAI__hendrycks_math__geometry", "math", "en"),
    ("EleutherAI__hendrycks_math__number_theory", "math", "en"),
    ("openai__gsm8k__main", "math", "en"),
    ("deepmind__aqua_rat__raw", "math", "en"),
    ("microsoft__orca-math-word-problems-200k", "math", "en"),
    ("open-web-math__open-web-math", "math", "en"),
    ("HuggingFaceTB__finemath__finemath-3plus", "math", "en"),
    ("nvidia__OpenMathInstruct-2", "math", "en"),
    # legal / finance
    ("coastalcph__lex_glue__ecthr_a", "legal_finance", "en"),
    ("coastalcph__lex_glue__eurlex", "legal_finance", "en"),
    ("coastalcph__lex_glue__ledgar", "legal_finance", "en"),
    ("coastalcph__lex_glue__scotus", "legal_finance", "en"),
    ("coastalcph__lex_glue__unfair_tos", "legal_finance", "en"),
    ("gbharti__finance-alpaca", "legal_finance", "en"),
    ("winddude__reddit_finance_43_250k", "legal_finance", "en"),
    # scientific
    ("BEE-spoke-data__peS2o-100k_en-xlong", "scientific", "en"),
    ("CShorten__ML-ArXiv-Papers", "scientific", "en"),
    ("EleutherAI__proof-pile-2__arxiv_train_arXiv_000.jsonl.zst", "scientific", "en"),
    ("qiaojin__PubMedQA__pqa_artificial", "scientific", "en"),
    # instruction / chat
    ("Anthropic__hh-rlhf", "instruction", "en"),
    ("HuggingFaceH4__no_robots", "instruction", "en"),
    ("HuggingFaceH4__ultrachat_200k", "instruction", "en"),
    ("OpenAssistant__oasst1", "instruction", "multi"),
    ("databricks__databricks-dolly-15k", "instruction", "en"),
    ("tatsu-lab__alpaca", "instruction", "en"),
    ("openbmb__UltraFeedback", "instruction", "en"),
    ("TFLai__Turkish-Alpaca", "instruction", "tr"),
    ("turkish-nlp-suite__InstrucTurca", "instruction", "tr"),
    # dialogue / QA
    ("HuggingFaceH4__stack-exchange-preferences", "dialogue", "en"),
    ("nvidia__HelpSteer2", "dialogue", "en"),
    ("rajpurkar__squad", "dialogue", "en"),
    # structured
    ("b-mc2__sql-create-context", "structured", "en"),
    ("gretelai__synthetic_text_to_sql", "structured", "en"),
    ("motherduckdb__duckdb-text2sql-25k", "structured", "en"),
    ("xlangai__spider", "structured", "en"),
    # web (held out: c4 is not a v3 input)
    ("allenai__c4__en", "web", "en"),
    ("allenai__c4__multilingual", "web", "multi"),
    # news, Turkish
    ("mcemilg__news-cat", "news", "tr"),
]

#: In-build-family web/encyclopedic sources by language (same HF datasets as v3 inputs).
LANG_PANEL_WEB = [
    ("HuggingFaceFW__fineweb__sample_10BT_000_00000.parquet", "en"),
    ("HuggingFaceFW__fineweb-2__data_tur_Latn_train_000_00000.parquet", "tr"),
    ("HuggingFaceFW__fineweb-2__data_deu_Latn_train_000_00000.parquet", "de"),
    ("HuggingFaceFW__fineweb-2__data_fra_Latn_train_000_00000.parquet", "fr"),
    ("HuggingFaceFW__fineweb-2__data_spa_Latn_train_000_00000.parquet", "es"),
    ("HuggingFaceFW__fineweb-2__data_rus_Cyrl_train_000_00000.parquet", "ru"),
    ("HuggingFaceFW__fineweb-2__data_arb_Arab_train_000_00000.parquet", "ar"),
    ("HuggingFaceFW__fineweb-2__data_cmn_Hani_train_000_00000.parquet", "zh"),
    ("HuggingFaceFW__fineweb-2__data_jpn_Jpan_train_000_00000.parquet", "ja"),
]
LANG_PANEL_WIKI = [(f"wikimedia__wikipedia__20231101.{l}", l) for l in
                   ("en", "tr", "de", "es", "fr", "ru", "ar", "zh", "ja", "ko", "hi", "fa", "az")]


def short_name(slug: str) -> str:
    s = slug
    for junk in ("_train_0000.parquet", "_train_000_00000.parquet", "_train_arXiv_000.jsonl.zst",
                 "_train_c0000.jsonl.zst", "-program-level", "__default_partial", "__data"):
        s = s.replace(junk, "")
    s = s.replace("HuggingFaceFW__", "").replace("HuggingFaceH4__", "").replace("HuggingFaceTB__", "")
    s = s.replace("EleutherAI__", "").replace("coastalcph__", "").replace("bigcode__", "")
    s = s.replace("codeparrot__", "").replace("wikimedia__wikipedia__20231101.", "wikipedia.")
    return s
