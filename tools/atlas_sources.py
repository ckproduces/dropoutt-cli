"""Frozen source and quota plan for the Atlas public-web corpus.

The corpus is sized in retained UTF-8 bytes. Row targets are only safety caps;
they never determine mixture weights. General web data follows a frozen
snapshot of W3Techs website content-language shares. Public-domain and openly
licensed probes consume part of the English quota, so adding a probe never
silently increases English's final share.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASELINE_CATALOGUE_COMMIT = "public-web-2026-08-30"
BASELINE_SCALE = 1.0
V1_REFERENCE_RECORDS = 2_125_556
GIB = 1024**3
LOGICAL_BYTE_TARGET = 300 * GIB
# Bytes already retained on ck512 from the 150 GiB freeze. New English web and
# encyclopedic quota above these amounts is fetched as extra sources so the
# completed shards are not rewritten from scratch.
CACHED_FINEWEB_BYTES = 63_619_203_072
CACHED_WIKIPEDIA_EN_BYTES = 4_831_838_208

# Retained rows averaged about 2.2 KiB in the previous build. This is a storage
# estimate and a builder safety cap, not a sampling target.
ESTIMATED_BYTES_PER_RECORD = 2_200
TARGET_ROWS = (LOGICAL_BYTE_TARGET + ESTIMATED_BYTES_PER_RECORD - 1) // ESTIMATED_BYTES_PER_RECORD

DEFAULT_STORAGE_ROOT = Path(
    os.environ.get("DROPOUTT_ATLAS_STORAGE", "/Volumes/ck512/dropoutt-atlas-v2")
)
DEFAULT_CACHE = DEFAULT_STORAGE_ROOT / "corpus-cache"
DEFAULT_WORK = DEFAULT_STORAGE_ROOT / "work"
DEFAULT_LOGS = DEFAULT_STORAGE_ROOT / "logs"
DEFAULT_RELEASE = DEFAULT_STORAGE_ROOT / "release"

# W3Techs, "Usage statistics of content languages for websites", frozen on
# 2026-08-27. The published numeric percentages sum to 99.6%. W3Techs groups
# the rest as <0.1% each; the final 0.4% is an explicit operational long-tail
# proxy split across eight languages from that group. Published shares such as
# Turkish's 1.6% are not renormalized.
WEB_LANGUAGE_SHARE_SOURCE = "https://w3techs.com/technologies/overview/content_language"
WEB_LANGUAGE_SHARE_DATE = "2026-08-27"
WEB_LANGUAGE_SHARES: dict[str, float] = {
    "en": 0.495,
    "es": 0.060,
    "de": 0.059,
    "ja": 0.049,
    "fr": 0.045,
    "pt": 0.041,
    "ru": 0.034,
    "it": 0.028,
    "nl": 0.022,
    "pl": 0.018,
    "tr": 0.016,
    "zh": 0.013,
    "id": 0.012,
    "cs": 0.009,
    "fa": 0.009,
    "vi": 0.009,
    "ko": 0.009,
    "uk": 0.006,
    "ar": 0.006,
    "hu": 0.006,
    "sv": 0.005,
    "ro": 0.005,
    "el": 0.004,
    "da": 0.004,
    "fi": 0.004,
    "he": 0.004,
    "sk": 0.004,
    "th": 0.003,
    "bg": 0.003,
    "no": 0.003,  # W3Techs Norwegian + Norwegian Bokmal.
    "hr": 0.002,
    "sr": 0.002,
    "lt": 0.002,
    "sl": 0.001,
    "ca": 0.001,
    "et": 0.001,
    "lv": 0.001,
    "bn": 0.001,
    "hi": 0.0005,
    "sw": 0.0005,
    "ta": 0.0005,
    "ur": 0.0005,
    "ka": 0.0005,
    "ms": 0.0005,
    "bs": 0.0005,
    "az": 0.0005,
}

if abs(sum(WEB_LANGUAGE_SHARES.values()) - 1.0) > 1e-9:
    raise RuntimeError("web language shares must sum to 1.0")


def _allocate_exact(total: int, shares: dict[str, float]) -> dict[str, int]:
    """Allocate an integer byte total with deterministic largest remainders."""
    raw = {key: total * share for key, share in shares.items()}
    result = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(shares, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    if sum(result.values()) != total:
        raise RuntimeError("byte allocation did not preserve the corpus total")
    return result


# FineWeb2 language-script directories. English comes from FineWeb.
FINEWEB2_SCRIPTS: dict[str, str] = {
    "es": "spa_Latn",
    "de": "deu_Latn",
    "ja": "jpn_Jpan",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "zh": "cmn_Hani",
    "id": "ind_Latn",
    "cs": "ces_Latn",
    "fa": "fas_Arab",
    "vi": "vie_Latn",
    "ko": "kor_Hang",
    "uk": "ukr_Cyrl",
    "ar": "arb_Arab",
    "hu": "hun_Latn",
    "sv": "swe_Latn",
    "ro": "ron_Latn",
    "el": "ell_Grek",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "he": "heb_Hebr",
    "sk": "slk_Latn",
    "th": "tha_Thai",
    "bg": "bul_Cyrl",
    "no": "nob_Latn",
    "hr": "hrv_Latn",
    "sr": "srp_Cyrl",
    "lt": "lit_Latn",
    "sl": "slv_Latn",
    "ca": "cat_Latn",
    "et": "ekk_Latn",
    "lv": "lvs_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "ur": "urd_Arab",
    "ka": "kat_Geor",
    "ms": "zsm_Latn",
    "bs": "bos_Latn",
    "az": "azj_Latn",
}

SUPPLEMENTAL_LANGUAGES: tuple[tuple[str, str], ...] = tuple(FINEWEB2_SCRIPTS.items())

# Web was 81.5% of the 200 GiB plan and came out 83.0% of the bytes actually
# retained. A map of what is in a training corpus that is five-sixths one axis
# is really a map of that axis. The 168 GiB of web already on disk is kept and
# diluted rather than refetched or discarded: at 300 GiB the same bytes are
# 56% of the corpus.
#
# Growth is weighted by which axes can carry non-English text, not by which
# are most interesting. Measured on 2026-09-05 the declared non-web sources are
# 91.1% English by planned bytes, and scientific, code, forum, educational,
# legal and training were 100% English apiece -- so growing the axes evenly
# would have raised English from 48.6% to 65.4% while lowering the web share.
# Wikipedia (322 languages) and Wikisource (73) are the only openly-licensed
# non-web pools large enough to move the mixture, which is why encyclopedic and
# books take most of the increase and scientific, code and forum take least.
AXIS_TARGET_SHARES: dict[str, float] = {
    "web": 0.560,
    "encyclopedic": 0.130,
    "books": 0.070,
    "legal_government": 0.055,
    "scientific": 0.045,
    "training": 0.100,
    "code": 0.020,
    "forum": 0.010,
    "educational": 0.010,
}
if abs(sum(AXIS_TARGET_SHARES.values()) - 1.0) > 1e-9:
    raise RuntimeError("axis shares must sum to 1.0")

AXIS_TARGET_BYTES = _allocate_exact(LOGICAL_BYTE_TARGET, AXIS_TARGET_SHARES)
LANGUAGE_TARGET_BYTES = _allocate_exact(LOGICAL_BYTE_TARGET, WEB_LANGUAGE_SHARES)

# The W3Techs shares describe the language mix of *websites*, so they now size
# the web axis alone instead of the whole corpus. Until this build they sized
# the corpus and the entire non-web allocation was subtracted from English's
# quota, which only balances if every non-web source is English -- true when it
# was written, and the thing this build is undoing. Non-web axes now carry
# whatever mix their sources actually hold, and the corpus-wide English share
# is a reported outcome rather than an input.
SPECIALTY_BYTES = LOGICAL_BYTE_TARGET - AXIS_TARGET_BYTES["web"]
WEB_LANGUAGE_TARGET_BYTES = _allocate_exact(
    AXIS_TARGET_BYTES["web"], WEB_LANGUAGE_SHARES
)
if sum(WEB_LANGUAGE_TARGET_BYTES.values()) != AXIS_TARGET_BYTES["web"]:
    raise RuntimeError("web-language allocation does not equal the web-axis target")

FINEWEB2_BASELINE = tuple(
    (lang, script, WEB_LANGUAGE_TARGET_BYTES[lang]) for lang, script in SUPPLEMENTAL_LANGUAGES
)

# Hugging Face text-dataset popularity was sampled on the same date as the
# language plan. Popularity is a discovery signal, never an inclusion rule.
# Benchmarks, synthetic/SFT/chat corpora, duplicate web releases, noncommercial
# licences, and repositories without a declared redistribution basis are out.
HF_POPULARITY_SNAPSHOT_DATE = "2026-08-27"
HF_POPULARITY_URL = "https://huggingface.co/datasets?modality=modality%3Atext"
PROBE_SELECTION_POLICY = {
    "discovery": "Hugging Face text downloads and likes",
    "required": [
        "public and ungated",
        "declared public-domain or open redistribution basis",
        "natural-text pretraining utility",
        "no benchmark, preference, or gated corpus",
        "no duplicate allocation already represented by the web tier",
    ],
    "training_axis": (
        "Up to 5 GiB of public multilingual LLM training data (instruction, "
        "human chat, synthetic textbooks, edu-filtered pretrain) so coverage "
        "can place typical training mixes. Evaluation benchmarks stay out."
    ),
}

SOURCE_REVISIONS = {
    "HuggingFaceFW/fineweb": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "wikimedia/wikisource": "f31a033f5f3d2107b3e864e578710df104a00baa",
    "common-pile/youtube_filtered": "dff8c8a54e98bce64c2e7ce9a8466c144c1cddd6",
    "common-pile/arxiv_papers_filtered": "033cf7f53f9b348deec868c1a5a48484f3ee9e52",
    "common-pile/peS2o_filtered": "297747513bfb0ff1fbf61ddad3b03319d0f04597",
    "common-pile/stackv2_edu_filtered": "c354dbe88469a1153e97c6a63ac50591849654de",
    "common-pile/stackexchange_filtered": "c0ac7373830c688a43fc12d1988c4b19ccd884ab",
    "common-pile/regulations_filtered": "3327364490dfc7929009226ad667eceb2441d93a",
    "common-pile/caselaw_access_project_filtered": "50e1961a5ed8fdddcee64c9e66e1338de8953b63",
    "common-pile/project_gutenberg_filtered": "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7",
    "common-pile/pre_1929_books_filtered": "23f9d96dbb1db3324bbc9fbfe1f8299cc799c4d1",
    "common-pile/libretexts_filtered": "70388bca52b4a93515e14b1d56618fd7944988fd",
    "common-pile/pubmed_filtered": "c156f0569a92d8f2edc33cebe1f72f7d3e1cae84",
    "common-pile/arxiv_abstracts_filtered": "dc1ceab4755eb037ec61e49cf1350dab7ceee6e7",
    "common-pile/biodiversity_heritage_library_filtered": "0486ed637d0d7aaff264bc77fe21a7444e0215cd",
    "common-pile/doab_filtered": "defb24ca72ef6aba6ce0228b669eec06dcfbffbc",
    "common-pile/library_of_congress_filtered": "56725c7aa1bb320703e22eb5f42903173d5bac3d",
    "common-pile/pressbooks_filtered": "1a1d3b50d77f834370f8eb4c0d174668dd1676bb",
    "common-pile/oercommons_filtered": "506b6159dadcbc0dc67611cea024eedb04232fb2",
    "common-pile/usgpo_filtered": "b150cc22211de4d57f1b7f570097a00e65042424",
    "CohereLabs/aya_dataset": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "CohereLabs/aya_collection_language_split": "a3af2fde4b4cb5b2775830b11244a1a20b5f004f",
    "OpenAssistant/oasst2": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "HuggingFaceTB/smollm-corpus": "3ba9d605774198c5868892d7a8deda78031a781f",
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "joelniklaus/MultiLegalPile_Wikipedia_Filtered": "d0925f0e223bcfb2840e66328835380f96f8f589",
}


@dataclass(frozen=True)
class Source:
    """One public source with a byte quota and reproducible loader declaration."""

    hf_id: str
    config: str | None
    split: str
    fields: tuple[str, ...]
    axis: str
    target: int
    lang: str
    loader: str = "hub"
    path: str | None = None
    revision: str = "main"
    canonical: str | None = None
    fallbacks: tuple[dict, ...] = field(default_factory=tuple)
    target_bytes: int = 0
    source_role: str = "probe"
    license_policy: str = "public-domain-or-open-license"

    @property
    def slug(self) -> str:
        base = self.hf_id.replace("/", "__")
        if self.config:
            base = f"{base}__{self.config.replace('/', '_')}"
        if self.path and not self.config:
            first = self.path.split(",")[0]
            tail = first.strip("/").replace("/", "_")[:40]
            base = f"{base}__{tail}"
        return base


def _probe(hf_id: str, axis: str, target_bytes: int) -> Source:
    return Source(
        hf_id,
        None,
        "train",
        ("text",),
        axis,
        1_000_000_000,
        "en",
        target_bytes=int(target_bytes),
        source_role="probe",
        revision=SOURCE_REVISIONS[hf_id],
    )


def _training(
    hf_id: str,
    target_bytes: int,
    *,
    fields: tuple[str, ...] = ("text",),
    config: str | None = None,
    license_policy: str = "apache-2.0",
    lang: str = "en",
) -> Source:
    """One training-axis source. ``lang`` must name what the config holds.

    It defaulted to English for every source, so the aya Spanish, German,
    French, Chinese and Japanese splits were all declared English. That is not
    only a wrong quota: the builder subtracts a language's mean from its rows,
    so those records were being centred on English and kept their own language
    intact -- exactly the leak the per-language centering exists to stop.
    """
    return Source(
        hf_id,
        config,
        "train",
        fields,
        "training",
        1_000_000_000,
        lang,
        target_bytes=int(target_bytes),
        source_role="training",
        license_policy=license_policy,
        revision=SOURCE_REVISIONS[hf_id],
    )


def _split_axis(axis: str, shares: dict[str, float]) -> dict[str, int]:
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise RuntimeError(f"{axis} probe shares must sum to 1.0")
    return _allocate_exact(AXIS_TARGET_BYTES[axis], shares)


# Existing ck512 shards keep their old quotas. Growth is extra sources so a
# retry cannot rewrite 59 GiB of FineWeb from byte zero.
FINEWEB_PRIMARY_BYTES = min(WEB_LANGUAGE_TARGET_BYTES["en"], CACHED_FINEWEB_BYTES)
FINEWEB_EXTRA_BYTES = WEB_LANGUAGE_TARGET_BYTES["en"] - FINEWEB_PRIMARY_BYTES
WIKIPEDIA_EN_BYTES = min(AXIS_TARGET_BYTES["encyclopedic"], CACHED_WIKIPEDIA_EN_BYTES)
# Simple English Wikipedia held the whole encyclopedic remainder and delivered
# 0.19 GiB of it, because that is roughly the size of the wiki. Cap it at what
# it can actually produce and route the rest across the language plan.
CACHED_WIKIPEDIA_SIMPLE_BYTES = 214_748_365
WIKIPEDIA_SIMPLE_BYTES = min(
    AXIS_TARGET_BYTES["encyclopedic"] - WIKIPEDIA_EN_BYTES, CACHED_WIKIPEDIA_SIMPLE_BYTES
)
WIKIPEDIA_MULTILINGUAL_TOTAL = (
    AXIS_TARGET_BYTES["encyclopedic"] - WIKIPEDIA_EN_BYTES - WIKIPEDIA_SIMPLE_BYTES
)
# Weighted towards the languages the plan already buys the most web for, so the
# encyclopedic axis reads like the rest of the corpus rather than like whichever
# wikis happen to be largest.
# Covers every non-English language the plan declares, not the fifteen largest.
# The weights follow roughly what each wiki holds, but the tail is floored
# rather than proportional: the point of the small entries is that a language
# ends up with enough records to have a mean of its own fitted (see
# MIN_LANGUAGE_ROWS in the builder), and a proportional weight would leave
# Latvian or Swahili with too few to estimate one.
_WIKIPEDIA_WEIGHTS = {
    "de": 100, "fr": 90, "es": 80, "ru": 80, "ja": 70, "it": 60, "zh": 50,
    "pt": 50, "nl": 40, "pl": 40, "ar": 35, "uk": 30, "vi": 25, "tr": 25,
    "fa": 20, "id": 20, "cs": 20, "ko": 20, "sv": 20, "hu": 15, "ro": 15,
    "fi": 15, "da": 12, "he": 12, "no": 12, "ca": 10, "bg": 10, "el": 10,
    "sk": 10, "th": 8, "hr": 8, "sr": 8, "lt": 7, "sl": 7, "et": 6, "lv": 6,
    "bn": 6, "hi": 6, "ms": 5, "az": 5, "ta": 5, "ka": 4, "sw": 4, "bs": 4,
    "ur": 4,
}
if set(_WIKIPEDIA_WEIGHTS) - set(WEB_LANGUAGE_SHARES):
    raise RuntimeError("Wikipedia weights name a language the plan does not declare")
_WIKIPEDIA_WEIGHT_TOTAL = sum(_WIKIPEDIA_WEIGHTS.values())
WIKIPEDIA_MULTILINGUAL_SHARES = {
    lang: weight / _WIKIPEDIA_WEIGHT_TOTAL
    for lang, weight in _WIKIPEDIA_WEIGHTS.items()
}
WIKIPEDIA_MULTILINGUAL_BYTES = _allocate_exact(
    WIKIPEDIA_MULTILINGUAL_TOTAL, WIKIPEDIA_MULTILINGUAL_SHARES
)

SCIENTIFIC_BYTES = _split_axis(
    "scientific",
    {
        "common-pile/arxiv_papers_filtered": 0.20,
        "common-pile/peS2o_filtered": 0.20,
        "common-pile/pubmed_filtered": 0.40,
        "common-pile/arxiv_abstracts_filtered": 0.10,
        "common-pile/biodiversity_heritage_library_filtered": 0.10,
    },
)
# The books and educational axes were sized against Common Pile repositories
# that cannot reach their quotas. Measured against the Hub on 2026-09-03, the
# whole educational allocation is 3.43 GiB of parquet across four repos —
# libretexts is 0.11 GiB, pressbooks 0.18 GiB and oercommons 0.02 GiB, so the
# 4 GiB target was unreachable no matter how long the fetch ran. Books had the
# opposite problem: enough raw data (34 GiB) behind a source that yields about
# 84 KiB/s. Both are also English-only, in a corpus whose whole point is that
# non-English coverage is real.
#
# So the shortfalls move to Wikimedia sister projects, which are ungated,
# CC-BY-SA/GFDL, natural text, and genuinely multilingual: Wikisource carries
# public-domain literature in 73 languages and Wikipedia in 322. The exhausted
# repos keep quotas they can actually fill, and nothing already cached is
# rewritten.
BOOKS_BYTES = _split_axis(
    "books",
    {
        "common-pile/project_gutenberg_filtered": 0.125,
        "common-pile/pre_1929_books_filtered": 0.125,
        "common-pile/library_of_congress_filtered": 0.25,
        "wikimedia/wikisource": 0.50,
    },
)
EDUCATIONAL_BYTES = _split_axis(
    "educational",
    {
        "common-pile/libretexts_filtered": 0.03,
        "common-pile/pressbooks_filtered": 0.0425,
        "common-pile/oercommons_filtered": 0.005,
        "common-pile/doab_filtered": 0.30,
        "common-pile/youtube_filtered": 0.6225,
    },
)
# Wikisource is very unevenly sized across languages (Russian alone is 4.3 GiB,
# French 46 MiB), so the split follows what each wiki actually holds, capped so
# no single language takes the axis. English is left out: this allocation exists
# to buy non-English books.
WIKISOURCE_SHARES = {
    "ru": 0.20, "zh": 0.15, "he": 0.075, "ar": 0.075, "de": 0.075, "ko": 0.06,
    "es": 0.06, "la": 0.05, "sl": 0.05, "it": 0.04, "cs": 0.04, "ja": 0.04,
    "el": 0.035, "pt": 0.03, "fr": 0.02,
}
WIKISOURCE_BYTES = _allocate_exact(
    BOOKS_BYTES["wikimedia/wikisource"], WIKISOURCE_SHARES
)
# The legal axis was three US repositories and therefore 100% English, in the
# axis where non-English public text is least scarce: EU law is published in
# every official language. MultiLegalPile's Wikipedia-filtered release is
# CC-BY-4.0 (the unfiltered Multi_Legal_Pile and pile-of-law are CC-BY-NC and
# so are out under the probe policy) and carries 24 languages. Its English
# share is deliberately small here; the US repositories already cover that.
MULTI_LEGAL_ID = "joelniklaus/MultiLegalPile_Wikipedia_Filtered"
MULTI_LEGAL_FILES: dict[str, tuple[str, ...]] = {
    "bg": ("bg_caselaw_train_0", "bg_contracts_train_0", "bg_legislation_train_0",
        "bg_wikipedia_train_0"),
    "cs": ("cs_caselaw_train_0", "cs_contracts_train_0", "cs_legislation_train_0",
        "cs_other_train_0", "cs_wikipedia_train_0"),
    "da": ("da_caselaw_train_0", "da_contracts_train_0", "da_legislation_train_0",
        "da_other_train_0", "da_wikipedia_train_0"),
    "de": ("de_caselaw_train_0", "de_contracts_train_0", "de_legislation_train_0",
        "de_other_train_0", "de_other_train_1", "de_other_train_2", "de_wikipedia_train_0",
        "de_wikipedia_train_1", "de_wikipedia_train_2"),
    "el": ("el_caselaw_train_0", "el_contracts_train_0", "el_legislation_train_0",
        "el_other_train_0", "el_wikipedia_train_0"),
    "en": ("en_caselaw_train_0", "en_caselaw_train_1", "en_caselaw_train_10",
        "en_caselaw_train_2", "en_caselaw_train_3", "en_caselaw_train_4", "en_caselaw_train_5",
        "en_caselaw_train_6", "en_caselaw_train_7", "en_caselaw_train_8", "en_caselaw_train_9",
        "en_contracts_train_0", "en_contracts_train_1", "en_contracts_train_2",
        "en_legislation_train_0", "en_other_train_0", "en_other_train_1",
        "en_wikipedia_train_0", "en_wikipedia_train_1", "en_wikipedia_train_2",
        "en_wikipedia_train_3", "en_wikipedia_train_4", "en_wikipedia_train_5"),
    "es": ("es_caselaw_train_0", "es_contracts_train_0", "es_legislation_train_0",
        "es_other_train_0", "es_other_train_1", "es_wikipedia_train_0", "es_wikipedia_train_1"),
    "et": ("et_caselaw_train_0", "et_contracts_train_0", "et_legislation_train_0",
        "et_other_train_0", "et_wikipedia_train_0"),
    "fi": ("fi_caselaw_train_0", "fi_contracts_train_0", "fi_legislation_train_0",
        "fi_other_train_0", "fi_wikipedia_train_0"),
    "fr": ("fr_caselaw_train_0", "fr_contracts_train_0", "fr_legislation_train_0",
        "fr_other_train_0", "fr_wikipedia_train_0", "fr_wikipedia_train_1",
        "fr_wikipedia_train_2"),
    "ga": ("ga_legislation_train_0", "ga_wikipedia_train_0"),
    "hr": ("hr_caselaw_train_0", "hr_contracts_train_0", "hr_legislation_train_0",
        "hr_wikipedia_train_0"),
    "hu": ("hu_caselaw_train_0", "hu_contracts_train_0", "hu_legislation_train_0",
        "hu_other_train_0", "hu_wikipedia_train_0"),
    "it": ("it_caselaw_train_0", "it_contracts_train_0", "it_legislation_train_0",
        "it_other_train_0", "it_wikipedia_train_0", "it_wikipedia_train_1"),
    "lt": ("lt_caselaw_train_0", "lt_contracts_train_0", "lt_legislation_train_0",
        "lt_other_train_0", "lt_wikipedia_train_0"),
    "lv": ("lv_caselaw_train_0", "lv_contracts_train_0", "lv_legislation_train_0",
        "lv_wikipedia_train_0"),
    "mt": ("mt_caselaw_train_0", "mt_contracts_train_0", "mt_legislation_train_0",
        "mt_other_train_0", "mt_wikipedia_train_0"),
    "nl": ("nl_caselaw_train_0", "nl_contracts_train_0", "nl_legislation_train_0",
        "nl_other_train_0", "nl_wikipedia_train_0"),
    "pl": ("pl_caselaw_train_0", "pl_contracts_train_0", "pl_legislation_train_0",
        "pl_other_train_0", "pl_wikipedia_train_0", "pl_wikipedia_train_1"),
    "pt": ("pt_caselaw_train_0", "pt_contracts_train_0", "pt_legislation_train_0",
        "pt_other_train_0", "pt_wikipedia_train_0"),
    "ro": ("ro_caselaw_train_0", "ro_contracts_train_0", "ro_legislation_train_0",
        "ro_other_train_0", "ro_wikipedia_train_0"),
    "sk": ("sk_caselaw_train_0", "sk_contracts_train_0", "sk_legislation_train_0",
        "sk_other_train_0", "sk_wikipedia_train_0"),
    "sl": ("sl_caselaw_train_0", "sl_contracts_train_0", "sl_legislation_train_0",
        "sl_other_train_0", "sl_wikipedia_train_0"),
    "sv": ("sv_caselaw_train_0", "sv_contracts_train_0", "sv_legislation_train_0",
        "sv_other_train_0", "sv_wikipedia_train_0"),
}
MULTI_LEGAL_LANGUAGES = tuple(
    lang for lang in sorted(MULTI_LEGAL_FILES)
    if lang != "en" and lang in WEB_LANGUAGE_SHARES
)
LEGAL_BYTES = _split_axis(
    "legal_government",
    {
        "multi_legal": 0.70,
        "common-pile/regulations_filtered": 0.12,
        "common-pile/caselaw_access_project_filtered": 0.12,
        "common-pile/usgpo_filtered": 0.06,
    },
)
# Split evenly rather than by web share: these are the same body of law in
# each language, so a proportional split would just buy more of the same text
# in the languages already best covered.
MULTI_LEGAL_BYTES = _allocate_exact(
    LEGAL_BYTES["multi_legal"],
    {lang: 1 / len(MULTI_LEGAL_LANGUAGES) for lang in MULTI_LEGAL_LANGUAGES},
)
# The training axis is what LLM builders actually feed models, so it is the one
# axis where the atlas is mapping the training distribution directly rather
# than a proxy for it. At 5 GiB it was 43% Cosmopedia and FineWeb-Edu -- both
# English -- with five aya languages beside them. aya publishes 132 language
# splits, 45 of which are languages this plan declares, so the growth to 30 GiB
# buys languages rather than more English synthetic text.
AYA_CONFIGS = {
    "en": "english", "es": "spanish", "de": "german", "ja": "japanese",
    "fr": "french", "pt": "portuguese", "ru": "russian", "it": "italian",
    "nl": "dutch", "pl": "polish", "tr": "turkish", "zh": "simplified_chinese",
    "id": "indonesian", "cs": "czech", "fa": "iranian_persian",
    "vi": "vietnamese", "ko": "korean", "uk": "ukrainian",
    "ar": "standard_arabic", "hu": "hungarian", "sv": "swedish",
    "ro": "romanian", "el": "greek", "da": "danish", "fi": "finnish",
    "he": "hebrew", "sk": "slovak", "th": "thai", "bg": "bulgarian",
    "no": "norwegian", "hr": "croatian", "sr": "serbian", "lt": "lithuanian",
    "sl": "slovenian", "ca": "catalan", "et": "estonian",
    "lv": "standard_latvian", "bn": "bengali", "hi": "hindi",
    "sw": "swahili", "ta": "tamil", "ur": "urdu", "ka": "georgian",
    "ms": "standard_malay", "az": "north_azerbaijani",
}
if set(AYA_CONFIGS) - set(WEB_LANGUAGE_SHARES):
    raise RuntimeError("aya configs name a language the plan does not declare")
# English takes one share like any other language here. The axis already holds
# 9 GiB of English-only Cosmopedia and FineWeb-Edu; weighting aya's English
# split by web share on top of that would put the axis back where it started.
_AYA_WEIGHTS = {lang: (3 if lang == "en" else 1) for lang in AYA_CONFIGS}
_AYA_TOTAL = sum(_AYA_WEIGHTS.values())
TRAINING_BYTES = _split_axis(
    "training",
    {
        "aya_dataset": 0.03,
        "oasst2": 0.02,
        "aya_multilingual": 0.65,
        "cosmopedia": 0.15,
        "fineweb_edu": 0.15,
    },
)
AYA_BYTES = _allocate_exact(
    TRAINING_BYTES["aya_multilingual"],
    {lang: weight / _AYA_TOTAL for lang, weight in _AYA_WEIGHTS.items()},
)

# FineWeb/FineWeb2 are ODC-By. Common Pile probes stay public-domain/open-license.
# The training axis is a 5 GiB slice of public multilingual LLM training data.
SOURCES: list[Source] = [
    Source(
        "HuggingFaceFW/fineweb",
        "sample-100BT",
        "train",
        ("text",),
        "web",
        1_000_000_000,
        "en",
        target_bytes=FINEWEB_PRIMARY_BYTES,
        source_role="web",
        license_policy="odc-by-1.0-plus-common-crawl-terms",
        revision=SOURCE_REVISIONS["HuggingFaceFW/fineweb"],
    ),
    Source(
        "HuggingFaceFW/fineweb",
        "CC-MAIN-2024-51",
        "train",
        ("text",),
        "web",
        1_000_000_000,
        "en",
        target_bytes=FINEWEB_EXTRA_BYTES,
        source_role="web",
        license_policy="odc-by-1.0-plus-common-crawl-terms",
        revision=SOURCE_REVISIONS["HuggingFaceFW/fineweb"],
    ),
    Source(
        "wikimedia/wikipedia",
        "20231101.en",
        "train",
        ("text",),
        "encyclopedic",
        1_000_000_000,
        "en",
        target_bytes=WIKIPEDIA_EN_BYTES,
        source_role="probe",
        license_policy="cc-by-sa-3.0-and-gfdl",
        revision=SOURCE_REVISIONS["wikimedia/wikipedia"],
    ),
    Source(
        "wikimedia/wikipedia",
        "20231101.simple",
        "train",
        ("text",),
        "encyclopedic",
        1_000_000_000,
        "en",
        target_bytes=WIKIPEDIA_SIMPLE_BYTES,
        source_role="probe",
        license_policy="cc-by-sa-3.0-and-gfdl",
        revision=SOURCE_REVISIONS["wikimedia/wikipedia"],
    ),
    _probe("common-pile/arxiv_papers_filtered", "scientific", SCIENTIFIC_BYTES["common-pile/arxiv_papers_filtered"]),
    _probe("common-pile/peS2o_filtered", "scientific", SCIENTIFIC_BYTES["common-pile/peS2o_filtered"]),
    _probe("common-pile/pubmed_filtered", "scientific", SCIENTIFIC_BYTES["common-pile/pubmed_filtered"]),
    _probe(
        "common-pile/arxiv_abstracts_filtered",
        "scientific",
        SCIENTIFIC_BYTES["common-pile/arxiv_abstracts_filtered"],
    ),
    _probe(
        "common-pile/biodiversity_heritage_library_filtered",
        "scientific",
        SCIENTIFIC_BYTES["common-pile/biodiversity_heritage_library_filtered"],
    ),
    _probe("common-pile/stackv2_edu_filtered", "code", AXIS_TARGET_BYTES["code"]),
    _probe("common-pile/stackexchange_filtered", "forum", AXIS_TARGET_BYTES["forum"]),
    *[
        Source(
            MULTI_LEGAL_ID,
            None,
            "train",
            ("text",),
            "legal_government",
            1_000_000_000,
            lang,
            loader="json",
            path=",".join(
                f"data/{name}.jsonl.xz" for name in MULTI_LEGAL_FILES[lang]
            ),
            target_bytes=MULTI_LEGAL_BYTES[lang],
            source_role="probe",
            license_policy="cc-by-4.0",
            revision=SOURCE_REVISIONS[MULTI_LEGAL_ID],
        )
        for lang in MULTI_LEGAL_LANGUAGES
    ],
    _probe("common-pile/regulations_filtered", "legal_government", LEGAL_BYTES["common-pile/regulations_filtered"]),
    _probe(
        "common-pile/caselaw_access_project_filtered",
        "legal_government",
        LEGAL_BYTES["common-pile/caselaw_access_project_filtered"],
    ),
    _probe("common-pile/usgpo_filtered", "legal_government", LEGAL_BYTES["common-pile/usgpo_filtered"]),
    _probe("common-pile/project_gutenberg_filtered", "books", BOOKS_BYTES["common-pile/project_gutenberg_filtered"]),
    _probe("common-pile/pre_1929_books_filtered", "books", BOOKS_BYTES["common-pile/pre_1929_books_filtered"]),
    _probe("common-pile/library_of_congress_filtered", "books", BOOKS_BYTES["common-pile/library_of_congress_filtered"]),
    _probe("common-pile/libretexts_filtered", "educational", EDUCATIONAL_BYTES["common-pile/libretexts_filtered"]),
    _probe("common-pile/pressbooks_filtered", "educational", EDUCATIONAL_BYTES["common-pile/pressbooks_filtered"]),
    _probe("common-pile/oercommons_filtered", "educational", EDUCATIONAL_BYTES["common-pile/oercommons_filtered"]),
    _probe("common-pile/doab_filtered", "educational", EDUCATIONAL_BYTES["common-pile/doab_filtered"]),
    _probe("common-pile/youtube_filtered", "educational", EDUCATIONAL_BYTES["common-pile/youtube_filtered"]),
    *[
        Source(
            "wikimedia/wikisource",
            f"20231201.{lang}",
            "train",
            ("text",),
            "books",
            1_000_000_000,
            lang,
            target_bytes=WIKISOURCE_BYTES[lang],
            source_role="probe",
            license_policy="cc-by-sa-3.0-and-gfdl",
            revision=SOURCE_REVISIONS["wikimedia/wikisource"],
        )
        for lang in WIKISOURCE_SHARES
    ],
    *[
        Source(
            "wikimedia/wikipedia",
            f"20231101.{lang}",
            "train",
            ("text",),
            "encyclopedic",
            1_000_000_000,
            lang,
            target_bytes=WIKIPEDIA_MULTILINGUAL_BYTES[lang],
            source_role="probe",
            license_policy="cc-by-sa-3.0-and-gfdl",
            revision=SOURCE_REVISIONS["wikimedia/wikipedia"],
        )
        for lang in WIKIPEDIA_MULTILINGUAL_SHARES
    ],
    _training(
        "CohereLabs/aya_dataset",
        TRAINING_BYTES["aya_dataset"],
        fields=("inputs", "targets"),
    ),
    _training("OpenAssistant/oasst2", TRAINING_BYTES["oasst2"], fields=("text",)),
    *[
        _training(
            "CohereLabs/aya_collection_language_split",
            AYA_BYTES[lang],
            fields=("inputs", "targets"),
            config=config,
            lang=lang,
        )
        for lang, config in sorted(AYA_CONFIGS.items())
    ],
    _training(
        "HuggingFaceTB/smollm-corpus",
        TRAINING_BYTES["cosmopedia"],
        fields=("text",),
        config="cosmopedia-v2",
        license_policy="odc-by-1.0",
    ),
    _training(
        "HuggingFaceFW/fineweb-edu",
        TRAINING_BYTES["fineweb_edu"],
        fields=("text",),
        config="CC-MAIN-2024-46",
        license_policy="odc-by-1.0-plus-common-crawl-terms",
    ),
]

SOURCES = [source for source in SOURCES if source.target_bytes > 0]
if len({source.slug for source in SOURCES}) != len(SOURCES):
    raise RuntimeError("atlas sources have duplicate slugs")
# SOURCES holds English web plus every non-web source; the non-English web
# shards are generated per language at fetch time. The old form of this check
# compared that total against English's corpus quota, which only balances while
# every non-web source is English -- it was charging Wikisource's Russian and
# aya's Japanese to the English budget. What the total must actually equal is
# English's *web* quota plus the whole non-web allocation.
_DECLARED_TOTAL = WEB_LANGUAGE_TARGET_BYTES["en"] + SPECIALTY_BYTES
if sum(source.target_bytes for source in SOURCES) != _DECLARED_TOTAL:
    raise RuntimeError(
        "declared sources must consume English's web quota plus the non-web "
        f"allocation ({_DECLARED_TOTAL:,} bytes)"
    )

#: What the plan actually buys each language: its web quota plus every non-web
#: source declared in that language. This is the number to measure a fetch
#: against. ``LANGUAGE_TARGET_BYTES`` spreads the whole corpus by the W3Techs
#: *website* shares, which the plan stopped promising when the non-web axes
#: were allowed their own mixture -- against that, a language whose non-web
#: supply is thin can never reach 100% however long the fetch runs.
_PLANNED_LANGUAGE_RAW: dict[str, int] = {
    lang: WEB_LANGUAGE_TARGET_BYTES.get(lang, 0)
    + sum(
        source.target_bytes
        for source in SOURCES
        if source.axis != "web" and source.lang == lang
    )
    # The union, not the web plan: Wikisource's Latin has no web share but its
    # bytes are still in the corpus and still need a language to belong to.
    for lang in set(WEB_LANGUAGE_SHARES) | {
        source.lang for source in SOURCES if source.axis != "web"
    }
}
#: Largest first, so a monitor reading it renders in a stable, useful order.
PLANNED_LANGUAGE_BYTES: dict[str, int] = dict(
    sorted(_PLANNED_LANGUAGE_RAW.items(), key=lambda item: (-item[1], item[0]))
)
if sum(PLANNED_LANGUAGE_BYTES.values()) != LOGICAL_BYTE_TARGET:
    raise RuntimeError(
        "per-language plan does not add up to the corpus target: "
        f"{sum(PLANNED_LANGUAGE_BYTES.values()):,} vs {LOGICAL_BYTE_TARGET:,}"
    )

#: Planned English bytes across every declared source, for reporting. Non-web
#: sources are counted by the language they actually carry.
PLANNED_ENGLISH_BYTES = sum(
    source.target_bytes for source in SOURCES if source.lang == "en"
) + sum(
    byte for lang, byte in WEB_LANGUAGE_TARGET_BYTES.items() if lang == "en"
) - WEB_LANGUAGE_TARGET_BYTES["en"]

# Compatibility gates used by the builder. Byte targets are authoritative.
AXIS_FLOORS: dict[str, int] = {
    axis: max(1, target // 4_000) for axis, target in AXIS_TARGET_BYTES.items()
}
NON_ENGLISH_FLOOR = 1.0 - WEB_LANGUAGE_SHARES["en"]
CODE_LANGUAGE_FLOOR = 2_000


def axis_totals(rows_by_source: dict[str, int]) -> dict[str, int]:
    totals = dict.fromkeys(AXIS_TARGET_SHARES, 0)
    for source in SOURCES:
        totals[source.axis] = totals.get(source.axis, 0) + int(rows_by_source.get(source.slug, 0))
    return totals


def corpus_plan() -> dict:
    """Return the frozen, text-free fetch plan for inspection and provenance."""
    return {
        "catalogue": BASELINE_CATALOGUE_COMMIT,
        "storage_root": str(DEFAULT_STORAGE_ROOT),
        "logical_bytes": LOGICAL_BYTE_TARGET,
        "logical_gib": LOGICAL_BYTE_TARGET / GIB,
        "language_share_source": WEB_LANGUAGE_SHARE_SOURCE,
        "language_share_date": WEB_LANGUAGE_SHARE_DATE,
        "hf_popularity_snapshot_date": HF_POPULARITY_SNAPSHOT_DATE,
        "hf_popularity_url": HF_POPULARITY_URL,
        "probe_selection_policy": PROBE_SELECTION_POLICY,
        "axis_targets": {
            axis: {"share": AXIS_TARGET_SHARES[axis], "bytes": AXIS_TARGET_BYTES[axis]}
            for axis in AXIS_TARGET_SHARES
        },
        "language_targets": {
            lang: {
                "share": WEB_LANGUAGE_SHARES[lang],
                "bytes": LANGUAGE_TARGET_BYTES[lang],
                "web_bytes": WEB_LANGUAGE_TARGET_BYTES[lang],
            }
            for lang in WEB_LANGUAGE_SHARES
        },
        "sources": [
            {
                "slug": source.slug,
                "hf_id": source.hf_id,
                "axis": source.axis,
                "lang": source.lang,
                "target_bytes": source.target_bytes,
                "revision": source.revision,
                "source_role": source.source_role,
                "license_policy": source.license_policy,
            }
            for source in SOURCES
        ],
    }
