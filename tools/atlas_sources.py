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
LOGICAL_BYTE_TARGET = 200 * GIB
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

AXIS_TARGET_SHARES: dict[str, float] = {
    "web": 0.815,
    "encyclopedic": 0.03,
    "scientific": 0.05,
    "code": 0.02,
    "forum": 0.01,
    "legal_government": 0.01,
    "books": 0.02,
    "educational": 0.02,
    "training": 0.025,
}
if abs(sum(AXIS_TARGET_SHARES.values()) - 1.0) > 1e-9:
    raise RuntimeError("axis shares must sum to 1.0")

AXIS_TARGET_BYTES = _allocate_exact(LOGICAL_BYTE_TARGET, AXIS_TARGET_SHARES)
LANGUAGE_TARGET_BYTES = _allocate_exact(LOGICAL_BYTE_TARGET, WEB_LANGUAGE_SHARES)

SPECIALTY_BYTES = LOGICAL_BYTE_TARGET - AXIS_TARGET_BYTES["web"]
if LANGUAGE_TARGET_BYTES["en"] <= SPECIALTY_BYTES:
    raise RuntimeError("English quota is too small for the public probe allocation")

WEB_LANGUAGE_TARGET_BYTES = dict(LANGUAGE_TARGET_BYTES)
WEB_LANGUAGE_TARGET_BYTES["en"] -= SPECIALTY_BYTES
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
) -> Source:
    return Source(
        hf_id,
        config,
        "train",
        fields,
        "training",
        1_000_000_000,
        "en",
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
WIKIPEDIA_SIMPLE_BYTES = AXIS_TARGET_BYTES["encyclopedic"] - WIKIPEDIA_EN_BYTES

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
BOOKS_BYTES = _split_axis(
    "books",
    {
        "common-pile/project_gutenberg_filtered": 0.125,
        "common-pile/pre_1929_books_filtered": 0.125,
        "common-pile/library_of_congress_filtered": 0.75,
    },
)
EDUCATIONAL_BYTES = _split_axis(
    "educational",
    {
        "common-pile/libretexts_filtered": 0.05,
        "common-pile/pressbooks_filtered": 0.15,
        "common-pile/oercommons_filtered": 0.02,
        "common-pile/doab_filtered": 0.78,
    },
)
LEGAL_BYTES = _split_axis(
    "legal_government",
    {
        "common-pile/regulations_filtered": 0.375,
        "common-pile/caselaw_access_project_filtered": 0.375,
        "common-pile/usgpo_filtered": 0.25,
    },
)
TRAINING_BYTES = _split_axis(
    "training",
    {
        "aya_dataset": 0.05,
        "oasst2": 0.04,
        "aya_en": 0.14,
        "aya_es": 0.08,
        "aya_de": 0.07,
        "aya_fr": 0.07,
        "aya_zh": 0.06,
        "aya_ja": 0.06,
        "cosmopedia": 0.20,
        "fineweb_edu": 0.23,
    },
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
    _training(
        "CohereLabs/aya_dataset",
        TRAINING_BYTES["aya_dataset"],
        fields=("inputs", "targets"),
    ),
    _training("OpenAssistant/oasst2", TRAINING_BYTES["oasst2"], fields=("text",)),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_en"],
        fields=("inputs", "targets"),
        config="english",
    ),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_es"],
        fields=("inputs", "targets"),
        config="spanish",
    ),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_de"],
        fields=("inputs", "targets"),
        config="german",
    ),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_fr"],
        fields=("inputs", "targets"),
        config="french",
    ),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_zh"],
        fields=("inputs", "targets"),
        config="simplified_chinese",
    ),
    _training(
        "CohereLabs/aya_collection_language_split",
        TRAINING_BYTES["aya_ja"],
        fields=("inputs", "targets"),
        config="japanese",
    ),
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
if sum(source.target_bytes for source in SOURCES) != LANGUAGE_TARGET_BYTES["en"]:
    raise RuntimeError("English sources must consume exactly the English language quota")

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
