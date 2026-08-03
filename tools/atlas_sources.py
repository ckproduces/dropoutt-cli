"""Reference-corpus catalogue for the atlas build.

Separated from the build logic so the corpus can be edited without touching the
pipeline, and so the fetcher and the builder read the same list.

Every entry names an explicit config and split. Leaving a config as ``None`` on
a multi-config repo is what invoked a retired dataset script during the v1
build and silently removed a load-bearing source from the map.

``target`` is a row count, not a byte budget. Sources smaller than their target
return what they have; that is expected and recorded, not an error. What is *not*
acceptable is an axis quietly collapsing to nothing, so :data:`AXIS_FLOORS`
states the minimum each axis must reach for the map to mean what the report says
it means. Falling short does not stop the build — it is printed as a warning and
written into the manifest, where the next build can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    """One reference-corpus source.

    ``fields`` are the row keys to keep; everything else in the row is dropped at
    fetch time so the on-disk cache holds text rather than whole HF records.

    ``loader`` selects how the repo is opened:

    ``hub``      ``load_dataset(hf_id, config, split, streaming=True)``
    ``parquet``  a direct ``refs/convert/parquet`` or ``main`` URL — used where
                 the canonical loader is a retired script or where the automatic
                 conversion is faster than the shard the loader picks
    ``data_dir`` ``load_dataset(hf_id, data_dir=..., split=...)`` for repos that
                 partition by directory rather than by config
    """

    hf_id: str
    config: str | None
    split: str
    fields: tuple[str, ...]
    axis: str
    target: int
    lang: str
    loader: str = "hub"
    #: For ``parquet``/``data_dir`` loaders: the path under the repo.
    path: str | None = None
    #: ``main`` for files committed to the repo, ``convert`` for the Hub's
    #: auto-generated ``refs/convert/parquet`` branch. The second one is how a
    #: dataset whose only loader is a retired Python script can still be read:
    #: the Hub converted it, and the conversion is a plain parquet file.
    revision: str = "main"
    #: Repo the rows originally came from, when this is a mirror or a sample.
    canonical: str | None = None
    #: Extra alternates tried in order if the primary load fails. Same schema.
    fallbacks: tuple[dict, ...] = field(default_factory=tuple)

    @property
    def slug(self) -> str:
        """Stable directory name for the on-disk cache."""
        base = self.hf_id.replace("/", "__")
        if self.config:
            base = f"{base}__{self.config.replace('/', '_')}"
        if self.path and not self.config:
            # Only the first shard of a multi-shard path contributes, so a
            # comma-joined list does not become the directory name. Single-path
            # sources are unaffected — the slug is the cache key, and changing
            # it silently orphans everything already fetched under the old one.
            first = self.path.split(",")[0]
            tail = first.strip("/").replace("/", "_")[:40]
            base = f"{base}__{tail}"
        return base


def _fineweb(name: str, path: str, target: int, lang: str) -> Source:
    """FineWeb family, read as one parquet shard.

    The v2 build pulled FineWeb through the dataset-viewer rows API and got
    3,300 of 20,000 rows before an HTTP error — the web backbone the whole map
    is supposed to sit on. A single shard is a couple of gigabytes but streams
    row group by row group, which is both faster and far more reliable than
    paginating the viewer.
    """
    return Source(
        hf_id=name,
        config=None,
        split="train",
        fields=("text",),
        axis="web",
        target=target,
        lang=lang,
        loader="parquet",
        path=path,
    )


SOURCES: list[Source] = [
    # -- web backbone ------------------------------------------------------
    # Spec calls this the backbone and v1/v2 both shipped without it.
    _fineweb("HuggingFaceFW/fineweb", "sample/10BT/000_00000.parquet", 200_000, "en"),
    _fineweb("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet", 120_000, "en"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/tur_Latn/train/000_00000.parquet", 45_000, "tr"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/deu_Latn/train/000_00000.parquet", 25_000, "de"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/spa_Latn/train/000_00000.parquet", 25_000, "es"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/fra_Latn/train/000_00000.parquet", 25_000, "fr"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/rus_Cyrl/train/000_00000.parquet", 25_000, "ru"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/arb_Arab/train/000_00000.parquet", 25_000, "ar"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/cmn_Hani/train/000_00000.parquet", 25_000, "zh"),
    _fineweb("HuggingFaceFW/fineweb-2", "data/jpn_Jpan/train/000_00000.parquet", 20_000, "ja"),
    # -- mixture with provenance (Dolma's role) ----------------------------
    # Dolma itself is gated behind an agreement, so it cannot be part of an
    # unattended build. C4 fills the same slot: a documented web mixture that
    # is not FineWeb, so the two do not agree with each other by construction.
    Source("allenai/c4", "en", "train", ("text",), "mixture", 60_000, "en"),
    Source("allenai/c4", "multilingual", "train", ("text",), "mixture", 40_000, "multi",
           fallbacks=({"hf_id": "allenai/c4", "config": "tr", "split": "train"},)),
    # -- encyclopedic, deliberately many languages --------------------------
    Source("wikimedia/wikipedia", "20231101.en", "train", ("text",), "encyclopedic", 60_000, "en"),
    Source("wikimedia/wikipedia", "20231101.tr", "train", ("text",), "encyclopedic", 45_000, "tr"),
    Source("wikimedia/wikipedia", "20231101.de", "train", ("text",), "encyclopedic", 25_000, "de"),
    Source("wikimedia/wikipedia", "20231101.es", "train", ("text",), "encyclopedic", 25_000, "es"),
    Source("wikimedia/wikipedia", "20231101.fr", "train", ("text",), "encyclopedic", 25_000, "fr"),
    Source("wikimedia/wikipedia", "20231101.ar", "train", ("text",), "encyclopedic", 25_000, "ar"),
    Source("wikimedia/wikipedia", "20231101.ru", "train", ("text",), "encyclopedic", 25_000, "ru"),
    # zh and az both returned zero rows in v2 on a 20s first-row timeout. They
    # are kept, with the fetcher's much longer per-source budget behind them.
    Source("wikimedia/wikipedia", "20231101.zh", "train", ("text",), "encyclopedic", 25_000, "zh"),
    Source("wikimedia/wikipedia", "20231101.ja", "train", ("text",), "encyclopedic", 20_000, "ja"),
    Source("wikimedia/wikipedia", "20231101.ko", "train", ("text",), "encyclopedic", 15_000, "ko"),
    Source("wikimedia/wikipedia", "20231101.fa", "train", ("text",), "encyclopedic", 15_000, "fa"),
    Source("wikimedia/wikipedia", "20231101.hi", "train", ("text",), "encyclopedic", 15_000, "hi"),
    Source("wikimedia/wikipedia", "20231101.az", "train", ("text",), "encyclopedic", 12_000, "az"),
    Source("mcemilg/news-cat", None, "train", ("text",), "encyclopedic", 5_000, "tr"),
    # -- code, spread across languages --------------------------------------
    # v2 asked the-stack-smol-XS for 6,000 rows per language. That repo holds
    # exactly 100 rows per language, so ten languages produced 662 rows in total
    # and one 0.3% StarCoderData mirror became 89% of the code axis — the map
    # learned "Python" where it was supposed to learn "code".
    #
    # the-stack-smol is gated and cannot be part of an unattended build.
    # CommitPackFT is not, holds 310,672 rows across these twenty languages, and
    # its records are whole source files. Its own loader is a retired script, so
    # it is read through the Hub's parquet conversion.
    *[
        Source("bigcode/commitpackft", None, "train",
               ("new_contents", "subject"), "code", target, f"code:{lang}",
               loader="parquet", revision="convert",
               path=f"{lang}/train/0000.parquet")
        for lang, target in (
            ("python", 12_000), ("javascript", 12_000), ("ruby", 10_000),
            ("php", 10_000), ("shell", 9_000), ("java", 9_000),
            ("c#", 8_000), ("c", 8_000), ("typescript", 5_500),
            ("go", 5_000), ("scala", 5_000), ("c++", 4_900),
            ("swift", 4_800), ("rust", 2_900), ("perl", 2_200),
            ("kotlin", 2_200), ("sql", 2_000), ("haskell", 1_300),
            ("lua", 900), ("r", 120),
        )
    ],
    # Text-to-code pairs: a different register from whole files, and the pair
    # form is what most instruction corpora containing code actually look like.
    *[
        Source("codeparrot/xlcost-text-to-code", None, "train", ("text", "code"),
               "code", 5_000, f"code:{lang.split('-')[0].lower()}",
               loader="parquet", revision="convert",
               path=f"{lang}/train/0000.parquet")
        for lang in ("Python-program-level", "C++-program-level",
                     "Java-program-level", "Javascript-program-level",
                     "C-program-level", "Csharp-program-level", "PHP-program-level")
    ],
    Source("codecomplete/starcoderdata_0.003", None, "train", ("text",), "code", 90_000,
           "code:multi", canonical="bigcode/starcoderdata"),
    Source("codeparrot/codeparrot-clean-valid", None, "train", ("content",), "code", 18_000, "code:python"),
    Source("iamtarun/python_code_instructions_18k_alpaca", None, "train",
           ("instruction", "output"), "code", 18_427, "en"),
    Source("google-research-datasets/mbpp", "full", "train", ("text", "code"), "code", 974, "en"),
    Source("openai/openai_humaneval", None, "test", ("prompt",), "code", 164, "en"),
    # proof-pile-2's only loader is a retired script and it has no parquet
    # conversion, so the underlying zstd-compressed JSONL shards are read
    # directly. Several shards are named because one is short of the target.
    Source("EleutherAI/proof-pile-2", None, "train", ("text",), "code", 30_000, "en",
           loader="json", path=",".join(
               f"algebraic-stack/train/{lang}0000.jsonl.zst"
               for lang in ("c", "cpp", "python", "haskell", "julia", "agda",
                            "fortran", "jupyter-notebook"))),
    # -- instruction and chat ----------------------------------------------
    Source("OpenAssistant/oasst1", None, "train", ("text",), "instruction", 84_000, "multi"),
    Source("openbmb/UltraFeedback", None, "train", ("instruction",), "instruction", 63_000, "en"),
    Source("HuggingFaceH4/ultrachat_200k", None, "train_sft", ("prompt",), "instruction", 80_000, "en"),
    Source("teknium/OpenHermes-2.5", None, "train", ("conversations",), "instruction", 50_000, "en"),
    Source("tatsu-lab/alpaca", None, "train", ("instruction", "output"), "instruction", 52_000, "en"),
    Source("databricks/databricks-dolly-15k", None, "train",
           ("instruction", "response"), "instruction", 15_011, "en"),
    Source("Anthropic/hh-rlhf", None, "train", ("chosen",), "instruction", 50_000, "en"),
    Source("HuggingFaceH4/no_robots", None, "train", ("messages",), "instruction", 9_500, "en"),
    Source("turkish-nlp-suite/InstrucTurca", None, "train", ("Input", "Output"), "instruction", 60_000, "tr"),
    Source("merve/turkish_instructions", None, "train", ("talimat", "çıktı"), "instruction", 9_000, "tr"),
    Source("TFLai/Turkish-Alpaca", None, "train", ("instruction", "output"), "instruction", 20_000, "tr"),
    # -- math ---------------------------------------------------------------
    Source("open-web-math/open-web-math", None, "train", ("text",), "math", 60_000, "en"),
    Source("HuggingFaceTB/finemath", "finemath-3plus", "train", ("text",), "math", 60_000, "en"),
    Source("microsoft/orca-math-word-problems-200k", None, "train",
           ("question", "answer"), "math", 60_000, "en"),
    Source("nvidia/OpenMathInstruct-2", None, "train", ("problem", "generated_solution"),
           "math", 40_000, "en"),
    Source("openai/gsm8k", "main", "train", ("question", "answer"), "math", 7_473, "en"),
    Source("deepmind/aqua_rat", "raw", "train", ("question", "rationale"), "math", 20_000, "en"),
    *[
        Source("EleutherAI/hendrycks_math", cfg, "train", ("problem", "solution"), "math", 2_000, "en")
        for cfg in (
            "algebra", "geometry", "counting_and_probability", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus",
        )
    ],
    # -- scientific ---------------------------------------------------------
    Source("BEE-spoke-data/peS2o-100k_en-xlong", None, "train", ("text",), "scientific", 100_000, "en"),
    Source("EleutherAI/proof-pile-2", None, "train", ("text",), "scientific", 60_000, "en",
           loader="json", path=",".join(
               f"arxiv/train/arXiv_{i:03d}.jsonl.zst" for i in range(4))),
    Source("CShorten/ML-ArXiv-Papers", None, "train", ("title", "abstract"), "scientific", 50_000, "en"),
    Source("qiaojin/PubMedQA", "pqa_artificial", "train", ("question", "long_answer"),
           "scientific", 40_000, "en"),
    Source("qiaojin/PubMedQA", "pqa_labeled", "train", ("question", "long_answer"),
           "scientific", 1_000, "en"),
    # -- dialogue and QA ----------------------------------------------------
    Source("HuggingFaceH4/stack-exchange-preferences", None, "train", ("question",),
           "dialogue", 80_000, "en"),
    Source("rajpurkar/squad", None, "train", ("context", "question"), "dialogue", 60_000, "en"),
    Source("tau/commonsense_qa", None, "train", ("question",), "dialogue", 9_741, "en"),
    Source("nvidia/HelpSteer2", None, "train", ("prompt", "response"), "dialogue", 20_000, "en"),
    # -- legal, finance, administrative -------------------------------------
    # Common Corpus' Finance/Legal Commons are not streamable inside a build
    # budget; LexGLUE's remaining configs plus contract and filing text cover
    # the same register at a size the fetcher can actually finish.
    *[
        Source("coastalcph/lex_glue", cfg, "train", ("text",), "legal_finance", 15_000, "en")
        for cfg in ("ecthr_a", "ecthr_b", "scotus", "eurlex", "ledgar", "unfair_tos")
    ],
    Source("albertvillanova/legal_contracts", None, "train", ("text",), "legal_finance", 25_000, "en",
           loader="parquet", revision="convert",
           path="default/partial-train/0000.parquet"),
    Source("gbharti/finance-alpaca", None, "train", ("instruction", "output"),
           "legal_finance", 20_000, "en"),
    Source("winddude/reddit_finance_43_250k", None, "train", ("selftext", "body"),
           "legal_finance", 20_000, "en"),
    # -- structured and tabular (exercises the extractor) --------------------
    # Salesforce/wikitablequestions 404s; the dataset moved. Text-to-SQL repos
    # cover the same "record is a table plus a question" shape and are current.
    Source("b-mc2/sql-create-context", None, "train", ("question", "answer"), "structured", 30_000, "en"),
    Source("gretelai/synthetic_text_to_sql", None, "train",
           ("sql_prompt", "sql", "sql_context"), "structured", 30_000, "en"),
    Source("xlangai/spider", None, "train", ("question", "query"), "structured", 7_000, "en"),
    Source("motherduckdb/duckdb-text2sql-25k", None, "train", ("prompt", "query"),
           "structured", 20_000, "en"),
    # WikiTableQuestions is deliberately absent. Salesforce/wikitablequestions
    # 404s, the stanfordnlp and bare copies are script-only, and the Hub has no
    # parquet conversion for either — so there is no way to read it in an
    # unattended build. The four text-to-SQL sources above clear the structured
    # floor without it; leaving a dead entry in the list is how v2 ended up
    # reporting a zero-row source as if it were a source.
]


#: Minimum retained rows per axis for the map to describe what the report claims.
#: A shortfall is printed and recorded; it never aborts the build.
AXIS_FLOORS: dict[str, int] = {
    "web": 200_000,
    "encyclopedic": 200_000,
    "code": 150_000,
    "instruction": 200_000,
    "math": 120_000,
    "scientific": 120_000,
    "dialogue": 100_000,
    "legal_finance": 60_000,
    "structured": 50_000,
    "mixture": 40_000,
}

#: Minimum rows per code language before the code axis is "spread across
#: languages" rather than "one mirror of mostly-Python". v2 managed 36-88.
CODE_LANGUAGE_FLOOR = 2_000

#: Non-English share of retained records, below which the map is an English map
#: with decoration. v2 landed near 20% and produced language-shaped L1 regions.
NON_ENGLISH_FLOOR = 0.30


def axis_totals(rows_by_source: dict[str, int]) -> dict[str, int]:
    """Sum retained rows per axis, keyed by :attr:`Source.slug`."""
    totals: dict[str, int] = {axis: 0 for axis in AXIS_FLOORS}
    for src in SOURCES:
        totals[src.axis] = totals.get(src.axis, 0) + rows_by_source.get(src.slug, 0)
    return totals
