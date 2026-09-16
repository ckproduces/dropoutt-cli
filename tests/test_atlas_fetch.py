"""Network-free contracts for the atlas-v2 corpus collector."""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import atlas_sources  # noqa: E402  (tools/ is on sys.path only from here)
import fetch_corpus  # noqa: E402


def test_baseline_catalogue_is_pinned_and_scaled() -> None:
    assert atlas_sources.BASELINE_CATALOGUE_COMMIT == "public-web-2026-08-30"
    assert atlas_sources.BASELINE_SCALE == 1.0
    assert atlas_sources.V1_REFERENCE_RECORDS == 2_125_556
    assert atlas_sources.LOGICAL_BYTE_TARGET == 300 * 1024 ** 3
    assert abs(sum(atlas_sources.AXIS_TARGET_SHARES.values()) - 1.0) < 1e-9
    assert atlas_sources.AXIS_TARGET_SHARES["web"] == 0.560
    assert atlas_sources.AXIS_TARGET_SHARES["training"] == 0.100
    assert atlas_sources.AXIS_TARGET_BYTES["training"] == 30 * 1024 ** 3
    assert sum(atlas_sources.AXIS_TARGET_BYTES.values()) == atlas_sources.LOGICAL_BYTE_TARGET
    assert sum(atlas_sources.LANGUAGE_TARGET_BYTES.values()) == atlas_sources.LOGICAL_BYTE_TARGET
    assert len(atlas_sources.FINEWEB2_BASELINE) == 45
    assert len({source.slug for source in atlas_sources.SOURCES}) == len(atlas_sources.SOURCES)
    # SOURCES is English web plus the whole non-web allocation; the non-English
    # web shards are generated per language at fetch time.
    assert sum(source.target_bytes for source in atlas_sources.SOURCES) == (
        atlas_sources.WEB_LANGUAGE_TARGET_BYTES["en"] + atlas_sources.SPECIALTY_BYTES
    )
    # Books now outweighs scientific: it is one of the few non-web axes with a
    # multilingual supply, and scientific publishing is overwhelmingly English.
    assert atlas_sources.AXIS_TARGET_BYTES["books"] > atlas_sources.AXIS_TARGET_BYTES["scientific"]


def test_the_non_web_axes_are_not_an_english_monoculture() -> None:
    """The reason the plan grew: diluting web must not concentrate English.

    Before this plan the declared non-web sources were 91.1% English by bytes
    and six of eight axes were English-only, so every GiB added to cut the web
    share raised the English share instead.
    """
    non_web = [s for s in atlas_sources.SOURCES if s.axis != "web"]
    total = sum(s.target_bytes for s in non_web)
    english = sum(s.target_bytes for s in non_web if s.lang == "en")
    assert english / total < 0.50, f"non-web is {english / total:.1%} English"
    languages = {s.lang for s in non_web}
    assert len(languages) >= 30
    # Every declared language must be reachable somewhere, or it cannot get a
    # mean of its own fitted and will cluster by language instead of subject.
    planned = set(atlas_sources.WEB_LANGUAGE_SHARES)
    assert planned <= ({s.lang for s in atlas_sources.SOURCES} | {
        lang for lang, _, _ in atlas_sources.FINEWEB2_BASELINE
    })


def test_multilingual_legal_sources_carry_their_own_language() -> None:
    legal = [s for s in atlas_sources.SOURCES if s.axis == "legal_government"]
    multi = [s for s in legal if s.hf_id == atlas_sources.MULTI_LEGAL_ID]
    assert len(multi) >= 20
    assert all(s.lang != "en" for s in multi)
    assert all(s.license_policy == "cc-by-4.0" for s in multi)
    assert all(s.path and s.loader == "json" for s in multi)
    english = sum(s.target_bytes for s in legal if s.lang == "en")
    assert english / sum(s.target_bytes for s in legal) < 0.40


def test_training_sources_declare_the_language_they_hold() -> None:
    """aya's per-language splits were all declared English.

    The builder subtracts a language's own mean from its rows, so a Spanish
    split labelled English got centred on English and kept its Spanish.
    """
    aya = [s for s in atlas_sources.SOURCES if s.config and "aya" in s.hf_id.lower()]
    by_config = {s.config: s.lang for s in aya}
    assert by_config.get("spanish") == "es"
    assert by_config.get("german") == "de"
    assert by_config.get("french") == "fr"
    assert by_config.get("japanese") == "ja"
    assert by_config.get("simplified_chinese") == "zh"
    assert by_config.get("english") == "en"
    # The axis maps the training distribution directly, so it should not be a
    # mostly-English block: aya publishes 132 language splits and the plan uses
    # every one that matches a declared language.
    training = [s for s in atlas_sources.SOURCES if s.axis == "training"]
    assert len({s.lang for s in training}) >= 40
    english = sum(s.target_bytes for s in training if s.lang == "en")
    assert english / sum(s.target_bytes for s in training) < 0.50
    web = [source for source in atlas_sources.SOURCES if source.axis == "web"]
    assert sum(source.target for source in web) >= atlas_sources.AXIS_FLOORS["web"]


def test_supplemental_paths_are_public_reservoir_only() -> None:
    first_path = "data/tur_Latn/train/000_00000.parquet"
    paths = [
        first_path,
        "data/tur_Latn/train/000_00001.parquet",
        "data/deu_Latn/train/000_00000.parquet",
        "data/deu_Latn/train/000_00001.parquet",
        "data/eng_Latn/train/000_00000.parquet",
        "README.md",
    ]
    sources = fetch_corpus.supplemental_sources(paths)

    assert sources["tr"][0].path == first_path
    assert sources["de"][0].path == "data/deu_Latn/train/000_00000.parquet"
    assert sources["de"][0].hf_id == "HuggingFaceFW/fineweb-2"
    assert sources["es"] == []
    assert all(source.source_role == "web" for items in sources.values() for source in items)


def test_byte_quotas_are_exact_and_stable() -> None:
    assert fetch_corpus.byte_quotas(10, ["tr", "de", "es"]) == {
        "tr": 4, "de": 3, "es": 3,
    }
    assert fetch_corpus.byte_quotas(0, ["tr"]) == {"tr": 0}


def test_fetch_source_counts_utf8_bytes_and_only_final_allocation_overshoots(
    tmp_path: Path, monkeypatch
) -> None:
    source = atlas_sources.Source(
        "public/example", None, "train", ("text",), "web", 1, "tr"
    )
    text = "ğ" * 80  # 160 UTF-8 bytes, above the extractor's character floor.
    monkeypatch.setattr(fetch_corpus, "preflight_source", lambda *_: {"public": True, "card": {}})
    monkeypatch.setattr(fetch_corpus, "open_dataset", lambda *_: [{"text": text}])

    no_overshoot = fetch_corpus.fetch_source(
        source, tmp_path / "first", budget=5, stall=1, scale=1,
        refresh=False, max_chars=4000, byte_limit=100, allow_overshoot=False,
    )
    assert no_overshoot["rows"] == 0
    assert no_overshoot["logical_bytes"] == 0

    final = fetch_corpus.fetch_source(
        source, tmp_path / "final", budget=5, stall=1, scale=1,
        refresh=False, max_chars=4000, byte_limit=100, allow_overshoot=True,
        source_role="supplemental",
    )
    assert final["rows"] == 1
    assert final["logical_bytes"] == len(text.encode("utf-8"))
    assert final["logical_bytes"] > 100
    assert final["status"] == "complete"


def test_source_ledger_has_metadata_without_text(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    meta = {
        "slug": "example", "source_role": "baseline", "rows": 1,
        "logical_bytes": 120, "status": "complete", "checksum": {"value": "abc"},
    }
    fetch_corpus.write_source_ledger(ledger, [meta], {"scale": 1.0})
    payload = ledger.read_text(encoding="utf-8")
    assert '"logical_bytes": 120' in payload
    assert '"text"' not in payload


def test_cached_shard_is_read_then_deleted(tmp_path: Path) -> None:
    import gzip
    import json

    slug = "example__shard"
    shard_dir = tmp_path / slug
    shard_dir.mkdir()
    text = "ğ" * 80
    with gzip.open(shard_dir / "records.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "a", "text": text}) + "\n")

    rows = list(fetch_corpus.iter_cached_texts(tmp_path, slug, 4000))
    assert rows == [text]
    fetch_corpus.delete_cached_source(tmp_path, slug)
    assert not (tmp_path / slug).exists()


def test_fetch_source_keeps_larger_existing_shard(tmp_path: Path, monkeypatch) -> None:
    source = atlas_sources.Source(
        "public/example", None, "train", ("text",), "web", 1_000_000_000, "en"
    )
    text = "ğ" * 80
    monkeypatch.setattr(fetch_corpus, "preflight_source", lambda *_: {"public": True, "card": {}})
    monkeypatch.setattr(fetch_corpus, "open_dataset", lambda *_: [{"text": text}] * 3)
    first = fetch_corpus.fetch_source(
        source, tmp_path, budget=5, stall=1, scale=1,
        refresh=False, max_chars=4000, byte_limit=10_000, allow_overshoot=True,
    )
    assert first["rows"] == 3
    monkeypatch.setattr(fetch_corpus, "open_dataset", lambda *_: [{"text": text}])
    second = fetch_corpus.fetch_source(
        source, tmp_path, budget=5, stall=1, scale=1,
        refresh=False, max_chars=4000, byte_limit=10_000, allow_overshoot=True,
    )
    assert second["rows"] == 3
    assert second["logical_bytes"] == first["logical_bytes"]


def test_fineweb2_baseline_paths_are_unique() -> None:
    paths = [f"data/{script}/train/000_00000.parquet" for _, script, _ in atlas_sources.FINEWEB2_BASELINE]
    assert len(paths) == len(set(paths))
    web = [source for source in atlas_sources.SOURCES if source.axis == "web"]
    assert sum(source.target for source in web) >= atlas_sources.AXIS_FLOORS["web"]
