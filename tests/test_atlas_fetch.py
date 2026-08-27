"""Network-free contracts for the atlas-v2 corpus collector."""

from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import atlas_sources
import fetch_corpus


def test_baseline_catalogue_is_pinned_and_scaled() -> None:
    assert atlas_sources.BASELINE_CATALOGUE_COMMIT == "dc12d8a"
    assert atlas_sources.BASELINE_SCALE == 1.0
    assert atlas_sources.LOGICAL_BYTE_TARGET == 40 * 1024 ** 3
    assert len(atlas_sources.FINEWEB2_BASELINE) == 69
    assert len(atlas_sources.SOURCES) == 184
    assert sum(source.target for source in atlas_sources.SOURCES) == 3_521_110
    assert len({source.slug for source in atlas_sources.SOURCES}) == 184


def test_supplemental_paths_are_public_reservoir_only_and_exclude_baseline() -> None:
    baseline_path = "data/tur_Latn/train/000_00000.parquet"
    paths = [
        baseline_path,
        "data/tur_Latn/train/000_00001.parquet",
        "data/deu_Latn/train/000_00000.parquet",
        "data/deu_Latn/train/000_00001.parquet",
        "data/eng_Latn/train/000_00000.parquet",
        "README.md",
    ]
    sources = fetch_corpus.supplemental_sources(paths)

    assert sources["tr"][0].path == "data/tur_Latn/train/000_00001.parquet"
    assert sources["de"][0].path == "data/deu_Latn/train/000_00001.parquet"
    assert sources["de"][0].hf_id == "HuggingFaceFW/fineweb-2"
    assert sources["es"] == []
    assert all(source.path != baseline_path for items in sources.values() for source in items)


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


def test_fineweb2_baseline_paths_are_unique() -> None:
    paths = [f"data/{script}/train/000_00000.parquet" for _, script, _ in atlas_sources.FINEWEB2_BASELINE]
    assert len(paths) == len(set(paths))
    web = [source for source in atlas_sources.SOURCES if source.axis == "web"]
    assert sum(source.target for source in web) >= atlas_sources.AXIS_FLOORS["web"]
