"""Records hidden in text files, and the checks calibrated on real generation output.

The scenarios here are taken from a 6,097-record Turkish generation run that
dropoutt originally read as 251 corpus documents. Every number asserted below
was measured on that corpus before the corresponding check was written.
"""

from __future__ import annotations

import json

from dropoutt.discovery import discover
from dropoutt.models import Profile
from dropoutt.readers import read_file
from dropoutt.sniff import iter_json_spans, sniff_text

RECORD = {"messages": [
    {"role": "user", "content": "Şu cümleyi düzeltir misin: bu cumle yalnış."},
    {"role": "assistant", "content": "Bu cümle yanlış."},
]}


def _jsonl(n: int) -> str:
    return "\n\n".join(json.dumps(RECORD, ensure_ascii=False) for _ in range(n))


def _fenced(n: int) -> str:
    body = json.dumps(RECORD, ensure_ascii=False, indent=2)
    return "\n\n".join(f"```json\n{body}\n```" for _ in range(n))


# -- the scanner ---------------------------------------------------------


def test_braces_inside_strings_do_not_split_a_record():
    """A record whose content mentions JSON must stay one record."""
    text = json.dumps({"messages": [{"role": "user", "content": 'use {"a": 1} here }}'}]})
    assert len(iter_json_spans(text + "\n" + text)) == 2


def test_an_escaped_quote_does_not_end_the_string_early():
    text = json.dumps({"messages": [{"role": "user", "content": 'he said "hi" }'}]})
    spans = iter_json_spans(text)
    assert len(spans) == 1
    assert json.loads(text[spans[0][0]:spans[0][1]])["messages"][0]["role"] == "user"


def test_an_unterminated_record_is_dropped_rather_than_guessed_at():
    assert iter_json_spans('{"messages": [{"role": "user"') == []


# -- the decision --------------------------------------------------------


def test_line_delimited_json_in_a_txt_file_is_recognised():
    framing = sniff_text(_jsonl(6))
    assert framing.is_records
    assert framing.kind == "line-delimited"
    assert framing.parsed == 6


def test_fenced_json_blocks_are_recognised():
    framing = sniff_text(_fenced(6))
    assert framing.is_records
    assert framing.kind == "fenced"


def test_prose_that_quotes_one_json_snippet_stays_prose():
    """The coverage rule, which is what stops this firing on documentation.

    Balanced braces are not evidence of anything. A blog post about a config
    file contains them and is still a blog post.
    """
    prose = (
        "Bir yapılandırma dosyası yazarken şu ayarı kullanabilirsiniz. "
        "Aşağıdaki örnek varsayılan değerleri gösterir ve çoğu durumda "
        "yeterlidir; üretim ortamında değiştirmeniz gereken tek alan zaman "
        "aşımı süresidir. " * 8
    )
    text = prose + '\n{"timeout": 30}\n{"retries": 3}\n' + prose
    framing = sniff_text(text)
    assert not framing.is_records
    assert framing.coverage < 0.5


def test_a_single_object_is_not_treated_as_a_record_container():
    assert not sniff_text(json.dumps(RECORD)).is_records


def test_malformed_records_do_not_make_a_file_read_as_prose():
    """A file that is 70% valid records is a record file with a parse problem.

    Reporting the malformed records is the entire point; refusing to read the
    file because some of them are malformed would hide them.
    """
    good = "\n\n".join(json.dumps(RECORD) for _ in range(7))
    bad = "\n\n".join('{"messages": [{"role": "user", "content": "a\\q"}]}' for _ in range(3))
    framing = sniff_text(good + "\n\n" + bad)
    assert framing.is_records
    assert 0.6 <= framing.parse_rate < 1.0


def test_text_between_records_is_reported_as_scaffolding():
    """The leaked control tag seen in the real corpus."""
    text = "<thinking_mode>off</thinking_mode>\n\n" + _jsonl(4)
    framing = sniff_text(text)
    assert framing.is_records
    assert any("thinking_mode" in s for s in framing.scaffolding)


def test_fence_markers_alone_are_not_reported_as_scaffolding():
    assert sniff_text(_fenced(5)).scaffolding == []


# -- reading -------------------------------------------------------------


def test_a_txt_file_of_records_is_read_as_records(tmp_path):
    path = tmp_path / "responses_0001.txt"
    path.write_text(_jsonl(5), encoding="utf-8")
    records = list(read_file(str(path), ".txt"))
    assert len(records) == 5
    assert all(r.payload and "messages" in r.payload for r in records)


def test_a_malformed_span_becomes_an_error_record_not_a_dropped_one(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text(_jsonl(3) + '\n\n{"messages": [{"role": "user", "content": "a\\q"}]}',
                    encoding="utf-8")
    records = list(read_file(str(path), ".txt"))
    assert len(records) == 4
    assert sum(1 for r in records if r.error) == 1


def test_genuine_prose_is_still_one_document_per_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Bu bir metin dosyasıdır ve içinde hiç kayıt yoktur.\n" * 20,
                    encoding="utf-8")
    records = list(read_file(str(path), ".txt"))
    assert len(records) == 1
    assert "text" in records[0].payload


def test_records_spanning_a_read_chunk_are_not_lost(tmp_path):
    """The incremental scanner must not drop a record straddling a chunk edge."""
    from dropoutt import readers

    path = tmp_path / "big.txt"
    path.write_text(_jsonl(200), encoding="utf-8")
    original = readers._SPAN_CHUNK
    try:
        readers._SPAN_CHUNK = 97  # deliberately smaller than one record
        records = list(read_file(str(path), ".txt"))
    finally:
        readers._SPAN_CHUNK = original
    assert len(records) == 200
    assert not any(r.error for r in records)


# -- shard folding -------------------------------------------------------


def test_sharded_siblings_become_one_dataset(tmp_path):
    for i in range(1, 6):
        (tmp_path / f"responses_{i:04d}.txt").write_text(_jsonl(3), encoding="utf-8")
    disc = discover(str(tmp_path))
    assert [d.name for d in disc.datasets] == ["responses"]
    assert len(disc.datasets[0].files) == 5
    assert disc.shard_families == ["responses"]


def test_hugging_face_shard_names_fold_too(tmp_path):
    for i in range(3):
        (tmp_path / f"train-{i:05d}-of-00003.jsonl").write_text(
            json.dumps(RECORD) + "\n", encoding="utf-8")
    disc = discover(str(tmp_path))
    assert [d.name for d in disc.datasets] == ["train"]


def test_two_files_are_a_coincidence_not_a_shard_family(tmp_path):
    for name in ("train.jsonl", "valid.jsonl"):
        (tmp_path / name).write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    disc = discover(str(tmp_path))
    assert sorted(d.name for d in disc.datasets) == ["train", "valid"]


def test_unsharded_names_keep_their_own_datasets(tmp_path):
    for name in ("alpha.jsonl", "beta.jsonl", "gamma.jsonl"):
        (tmp_path / name).write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
    disc = discover(str(tmp_path))
    assert sorted(d.name for d in disc.datasets) == ["alpha", "beta", "gamma"]


# -- the end-to-end failure this was written for -------------------------


def test_a_folder_of_record_bearing_text_files_scans_as_sft(tmp_path):
    """The whole point, stated as one test.

    Read literally, this folder is 5 corpus documents. Read correctly, it is 60
    SFT records. Everything downstream — the profile, the record count, which
    checks even run — follows from which of those two happens.
    """
    from dropoutt.runner import scan

    for i in range(1, 6):
        (tmp_path / f"responses_{i:04d}.txt").write_text(_jsonl(12), encoding="utf-8")
    result = scan(str(tmp_path))

    assert result.records_scanned == 60
    assert result.ctx.profile is Profile.SFT
    assert len(result.discovery.datasets) == 1
    assert any(f.check_id == "T0-FORMAT-001" for f in result.findings)
