"""Each check gets a fixture built to trip it, and one built not to.

Fixtures are constructed in code rather than committed as blobs, so a reviewer
can read what each test is actually asserting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dropoutt.langid import LanguageDetector
from dropoutt.models import Profile, Severity
from dropoutt.runner import scan


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(root: Path, **kw):
    kw.setdefault("detector", LanguageDetector())
    return scan(str(root), **kw)


def findings_by_id(result) -> dict[str, object]:
    return {f.check_id: f for f in result.findings}


def chat(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


LONG = "bu bir yeterince uzun cevap metnidir ve ayrintili aciklama icerir"


# -- discovery and schema ------------------------------------------------


def test_detects_agent_logs_as_not_training_data(tmp_path):
    write_jsonl(tmp_path / "logs" / "a.jsonl", [
        {"type": "queue-operation", "operation": "enqueue",
         "timestamp": "2026-03-13T19:05:09Z", "sessionId": "abc"}
        for _ in range(30)
    ])
    result = run(tmp_path)
    assert "T0-SCHEMA-001" in findings_by_id(result)


def test_clean_dataset_produces_no_structural_findings(tmp_path):
    write_jsonl(tmp_path / "clean" / "train.jsonl",
                [chat(f"Soru numarasi {i} hakkinda bilgi verir misin", f"{LONG} {i}")
                 for i in range(80)])
    ids = set(findings_by_id(run(tmp_path)))
    # Zero findings must be a reachable state for the structural checks.
    assert not ids & {"T0-ROLE-001", "T0-ROLE-002", "T0-SCHEMA-002",
                      "T0-SCHEMA-003", "T0-ENC-001", "T0-DUP-001"}


def test_mixed_schemas_in_one_folder(tmp_path):
    records = [{"instruction": f"Konu {i}", "output": f"{LONG} {i}"} for i in range(60)]
    records += [{"prompt": f"Farkli sema {i}", "completion": LONG} for i in range(20)]
    write_jsonl(tmp_path / "mixed" / "train.jsonl", records)
    assert "T0-SCHEMA-002" in findings_by_id(run(tmp_path))


def test_unparseable_records_are_reported_not_raised(tmp_path):
    path = tmp_path / "broken" / "train.jsonl"
    write_jsonl(path, [chat(f"Soru {i}", f"{LONG} {i}") for i in range(30)])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"messages": [{"role": "user",\n')
        fh.write("not json at all\n")
    result = run(tmp_path)
    assert findings_by_id(result)["T0-SCHEMA-003"].count == 2


# -- roles ---------------------------------------------------------------


def test_sharegpt_role_vocabulary_is_flagged(tmp_path):
    """The quietest way to lose a dataset: `from: gpt` is not `role: assistant`."""
    write_jsonl(tmp_path / "sg" / "train.jsonl", [
        {"conversations": [{"from": "human", "value": f"Soru {i} hakkinda bilgi"},
                           {"from": "gpt", "value": f"{LONG} {i}"}]}
        for i in range(40)
    ])
    f = findings_by_id(run(tmp_path))["T0-ROLE-002"]
    assert f.count == 40
    assert set(f.data["roles"]) == {"human", "gpt"}


def test_missing_assistant_turn(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [{"messages": [{"role": "user", "content": f"Sadece kullanici {i}"}]}
                 for i in range(20)])
    f = findings_by_id(run(tmp_path))["T0-ROLE-001"]
    assert f.data["no_assistant"] == 20


# -- hygiene -------------------------------------------------------------


def test_mojibake_and_control_characters(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat("KodlamaÂ hatasÄ± burada", "Ã‡ok kÃ¶tÃ¼ bir kodlama\x07 var")] * 5)
    f = findings_by_id(run(tmp_path))["T0-ENC-001"]
    assert f.data["mojibake"] and f.data["control"]


def test_exact_duplicates_report_cluster_size(tmp_path):
    rec = chat("Ayni soru tekrar tekrar sorulur", LONG)
    write_jsonl(tmp_path / "d" / "train.jsonl", [rec] * 9 + [
        chat(f"Farkli soru {i}", f"{LONG} {i}") for i in range(20)])
    f = findings_by_id(run(tmp_path))["T0-DUP-001"]
    assert f.count == 8
    assert f.data["largest_cluster"] == 9


def test_ascii_folded_turkish(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat("Bu bir soru degil mi ve bu daha iyi bir cevap icin gerekli olan sey",
             "Evet bu dogru degil ve bir sey daha var gibi gorunuyor icin bu")
    ] * 10)
    assert "T1-LANG-004" in findings_by_id(run(tmp_path))


def test_proper_turkish_is_not_flagged_as_ascii_folded(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat("Bu bir soru değil mi ve bu daha iyi bir cevap için gerekli olan şey",
             "Evet bu doğru değil ve bir şey daha var gibi görünüyor için bu")
    ] * 10)
    assert "T1-LANG-004" not in findings_by_id(run(tmp_path))


def test_degenerate_responses(tmp_path):
    records = [chat(f"Yeterince uzun bir soru metni {i}", "Ok") for i in range(10)]
    records += [chat("Dongu testi burada uzun", "tekrar eden cumle bu ve devam ediyor " * 30)]
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    f = findings_by_id(run(tmp_path))["T0-DEGEN-001"]
    assert f.data["trivial"] == 10 and f.data["looping"] >= 1


# -- content -------------------------------------------------------------


def test_pii_is_detected_but_never_echoed(tmp_path):
    """A linter that leaks secrets into a shareable report is worse than none."""
    secret = "sk-proj-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH"
    email = "gizli.kullanici@ornek.com"
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat("Iletisim bilgileri nelerdir", f"Bana {email} veya {secret} ile ulasin")] * 3)
    f = findings_by_id(run(tmp_path))["T1-PII-001"]
    assert f.count == 3
    blob = " ".join(e.excerpt for e in f.evidence) + f.detail
    assert secret not in blob
    assert email not in blob


def test_tckn_checksum_rejects_arbitrary_eleven_digit_numbers(tmp_path):
    """A bare 11-digit regex would match every order id in the corpus."""
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat("Siparis numarasi", f"Numaraniz 12345678901 ve {LONG}")] * 5)
    f = findings_by_id(run(tmp_path)).get("T1-PII-001")
    assert f is None or "tckn" not in f.data.get("by_kind", {})


def test_identity_leakage_english_and_turkish(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat("Sen kimsin", "Bir yapay zeka dil modeli olarak kisisel gorusum yok"),
        chat("Who are you", "As an AI language model I do not have opinions"),
    ] * 5)
    assert findings_by_id(run(tmp_path))["T1-IDENT-001"].data["identity"] == 10


def test_style_tics_only_fire_above_the_rate_threshold(tmp_path):
    """One 'Elbette!' is a sentence. Forty percent of them is a style."""
    few = [chat(f"Soru {i} uzun metin", "Elbette! Yardimci olabilirim.") for i in range(3)]
    many = [chat(f"Baska soru {i} uzun metin", f"{LONG} {i}") for i in range(97)]
    write_jsonl(tmp_path / "low" / "train.jsonl", few + many)
    assert "T1-STYLE-001" not in findings_by_id(run(tmp_path / "low"))

    lots = [chat(f"Soru {i} uzun metin", "Elbette! Yardimci olabilirim.") for i in range(60)]
    write_jsonl(tmp_path / "high" / "train.jsonl", lots + many[:40])
    assert "T1-STYLE-001" in findings_by_id(run(tmp_path / "high"))


# -- deduplication and overlap -------------------------------------------


def test_cross_dataset_overlap_is_directional(tmp_path):
    """Containment of a small set inside a large one is the actionable case."""
    shared = [chat(f"Paylasilan soru {i} hakkinda ayrintili bilgi", f"{LONG} {i}")
              for i in range(20)]
    write_jsonl(tmp_path / "small" / "train.jsonl", shared)
    write_jsonl(tmp_path / "large" / "train.jsonl", shared + [
        chat(f"Sadece buyuk kumede olan soru {i}", f"{LONG} farkli {i}") for i in range(180)])

    f = findings_by_id(run(tmp_path))["T1-OVERLAP-001"]
    matrix = {(r["from"], r["to"]): r["fraction"] for r in f.data["matrix"]}
    small_in_large = matrix[("small", "large")]
    large_in_small = matrix[("large", "small")]
    assert small_in_large > 0.9, "the small set is contained in the large one"
    assert large_in_small < 0.2, "the reverse must not be true"
    assert small_in_large > large_in_small * 3, "matrix must not be symmetric"


def test_near_duplicate_count_never_exceeds_record_count(tmp_path):
    """Regression: two checks share the signature store and once double-counted."""
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat(f"Benzer soru {i} hakkinda ayrintili aciklama", f"{LONG} {i % 5}")
                 for i in range(100)])
    result = run(tmp_path)
    f = findings_by_id(result).get("T1-NDUP-001")
    if f is not None:
        assert f.count <= result.records_scanned


# -- policy --------------------------------------------------------------


def test_no_blocking_without_a_declared_target(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [{"messages": [{"role": "user", "content": f"Sadece kullanici {i}"}]}
                 for i in range(20)])
    result = run(tmp_path)
    assert any(f.severity is Severity.BLOCKING for f in result.findings)
    assert not result.blocking, "nothing may block when no purpose was declared"
    assert not result.ctx.blocking_enabled


def test_blocking_activates_only_with_a_target(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [{"messages": [{"role": "user", "content": f"Sadece kullanici {i}"}]}
                 for i in range(20)])
    result = run(tmp_path, target="sft", profile=Profile.SFT)
    assert result.blocking


def test_blocking_uses_declared_target_not_inferred_data_profile(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [{"messages": [{"role": "user", "content": f"Sadece kullanici {i}"}]}
                 for i in range(20)])
    result = run(tmp_path, target="corpus", profile=Profile.SFT)
    assert not result.blocking


def test_minhash_preset_reaches_the_scan_context(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl", [chat("Soru", LONG)])
    result = run(tmp_path, minhash_preset="hf-neardedup")
    assert result.ctx.stats["minhash_preset"] == "hf-neardedup"


def test_every_finding_is_unverified_in_this_build(tmp_path):
    """v0.1 ships no measured effect sizes and must not imply otherwise."""
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat(f"Soru {i}", f"{LONG} {i}") for i in range(30)] * 3)
    for f in run(tmp_path).findings:
        assert f.confidence.value == "unverified"


def test_skipped_checks_name_the_flag_that_unlocks_them(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat(f"Soru {i}", f"{LONG} {i}") for i in range(20)])
    result = run(tmp_path)
    token_skips = [s for s in result.skipped if "model" in s.unlock]
    assert token_skips, "token checks must be skipped with an actionable unlock"
    assert all(s.reason for s in result.skipped)


def test_fingerprint_shape_counts_text_not_bytes_on_disk(tmp_path):
    """Regression: the shape facet reported file bytes and a hardcoded 0 words.

    The fingerprint is the one artifact meant to be compared across datasets.
    `total_bytes` includes JSON syntax, keys and metadata columns, and in UTF-8 a
    Turkish corpus costs more bytes per character than an English one, so a
    shape facet built from bytes varies with language and file format rather than
    with the data. It must come from the normalised text the runner actually saw.
    """
    import json as _json
    import subprocess
    import sys

    d = tmp_path / "d"
    # Turkish diacritics are 2 bytes each in UTF-8 but 1 character.
    text = "çğıöşü " * 40
    write_jsonl(d / "train.jsonl", [chat(f"Soru {i}", text) for i in range(25)])

    subprocess.run(
        [sys.executable, "-m", "dropoutt", "scan", str(tmp_path), "--quiet", "--no-html"],
        check=False, capture_output=True,
    )
    values = _json.loads((tmp_path / ".dropoutt" / "fingerprint.json").read_text(
        encoding="utf-8"))["facets"]["shape"]["values"]

    assert values["total_words"] > 0, "words were hardcoded to 0"
    on_disk = sum(f.stat().st_size for f in d.rglob("*.jsonl"))
    assert values["total_chars"] < on_disk, (
        "character count must exclude JSON syntax and multi-byte overhead; "
        f"got {values['total_chars']} against {on_disk} bytes on disk"
    )


def test_fingerprint_content_hash_depends_on_content_not_only_size(tmp_path):
    from dropoutt.fingerprint import build as build_fingerprint

    left = tmp_path / "left"
    right = tmp_path / "right"
    write_jsonl(left / "d" / "train.jsonl", [chat("Soru", "AAAA")])
    write_jsonl(right / "d" / "train.jsonl", [chat("Soru", "BBBB")])
    assert (
        (left / "d" / "train.jsonl").stat().st_size
        == (right / "d" / "train.jsonl").stat().st_size
    )

    left_result = run(left)
    right_result = run(right)
    left_fp = build_fingerprint(
        left_result.ctx, left_result.findings, total_chars=8, total_words=2
    )
    right_fp = build_fingerprint(
        right_result.ctx, right_result.findings, total_chars=8, total_words=2
    )

    assert left_fp.provenance["content_hash"] != right_fp.provenance["content_hash"]
    assert left_fp.fingerprint_id != right_fp.fingerprint_id


def test_offline_reads_the_cache_instead_of_refusing(monkeypatch, tmp_path):
    """Regression: --offline returned {} without ever consulting the cache.

    The chat template decides how records render, which spans are trainable and
    how many tokens a run costs. Dropping it on a compute node silently changed
    every token number and disabled the loss-mask checks, which defeats the
    whole point of `dropoutt fetch` on a login node.
    """
    import json as _json

    import dropoutt.config as config_mod

    cfg_file = tmp_path / "tokenizer_config.json"
    cfg_file.write_text(_json.dumps({
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "eos_token": "<|im_end|>",
    }), encoding="utf-8")

    calls: list[dict] = []

    def fake_download(repo_id, filename, **kwargs):
        calls.append(kwargs)
        return str(cfg_file)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    cfg = config_mod._load_tokenizer_config("Qwen/Qwen3-8B", offline=True)

    assert calls, "offline gave up before it ever looked in the cache"
    assert calls[0].get("local_files_only") is True, (
        "offline must resolve from the cache without reaching the network"
    )
    assert cfg.get("chat_template"), "a cached template must still be usable offline"


def test_offline_scan_makes_no_network_connection(tmp_path, monkeypatch):
    """--offline is a promise, and the atlas embedder used to break it.

    `scan()` took no offline argument at all, so `_compute_coverage` called
    load_embedder unconditionally and tried to download a ~500 MB model on a
    compute node with no egress. Any connection attempt fails this test.
    """
    import socket

    from dropoutt.runner import scan as run_scan

    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat(f"Soru {i}", f"{LONG} {i}") for i in range(25)])

    attempts: list[object] = []

    def blocked(self, address):
        attempts.append(address)
        raise OSError("network access attempted under --offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)

    from dropoutt.atlas import load_bundled

    run_scan(str(tmp_path), atlas=load_bundled(), offline=True)
    assert not attempts, f"--offline reached the network: {attempts}"


def test_a_private_eval_index_does_not_hide_the_shipped_ones(tmp_path):
    """Regression: index lookup returned one directory or the other, never both.

    `_contamination_dir()` preferred the cache when it held any .idx, so the
    first `index-eval` silently switched off all ten bundled benchmarks.
    """
    from dropoutt.contamination import BenchmarkIndex, load_indices

    shipped, cache = tmp_path / "shipped", tmp_path / "cache"
    shipped.mkdir()
    cache.mkdir()

    for directory, name in ((shipped, "gsm8k"), (cache, "my-holdout")):
        idx = BenchmarkIndex(name=name, n_instances=0)
        idx.add_instance(0, " ".join(f"word{i}" for i in range(40)))
        idx.n_instances = 1
        idx.save(directory / f"{name}.idx")

    merged = load_indices(cache, shipped)
    names = set(merged.benchmarks)
    assert names == {"gsm8k", "my-holdout"}, (
        f"both locations must be searched, got {names}"
    )


# -- generation integrity ------------------------------------------------
#
# Calibrated against a real 6,097-record Turkish generation run. Each check gets
# a fixture built to trip it and one built not to, and the "not to" case matters
# more here than usual: two of these checks are meant to stay silent on that
# corpus, and a check that fires on everything is worse than no check.


def test_content_in_a_key_the_layout_never_reads_is_reported(tmp_path):
    """The orphaned assistant turn: 26 records lost their answer this way."""
    records = [chat(f"Soru {i} nedir acaba diye merak ediyorum", f"{LONG} {i}")
               for i in range(60)]
    for r in records[:12]:
        r["assistant"] = [{"role": "assistant", "content": f"{LONG} gercek cevap burada"}]
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    found = findings_by_id(run(tmp_path))
    assert "T0-SCHEMA-005" in found
    assert found["T0-SCHEMA-005"].count == 12
    assert "assistant" in found["T0-SCHEMA-005"].data["keys"]


def test_bookkeeping_columns_are_not_mistaken_for_lost_content(tmp_path):
    """id, source and language ride along on purpose and must stay quiet."""
    records = []
    for i in range(60):
        r = chat(f"Soru {i} nedir acaba diye merak ediyorum", f"{LONG} {i}")
        r.update({"id": f"rec-{i}", "language": "tr",
                  "source": "a fairly long provenance string naming the origin dataset"})
        records.append(r)
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    assert "T0-SCHEMA-005" not in findings_by_id(run(tmp_path))


def test_a_mixture_of_reasoning_and_plain_responses_is_reported(tmp_path):
    records = [chat(f"Soru {i} nedir acaba diye merak ediyorum", f"{LONG} {i}")
               for i in range(400)]
    for r in records[:48]:  # 12%, the share measured on the real corpus
        r["messages"][1]["content"] = f"<think>once dusunelim</think>\n\n{LONG}"
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    found = findings_by_id(run(tmp_path))
    assert "T0-REASON-001" in found
    assert 0.10 < found["T0-REASON-001"].data["share_with_reasoning"] < 0.14


def test_reasoning_on_every_record_is_a_decision_not_a_finding(tmp_path):
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        {"messages": [{"role": "user", "content": f"Soru {i} nedir acaba diye"},
                      {"role": "assistant", "content": f"<think>dusun</think>\n\n{LONG} {i}"}]}
        for i in range(400)
    ])
    assert "T0-REASON-001" not in findings_by_id(run(tmp_path))


def test_responses_piling_up_at_one_length_are_read_as_a_cap(tmp_path):
    body = "a" * 900
    records = [chat(f"Soru {i} nedir acaba diye merak ediyorum", body[:900 - i % 3])
               for i in range(300)]
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    found = findings_by_id(run(tmp_path))
    assert "T0-TRUNC-002" in found
    assert found["T0-TRUNC-002"].data["pileup_share"] > 0.9


def test_varied_response_lengths_are_not_read_as_a_cap(tmp_path):
    """Silent on the real corpus, whose most common exact length held 1.2%."""
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat(f"Soru {i} nedir acaba diye merak ediyorum", LONG * (1 + i % 17))
        for i in range(400)
    ])
    assert "T0-TRUNC-002" not in findings_by_id(run(tmp_path))


def test_one_prompt_with_two_different_answers_is_a_contradiction(tmp_path):
    records = [chat(f"Soru {i} nedir acaba diye merak ediyorum", f"{LONG} {i}")
               for i in range(60)]
    records.append(chat("Soru 0 nedir acaba diye merak ediyorum", f"{LONG} bambaska bir cevap"))
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    found = findings_by_id(run(tmp_path))
    assert "T1-DUP-002" in found
    assert found["T1-DUP-002"].count == 1


def test_an_exactly_repeated_record_is_redundancy_not_contradiction(tmp_path):
    """T0-DUP-001 owns this case; the contradiction check must not double-report."""
    records = [chat(f"Soru {i} nedir acaba diye merak ediyorum", f"{LONG} {i}")
               for i in range(60)]
    records.append(records[0].copy())
    write_jsonl(tmp_path / "d" / "train.jsonl", records)
    found = findings_by_id(run(tmp_path))
    assert "T0-DUP-001" in found
    assert "T1-DUP-002" not in found


# -- corpus quality filters ----------------------------------------------


def _corpus(tmp_path, texts: list[str]) -> None:
    write_jsonl(tmp_path / "c" / "docs.jsonl", [{"text": t} for t in texts])


#: Nine distinct sentences. An earlier version of this fixture repeated three
#: lines three times and tripped the duplicated-line filter, which was that
#: filter working correctly on a fixture that was wrong.
PARAGRAPH = "\n".join(
    f"Bu {n}. satir yeterince uzun bir cumledir ve dogru noktalama ile sona erer."
    for n in range(1, 10)
)


def test_navigation_shaped_documents_are_reported(tmp_path):
    """Lines that do not end in punctuation: the shape of an extracted menu."""
    menu = "\n".join(f"Kategori {i} baglantisi" for i in range(40))
    _corpus(tmp_path, [menu] * 60 + [PARAGRAPH] * 60)
    found = findings_by_id(run(tmp_path, profile=Profile.CORPUS))
    assert "T0-QUAL-001" in found
    assert found["T0-QUAL-001"].count == 60


def test_short_line_documents_are_reported(tmp_path):
    listy = "\n".join(f"Madde {i}." for i in range(60))
    _corpus(tmp_path, [listy] * 60 + [PARAGRAPH] * 60)
    found = findings_by_id(run(tmp_path, profile=Profile.CORPUS))
    assert "T0-QUAL-002" in found


def test_documents_repeating_their_own_lines_are_reported(tmp_path):
    repeated = ("Tekrar eden bir alt bilgi satiri burada yer aliyor ve devam ediyor.\n" * 12)
    _corpus(tmp_path, [repeated] * 60 + [PARAGRAPH] * 60)
    found = findings_by_id(run(tmp_path, profile=Profile.CORPUS))
    assert "T0-QUAL-003" in found


def test_ordinary_prose_trips_none_of_the_corpus_filters(tmp_path):
    _corpus(tmp_path, [PARAGRAPH + f" Belge {i}." for i in range(120)])
    ids = set(findings_by_id(run(tmp_path, profile=Profile.CORPUS)))
    assert not ids & {"T0-QUAL-001", "T0-QUAL-002", "T0-QUAL-003"}


def test_corpus_filters_do_not_run_against_conversational_data(tmp_path):
    """They measure line shape, which says nothing about a chat record."""
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat(f"Soru {i} nedir acaba", f"{LONG} {i}") for i in range(120)
    ])
    ids = set(findings_by_id(run(tmp_path, profile=Profile.SFT)))
    assert not ids & {"T0-QUAL-001", "T0-QUAL-002", "T0-QUAL-003"}


def test_a_multiple_choice_answer_is_not_a_prompt_copy(tmp_path):
    """All 38 prompt-copy hits on a real corpus were this shape or extractive QA.

    The answer is necessarily a substring of a prompt that listed the options.
    That is correct supervision, not degeneracy.
    """
    options = ("A) birinci secenek metni B) ikinci secenek metni "
               "C) ucuncu secenek metni D) dorduncu secenek metni")
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat(f"Soru {i} icin dogru siki sec. {options}", "C) ucuncu secenek metni")
        for i in range(60)
    ])
    found = findings_by_id(run(tmp_path))
    assert found.get("T0-DEGEN-001") is None or found["T0-DEGEN-001"].data["copying"] == 0


def test_extractive_qa_is_not_a_prompt_copy(tmp_path):
    passage = ("Izlanda'nin baskenti Reykjavik jeotermal isitma kullanir. "
               "Donus suyu sicakligi genellikle 35 derece civarindadir. "
               "Bu su kaldirim buz cozme sistemlerinde yeniden kullanilir. ") * 3
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat(f"Metne dayanarak cevapla, alintila. Metin: {passage} Soru {i}?",
             "Donus suyu sicakligi genellikle 35 derece civarindadir.")
        for i in range(60)
    ])
    found = findings_by_id(run(tmp_path))
    assert found.get("T0-DEGEN-001") is None or found["T0-DEGEN-001"].data["copying"] == 0


def test_an_answer_that_is_the_whole_prompt_is_still_a_copy(tmp_path):
    """The case the check was written for has to keep firing."""
    prompt = "Bu cumleyi aynen tekrar et ve baska hicbir sey ekleme lutfen efendim"
    write_jsonl(tmp_path / "d" / "train.jsonl",
                [chat(f"{prompt} {i}", f"{prompt} {i}") for i in range(60)])
    found = findings_by_id(run(tmp_path))
    assert found["T0-DEGEN-001"].data["copying"] == 60


def test_a_classification_set_of_short_labels_is_not_a_length_cap(tmp_path):
    """Every answer is one of three labels: 100% in the top bucket, nothing beneath.

    Both the share test and the wall test pass, and the dataset is fine. Nobody
    sets max_tokens to twelve characters.
    """
    labels = ["olumlu", "olumsuz", "notr"]
    write_jsonl(tmp_path / "d" / "train.jsonl", [
        chat(f"Bu yorumun duygusunu siniflandir numara {i} icin lutfen", labels[i % 3])
        for i in range(400)
    ])
    assert "T0-TRUNC-002" not in findings_by_id(run(tmp_path))
