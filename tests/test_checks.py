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
