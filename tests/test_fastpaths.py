"""The fast paths must agree with the definitions they replaced.

Every optimisation in the scan is only allowed because it computes the same
answer. Each one below has a slow, obvious form beside it, and these tests are
what stop the two drifting apart — particularly the contamination hashes, which
are frozen by files we ship and cannot be recomputed from anything.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np
import pytest

from dropoutt.registry_data import detect_template_family, template_index
from dropoutt.sniff import _scan_char_by_char, scan_json_spans
from dropoutt.textutil import (
    _eval_ngram_hashes_slow,
    eval_ngram_hashes,
    non_whitespace_prefix,
    normalize_ws,
)

WORDS = ["veri", "kümesi", "için", "eğitim", "modeli", "değil", "mi", "日本語", "δοκιμή", "тест", "a"]
JSON_CHARS = [*'{}"\\ abc\n:,[]', '\\"', '\\\\', "```json", '{"a": "b}c"}']


def _random_words(rng: random.Random, upper: int = 40) -> list[str]:
    return [rng.choice(WORDS) for _ in range(rng.randint(0, upper))]


def _random_jsonish(rng: random.Random) -> str:
    return "".join(rng.choice(JSON_CHARS) for _ in range(rng.randint(0, 80)))


def test_contamination_hashes_match_the_definition_byte_for_byte():
    """The shipped .idx files are tables of exactly these numbers.

    There is no benchmark text left anywhere to recompute them from, so a
    change here silently stops matching every index we ship.
    """
    rng = random.Random(3)
    for _ in range(400):
        words = _random_words(rng)
        if len(words) < 8:
            continue
        assert np.array_equal(
            eval_ngram_hashes(words), _eval_ngram_hashes_slow(words, 8)
        ), words


def test_a_word_containing_a_space_falls_back_rather_than_misindexing():
    """The fast path indexes one joined buffer by counting separators."""
    words = ["one", "two three", "four", "five", "six", "seven", "eight", "nine"]
    assert np.array_equal(eval_ngram_hashes(words), _eval_ngram_hashes_slow(words, 8))


def test_whitespace_collapse_matches_the_regex_it_replaced():
    """``str.split`` and ``re``'s ``\\s`` classify every codepoint identically."""
    pattern = re.compile(r"\s+")
    disagree = [
        cp for cp in range(0x110000)
        if (pattern.fullmatch(chr(cp)) is not None) != chr(cp).isspace()
    ]
    assert disagree == []
    for text in ("  a\t\tb \n c  ", "", "   ", "a", "a b c"):
        assert normalize_ws(text) == pattern.sub(" ", text).strip()


def test_non_whitespace_prefix_counts_what_a_generator_would():
    for text in ("", " ", "abc", " a b  c ", "日本 語\tx", "\n\n\n"):
        prefix = non_whitespace_prefix(text)
        assert len(prefix) == len(text) + 1
        for i in range(len(text) + 1):
            assert int(prefix[i]) == sum(1 for c in text[:i] if not c.isspace())


def test_the_vectorised_span_scanner_matches_the_character_loop():
    rng = random.Random(11)
    for _ in range(2000):
        text = _random_jsonish(rng)
        assert scan_json_spans(text) == _scan_char_by_char(text), repr(text)


@pytest.mark.parametrize("text", [
    '{"messages":[{"role":"user","content":"say \\"hi\\" }{"}]}\n{"a":1}',
    'preamble\n```json\n{"x": {"y": 2}}\n```\ntail {"z": 3}',
    '{"t":"a\\\\"}{"t":"b"}',
    "",
    '{"unbalanced": ',
])
def test_real_framings_scan_identically(text):
    assert scan_json_spans(text) == _scan_char_by_char(text)


def test_resuming_from_the_reported_offset_reproduces_a_whole_scan():
    """The incremental reader drops everything before this offset.

    Getting it wrong loses records at a chunk boundary, which is the one bug
    in this file that a corpus would never make obvious.
    """
    rng = random.Random(23)
    for _ in range(1000):
        text = _random_jsonish(rng)
        cut = rng.randint(0, len(text))
        head, tail = text[:cut], text[cut:]
        found, resume = scan_json_spans(head)
        rest, _ = scan_json_spans(head[resume:] + tail)
        consumed = found[-1][1] if found else 0
        combined = found + [
            (s + resume, e + resume) for s, e in rest if s + resume >= consumed
        ]
        assert combined == scan_json_spans(text)[0], (text, cut)


def test_the_template_gate_never_hides_a_family_that_is_present():
    def unguarded(text: str) -> list[tuple[str, int]]:
        hits = [
            (family["id"], sum(1 for d in family["delimiters"] if d in text))
            for family in template_index()
        ]
        return sorted(((f, n) for f, n in hits if n), key=lambda kv: -kv[1])

    # Every delimiter we ship, one at a time, plus text that has none.
    texts = [d for family in template_index() for d in family["delimiters"]]
    texts += [
        "ordinary English prose, with punctuation and no delimiters.",
        "A Sentence With Uppercase U And A But No Templates.",
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        "<|im_start|>user\nhi<|im_end|>",
        "### Instruction:\ndo it\n### Response:\nok",
        "USER: hi\nASSISTANT: hello",
    ]
    for text in texts:
        assert detect_template_family(text) == unguarded(text), text
