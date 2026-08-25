"""The vectorised classifier has to be the library's classifier, not a version of it.

py3langid's ``instance2fv`` is a Python loop over every byte of the text.
``dropoutt.ngram_langid`` replaces it with a bag-of-n-grams count derived from
the same automaton, which is only legitimate if the derivation is exact — so
these tests check the derivation itself, the feature vectors it produces, and
the labels that come out the far end, against the library on real text.
"""

from __future__ import annotations

import numpy as np
import pytest

from dropoutt.compat import HAVE_PY3LANGID
from dropoutt.langid import LanguageDetector, dominant_script, dominant_scripts
from dropoutt.ngram_langid import (
    NgramModel,
    patterns_from_automaton,
    state_strings,
    verify_reconstruction,
)

pytestmark = pytest.mark.skipif(not HAVE_PY3LANGID, reason="py3langid not installed")

SAMPLES = [
    "The quick brown fox jumps over the lazy dog, again and again, tirelessly.",
    "Merhaba dünya, bugün hava çok güzel ve ben dışarıya çıkmak istiyorum.",
    "Die Wirtschaft wächst langsamer als erwartet, sagen die Ökonomen heute.",
    "Сегодня прекрасная погода, и я собираюсь пойти гулять в парке.",
    "今日はとても良い天気ですね。散歩に行きましょう、そして写真を撮ります。",
    "안녕하세요, 만나서 반갑습니다. 오늘 날씨가 정말 좋네요.",
    "مرحبا بالعالم، الطقس جميل اليوم وأريد الخروج للتنزه في الحديقة.",
    "El rápido zorro marrón salta sobre el perro perezoso una y otra vez.",
    "",
    "a",
    "xxx",
    "   \n\t  ",
    "1234567890 !@#$%^&*() 1234567890",
    "Ceci est une phrase française avec des accents: é è ê ë à ù ô.",
    "混合 text with 多种 languages mixed together in one single record.",
]


@pytest.fixture(scope="module")
def library():
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    return LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


@pytest.fixture(scope="module")
def model(library):
    return NgramModel(library)


def test_every_state_output_is_reproduced_by_the_recovered_patterns(library):
    """The proof, run exhaustively over the automaton rather than sampled.

    If each state emits exactly the features whose recovered pattern is a suffix
    of that state's string, then the loop and the count agree at every position
    of every input — which is why this is checked, and why the model refuses to
    build when it fails.
    """
    moves = np.asarray(library.tk_nextmove).reshape(-1, 256)
    patterns = patterns_from_automaton(moves, library.tk_output, library.nb_numfeats)

    verify_reconstruction(moves, library.tk_output, patterns)

    assert len(patterns) == library.nb_numfeats
    assert len(set(patterns)) == len(patterns), "two features share a byte string"
    assert min(len(p) for p in patterns) >= 1


def test_state_strings_are_reached_in_as_many_steps_as_they_have_bytes(library):
    moves = np.asarray(library.tk_nextmove).reshape(-1, 256)
    strings = state_strings(moves)

    assert strings[0] == b""
    for state, text in enumerate(strings):
        walked = 0
        for byte in text:
            walked = int(moves[walked][byte])
        assert walked == state, f"state {state} is not reached by its own string"


def test_feature_vectors_match_the_library_loop(library, model):
    for text in SAMPLES:
        expected = library.instance2fv(text)
        actual = model.counts([text.encode("utf-8", "surrogatepass")])[0]
        assert np.array_equal(expected, actual), text[:40]


def test_a_batch_counts_each_record_separately(library, model):
    """No n-gram may straddle two records inside the concatenated buffer."""
    buffers = [t.encode("utf-8", "surrogatepass") for t in SAMPLES]
    batched = model.counts(buffers)

    for row, text in enumerate(SAMPLES):
        assert np.array_equal(batched[row], library.instance2fv(text)), text[:40]


def test_labels_and_probabilities_match_the_library(library, model):
    labels, scores = model.classify_many(SAMPLES)
    for i, text in enumerate(SAMPLES):
        expected_label, expected_score = library.classify(text)
        assert labels[i] == expected_label, text[:40]
        assert scores[i] == pytest.approx(float(expected_score), abs=1e-4)


def test_detect_many_agrees_with_detect_record_by_record():
    detector = LanguageDetector()
    batched = detector.detect_many(SAMPLES)
    one_at_a_time = [detector.detect(text) for text in SAMPLES]

    assert len(batched) == len(SAMPLES)
    for batch, single in zip(batched, one_at_a_time, strict=True):
        assert batch.lang == single.lang
        assert batch.script == single.script
        assert batch.confidence == pytest.approx(single.confidence, abs=1e-9)


def test_dominant_scripts_agrees_with_the_per_record_form():
    assert dominant_scripts(SAMPLES) == [dominant_script(t) for t in SAMPLES]
    assert dominant_scripts([]) == []


def test_an_empty_batch_classifies_to_nothing(model):
    labels, scores = model.classify_many([])
    assert labels == []
    assert scores.size == 0


def test_a_corrupted_failure_transition_is_rejected():
    """The load-time proof must reject a table whose failure links are wrong.

    Checking per-state output sets alone accepts this table: the outputs match
    the recovered patterns, but delta("ab", 'a') pointing at the root instead
    of state "a" makes the automaton walk of "abab" emit the feature once where
    the bag-of-n-grams counts it twice. The transition half of the check is
    what catches it.
    """
    a, b = ord("a"), ord("b")
    moves = np.zeros((3, 256), dtype=np.int64)
    moves[0, a] = 1
    moves[1, a] = 1
    moves[1, b] = 2
    moves[2, a] = 1
    outputs = {2: [0]}
    patterns = patterns_from_automaton(moves, outputs, 1)

    verify_reconstruction(moves, outputs, patterns)  # the honest table passes

    corrupted = moves.copy()
    corrupted[2, a] = 0
    with pytest.raises(ValueError, match="longest-suffix"):
        verify_reconstruction(corrupted, outputs, patterns)


def test_the_model_needs_scipy_at_construction_not_first_use():
    """A missing scipy must fail construction, so langid falls back to the library.

    If the import waited until the first batch, the model would build fine,
    langid's fallback would never trigger, and every classification would be
    swallowed into "unknown" for the whole corpus.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import builtins
        real_import = builtins.__import__

        def no_scipy(name, *args, **kwargs):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy blocked for the test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_scipy

        from dropoutt.langid import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect(
            "The quick brown fox jumps over the lazy dog, again and again."
        )
        assert result.lang == "en", result
        print("FELL BACK", result.lang)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, f"stdout: {done.stdout}\nstderr: {done.stderr}"
    assert "FELL BACK en" in done.stdout
