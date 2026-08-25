"""The scan pass runs in forked workers, so it may not touch a BLAS ``gemm``.

On macOS numpy links against Accelerate, whose ``gemm`` dispatches through
libdispatch. A process that has called it cannot be forked and then call it
again: the child dies with SIGSEGV before raising anything Python can catch, so
the pool reports ``BrokenProcessPool`` and the scan silently falls back to one
core. That is exactly what happened when the language classifier was first
vectorised with a dense matrix multiply — a scan that had been using thirteen
workers quietly started using one, and got slower than before.

``gemv``, SciPy's sparse multiply and ``einsum`` are all unaffected, so the fix
was to phrase the multiply sparsely. These tests keep it that way, and two
details are what make them able to:

**The warm-up has to actually poison.** Accelerate only routes a ``gemm``
through libdispatch above a size threshold; a 128x512 warm-up never engages it,
and a test warmed that way passes with the exact dense multiply it exists to
forbid. The warm-up here is 2048x2048, verified to arm the failure on the
machine that found it.

**The scenario runs in a fresh subprocess.** A genuinely poisoning warm-up
would poison the pytest process itself, and every later test that forks — the
parallel-scan tests included — would inherit the damage. Each test below spawns
a clean interpreter that mirrors production exactly: build the detector in the
parent (``prime_worker`` guarantees this before the pool forks), run the large
multiply, fork, classify in the child with the inherited detector. Building
the detector *inside* the forked child instead would die in the ObjC runtime
on macOS even with correct sparse code, which is not the regression guarded
here.

On Linux and Windows these pass trivially; on macOS they are the only thing
standing between a dense multiply and a scan that loses its pool.
"""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
import textwrap

import pytest

from dropoutt.compat import HAVE_PY3LANGID

pytestmark = pytest.mark.skipif(
    "fork" not in mp.get_all_start_methods(), reason="no fork on this platform"
)


def _run_forked_scenario(child_body: str) -> None:
    """Run warm-up → fork → ``child_body`` in a fresh interpreter."""
    script = textwrap.dedent(
        """
        import multiprocessing as mp
        import sys

        import numpy as np

        # At least one full classification batch (ngram_langid.BATCH is 256):
        # a smaller corpus makes the child's would-be dense multiply too small
        # to route through Accelerate's threaded path, and a regression to a
        # dense gemm would survive the fork and pass. Verified at this size:
        # the dense regression dies with SIGSEGV, the sparse path lives.
        TEXTS = [
            "The quick brown fox jumps over the lazy dog and keeps on running today.",
            "Merhaba dünya, bugün hava çok güzel ve dışarı çıkmak istiyorum bence.",
            "Сегодня прекрасная погода, и я собираюсь пойти гулять в парке снова.",
        ] * 128

        PARENT_STATE = {}

        def child(queue):
            try:
        %s
            except BaseException as exc:
                queue.put(("error", repr(exc)))

        def main():
        %s
            # Large enough to engage Accelerate's threaded gemm path; smaller
            # multiplies do not poison the fork and guard nothing.
            big = np.ones((2048, 2048), dtype=np.float32)
            assert float((big @ big).sum()) > 0

            context = mp.get_context("fork")
            queue = context.Queue()
            worker = context.Process(target=child, args=(queue,))
            worker.start()
            worker.join(120)
            assert worker.exitcode == 0, (
                f"forked child died with {worker.exitcode}; something on the "
                "scan path reached a BLAS gemm"
            )
            assert not queue.empty(), "the child produced no result"
            status, payload = queue.get()
            assert status == "ok", payload
            print("RESULT", payload)

        main()
        """
    )
    parent_setup, child_code = child_body.split("# ---fork---")
    script = script % (
        textwrap.indent(textwrap.dedent(child_code).strip(), " " * 8),
        textwrap.indent(textwrap.dedent(parent_setup).strip(), " " * 4),
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, f"stdout: {done.stdout}\nstderr: {done.stderr}"
    assert "RESULT" in done.stdout, done.stdout


@pytest.mark.skipif(not HAVE_PY3LANGID, reason="py3langid not installed")
def test_language_identification_survives_a_fork_after_a_dense_multiply():
    _run_forked_scenario(
        """
        from dropoutt.langid import LanguageDetector

        detector = LanguageDetector()
        first = [r.lang for r in detector.detect_many(TEXTS[:3])]
        assert first == ["en", "tr", "ru"], first
        PARENT_STATE["detector"] = detector
        # ---fork---
        detector = PARENT_STATE["detector"]
        results = detector.detect_many(TEXTS)
        labels = [r.lang for r in results[:3]]
        assert labels == ["en", "tr", "ru"], labels
        queue.put(("ok", labels))
        """
    )


def test_the_sparse_multiply_the_classifier_uses_is_itself_fork_safe():
    """Narrow the guard to the operation, so a regression names its own cause."""
    _run_forked_scenario(
        """
        from scipy.sparse import csr_matrix

        PARENT_STATE["sparse"] = csr_matrix(np.eye(512, dtype=np.float32))
        PARENT_STATE["dense"] = np.ones((512, 97), dtype=np.float32)
        # ---fork---
        result = PARENT_STATE["sparse"] @ PARENT_STATE["dense"]
        queue.put(("ok", float(result.sum())))
        """
    )
