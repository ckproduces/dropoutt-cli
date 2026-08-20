"""The install must not need a compiler, and there must be nothing to choose.

Both properties came from the same support ticket. A user on Windows and
CPython 3.14 ran `pip install dropoutt` and was told to install Microsoft Visual
C++ Build Tools, because a dependency published no wheel for that interpreter
and pip fell through to building it from source. The dependency was there to
support one extra, and the extras existed so people could avoid installing
things — which meant most installs were missing something.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.10
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project() -> dict:
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh)["project"]


def test_there_are_no_feature_extras():
    """`pip install dropoutt` installs everything. There is nothing else to type.

    Every extra was a way to end up with a dropoutt that silently could not read
    your Parquet or identify your languages, and `dropoutt doctor` existed to
    explain which one you were missing.
    """
    extras = set(_project().get("optional-dependencies", {}))
    assert extras <= {"dev"}, (
        f"feature extras are back: {sorted(extras - {'dev'})}. "
        "One install has to bring everything."
    )


def test_no_dependency_requires_a_compiler():
    """Every runtime dependency must ship wheels for every supported target.

    Asserted by name rather than by resolving, because a resolver needs a
    network. `fasttext-langdetect` is the specific one that broke: it pulls
    `fasttext-predict`, which publishes no wheel for CPython 3.14, so pip
    compiled it — and on Windows that means MSVC.

    Anything added here must resolve under:

        uv pip compile pyproject.toml --python-version 3.14 \\
            --python-platform windows --only-binary :all:
    """
    forbidden = {
        "fasttext", "fasttext-predict", "fasttext-langdetect", "fasttext-wheel",
        # Source-only or partially-wheeled packages this project has considered.
        "pyicu", "python-levenshtein", "kenlm", "sentencepiece-cpp",
    }
    names = {
        _requirement_name(spec)
        for spec in _project()["dependencies"]
    }
    assert not (names & forbidden), (
        f"{sorted(names & forbidden)} needs a C or C++ toolchain on at least one "
        "supported interpreter, and an install that can fail is not an install"
    )


def test_language_identification_is_a_pure_python_dependency():
    names = {_requirement_name(spec) for spec in _project()["dependencies"]}
    assert "py3langid" in names


def test_every_declared_dependency_is_importable_here():
    """The dev environment is a normal install, so nothing may be missing."""
    from dropoutt.compat import capability_report

    missing = [name for name, info in capability_report().items() if not info["available"]]
    assert not missing, f"not importable: {missing}"


def _requirement_name(spec: str) -> str:
    head = spec.split(";", maxsplit=1)[0].strip()
    for separator in ("[", ">", "<", "=", "!", "~", " "):
        head = head.split(separator)[0]
    return head.strip().lower().replace("_", "-")
