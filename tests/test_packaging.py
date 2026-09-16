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
ATLAS_DATA = Path(__file__).resolve().parents[1] / "src" / "dropoutt" / "data" / "atlas"


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


def test_the_atlas_data_directory_holds_only_what_ships():
    """Every file here rides in the wheel, so a stray one is a shipped one.

    The builder and the calibration tool write their scratch beside the product
    (`atlas-v3.npz.tmp.npz`, `atlas-v3.npz.patching.npz`). One of those rode
    inside a wheel and made it 45.9 MB, because the exclude pattern named only
    `*.tmp.npz`. The pattern is broader now, and this asserts the directory
    itself rather than trusting a glob.
    """
    import re

    allowed = re.compile(
        r"^atlas-v\d+(-lite)?\.npz$|^atlas-v\d+-SHA256SUMS$|^atlas-v\d+-release-notes\.json$"
    )
    names = sorted(p.name for p in ATLAS_DATA.iterdir() if p.name != ".DS_Store")
    stray = [name for name in names if not allowed.match(name)]
    assert not stray, f"not a product and would ship in the wheel: {stray}"
    assert {"atlas-v3.npz", "atlas-v2.npz", "atlas-v2-lite.npz", "atlas-v1-lite.npz"} <= set(names)


def test_shipped_maps_match_their_stamped_checksums():
    """The SHA256SUMS files are the release record; the bytes must agree."""
    import hashlib

    for sums in sorted(ATLAS_DATA.glob("atlas-v*-SHA256SUMS")):
        for line in sums.read_text(encoding="utf-8").splitlines():
            digest, _, name = line.strip().partition("  ")
            target = ATLAS_DATA / name
            assert target.exists(), f"{sums.name} names {name}, which is not bundled"
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            assert actual == digest, f"{name} does not match {sums.name}"
