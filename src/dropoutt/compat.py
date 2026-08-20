"""Import shims for things that should always be here, and a floor for when they are not.

Every dependency in this list is a hard requirement of the package: `pip install
dropoutt` installs all of them, there are no extras, and on a normal install
every ``HAVE_*`` flag below is True. The extras were removed in 1.1 because
choosing between them was a way to end up with a dropoutt that silently could
not read your Parquet.

The shims stayed. A package can still be *reached* in a state where one of these
is missing — a vendored copy on an air-gapped node, a partially-restored image,
a site-packages someone pruned to fit a container — and the failure mode that
matters is a scan that crashes on record four million rather than one that says
which check it could not run. So each shim exposes a flag plus a fallback that
produces the same answers, slower or less accurate, and when the fallback is
less accurate rather than merely slower the scan records that so the report says
so instead of quietly degrading.
"""

from __future__ import annotations

import json as _stdlib_json
from typing import Any

# --------------------------------------------------------------------------
# JSON. orjson is ~2-4x faster on the deserialize-heavy JSONL path.
# --------------------------------------------------------------------------
try:
    import orjson as _orjson

    HAVE_ORJSON = True

    def json_loads(data: bytes | str) -> Any:
        return _orjson.loads(data)

    def json_dumps(obj: Any, *, indent: bool = False) -> str:
        opts = _orjson.OPT_INDENT_2 if indent else 0
        return _orjson.dumps(obj, option=opts).decode("utf-8")

except ImportError:  # pragma: no cover - exercised on minimal installs
    HAVE_ORJSON = False

    def json_loads(data: bytes | str) -> Any:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return _stdlib_json.loads(data)

    def json_dumps(obj: Any, *, indent: bool = False) -> str:
        return _stdlib_json.dumps(obj, ensure_ascii=False, indent=2 if indent else None)


# --------------------------------------------------------------------------
# Tokenizers. Without this we cannot count tokens exactly or render templates,
# so every token-dependent check is reported as skipped rather than guessed at.
# --------------------------------------------------------------------------
try:
    from tokenizers import Tokenizer as _Tokenizer  # noqa: F401

    HAVE_TOKENIZERS = True
except ImportError:
    HAVE_TOKENIZERS = False

try:
    import huggingface_hub as _hf  # noqa: F401

    HAVE_HF_HUB = True
except ImportError:
    HAVE_HF_HUB = False


# --------------------------------------------------------------------------
# Parquet. By far the heaviest optional dependency, so it is never in core.
# --------------------------------------------------------------------------
try:
    import pyarrow.parquet as _pq  # noqa: F401

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False


# --------------------------------------------------------------------------
# Zstandard. gzip, bzip2 and xz are in the standard library; zstd is optional.
# --------------------------------------------------------------------------
try:
    import zstandard as _zstandard  # noqa: F401

    HAVE_ZSTANDARD = True
except ImportError:
    HAVE_ZSTANDARD = False


# --------------------------------------------------------------------------
# Language identification. The fallback is a small character-profile detector
# that covers far fewer languages and is materially less accurate. Because that
# is a quality difference rather than a speed difference, langid.py marks its
# results as low-trust and the report states which backend produced them.
#
# py3langid rather than fasttext-langdetect: see the module docstring in
# langid.py. The short version is that fastText has no wheel for every
# interpreter this package supports, and a dependency that compiles at install
# time is a dependency that fails at install time.
# --------------------------------------------------------------------------
try:
    import py3langid as _py3langid  # noqa: F401

    HAVE_PY3LANGID = True
except ImportError:
    HAVE_PY3LANGID = False


# --------------------------------------------------------------------------
# Atlas embeddings. No fallback: without an embedding model there is no honest
# way to place records on the atlas, so coverage is reported as skipped.
# --------------------------------------------------------------------------
try:
    from model2vec import StaticModel as _StaticModel  # noqa: F401

    HAVE_MODEL2VEC = True
except ImportError:
    HAVE_MODEL2VEC = False


#: What every dependency is for, and what its absence costs. All of them are
#: required, so on a normal install every row here reads "yes" — the table is
#: for diagnosing the abnormal one. ``install`` is the same command for every
#: row because there is only one command.
REINSTALL = "pip install --force-reinstall dropoutt"


def capability_report() -> dict[str, dict[str, Any]]:
    """What is importable, and what the absence of each thing costs."""
    return {
        "orjson": {
            "available": HAVE_ORJSON,
            "impact": "speed only",
            "install": REINSTALL,
        },
        "tokenizers": {
            "available": HAVE_TOKENIZERS,
            "impact": "exact token counts, chat template render, loss mask checks",
            "install": REINSTALL,
        },
        "pyarrow": {
            "available": HAVE_PYARROW,
            "impact": "reading .parquet, .arrow, .feather, and .orc files",
            "install": REINSTALL,
        },
        "zstandard": {
            "available": HAVE_ZSTANDARD,
            "impact": "reading .zst-compressed files",
            "install": REINSTALL,
        },
        "py3langid": {
            "available": HAVE_PY3LANGID,
            "impact": "language identification across 97 languages",
            "install": REINSTALL,
        },
        "model2vec": {
            "available": HAVE_MODEL2VEC,
            "impact": "atlas coverage",
            "install": REINSTALL,
        },
    }
