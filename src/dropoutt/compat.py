"""Optional-dependency shims.

Every accelerator in this package is optional. The rule from the design is that
the core scan must run with no compiled dependency other than ``tokenizers``,
because that is the only one still shipping ``manylinux_2_17`` and therefore the
only one that reliably installs on older cluster images.

Each shim exposes a ``HAVE_*`` flag plus a working fallback that produces the
same answers, only slower or less accurate. When a fallback is less accurate
rather than merely slower, the scan records that fact so the report can say so
instead of quietly degrading.
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
# MinHash. rensa is a Rust implementation; the fallback in minhash.py is a
# numpy-vectorised implementation using the same permutation scheme and banding,
# so cluster membership agrees. It is slower, not different.
# --------------------------------------------------------------------------
try:
    import rensa as _rensa  # noqa: F401

    HAVE_RENSA = True
except ImportError:
    HAVE_RENSA = False


# --------------------------------------------------------------------------
# Language identification. The fallback is a small character-profile detector
# that covers far fewer languages and is materially less accurate. Because that
# is a quality difference rather than a speed difference, langid.py marks its
# results as low-trust and the report states which backend produced them.
# --------------------------------------------------------------------------
try:
    from ftlangdetect import detect as _ftdetect  # noqa: F401

    HAVE_FASTTEXT_LID = True
except ImportError:
    HAVE_FASTTEXT_LID = False


# --------------------------------------------------------------------------
# Atlas embeddings. No fallback: without an embedding model there is no honest
# way to place records on the atlas, so coverage is reported as skipped.
# --------------------------------------------------------------------------
try:
    from model2vec import StaticModel as _StaticModel  # noqa: F401

    HAVE_MODEL2VEC = True
except ImportError:
    HAVE_MODEL2VEC = False


def capability_report() -> dict[str, dict[str, Any]]:
    """What is installed, and what the absence of each thing costs."""
    return {
        "orjson": {
            "available": HAVE_ORJSON,
            "impact": "speed only",
            "install": "pip install 'dropoutt[fast]'",
        },
        "tokenizers": {
            "available": HAVE_TOKENIZERS,
            "impact": "exact token counts, chat template render, loss mask checks",
            "install": "pip install 'dropoutt[tokenizer]'",
        },
        "pyarrow": {
            "available": HAVE_PYARROW,
            "impact": "reading .parquet files",
            "install": "pip install 'dropoutt[parquet]'",
        },
        "rensa": {
            "available": HAVE_RENSA,
            "impact": "speed only, identical clusters",
            "install": "pip install 'dropoutt[fast]'",
        },
        "fasttext-langdetect": {
            "available": HAVE_FASTTEXT_LID,
            "impact": "accurate language identification across 176 languages",
            "install": "pip install 'dropoutt[lid]'",
        },
        "model2vec": {
            "available": HAVE_MODEL2VEC,
            "impact": "atlas coverage",
            "install": "pip install 'dropoutt[atlas]'",
        },
    }
