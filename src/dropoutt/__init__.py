"""dropoutt: pre-flight checks and comparable fingerprints for LLM training datasets.

Two objects, one loop.

A **check** is a test run against a training dataset before a training run. It is
deterministic, it names the fix, and it states what it could not verify.

A **fingerprint** is a fixed-schema description of a dataset that can be compared
with any other fingerprint from the same pipeline version, and that contains no
recoverable records.

Design rules that are not negotiable, and the reasons they exist, are documented
in ``docs/design.md``. The short version: nothing blocks a run whose purpose was
never declared, nothing recommends deleting data without a measured effect, and
everything that degraded says so.
"""

from __future__ import annotations

#: The single source of truth for the package version. `pyproject.toml` reads it
#: from here via `[tool.hatch.version]`, so bump it in this file only. Carrying
#: the number in both places let them drift: 0.1.4 was tagged in pyproject while
#: `dropoutt doctor` still reported 0.1.3 from this constant.
__version__ = "1.0.0"

from .models import (
    Confidence,
    Document,
    Finding,
    Profile,
    Severity,
)

__all__ = ["Confidence", "Document", "Finding", "Profile", "Severity", "__version__"]
