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

__version__ = "0.1.0"

from .models import (  # noqa: F401
    Confidence,
    Document,
    Finding,
    Profile,
    Severity,
)

__all__ = ["__version__", "Document", "Finding", "Profile", "Severity", "Confidence"]
