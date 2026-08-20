"""The scan report as one JSON document.

There were two JSON artifacts before this one and neither is a report.

``findings.jsonl`` is the findings and nothing else — no composition, no token
budget, no atlas, no provenance — because it is a stream for a script that wants
to iterate over problems.

``fingerprint.json`` is the opposite: it is the comparable measurement, and it
is deliberately kept free of anything quotable from the corpus so that it can be
shared. It answers "did this dataset change" and refuses to answer "what is in
it".

So anyone who wanted the *report* programmatically — a dashboard, a pipeline
stage that gates on coverage, a script that posts a summary somewhere — had to
parse the Markdown or scrape the HTML. This is that report, with exactly the
content of the HTML page and the Markdown file, in the shape a program wants.

``--no-evidence`` applies here as it does everywhere else; the flag is honoured
in :mod:`dropoutt.report.payload`, which is where all four formats get their
content.
"""

from __future__ import annotations

from typing import Any

from ..compat import json_dumps
from ..fingerprint import Fingerprint
from ..runner import ScanResult
from .payload import build as build_payload
from .summary import ScanSummary


def render(
    result: ScanResult,
    fp: Fingerprint,
    budget: Any = None,
    *,
    include_evidence: bool = True,
    summary: ScanSummary | None = None,
) -> str:
    return json_dumps(
        build_payload(result, fp, budget, include_evidence=include_evidence,
                      summary=summary),
        indent=True,
    ) + "\n"
