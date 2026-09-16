from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RESULTS = EXP / "results"
CACHE = Path("/Volumes/ck512/dropoutt-atlas-v2/corpus-cache")
WORK = Path("/Volumes/ck512/dropoutt-atlas-v2/work")
SCRATCH = Path(os.environ.get("TOKENS_SCRATCH", "/private/tmp/claude-501/-Users-crokan-Documents-dropoutt-cli/84eb0be5-0e3c-48c1-9cdd-8000b341f0ec/scratchpad/tokens"))
SAMPLES = SCRATCH / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

#: Records to draw in total, spread over sources in proportion to their share of
#: the map's corpus, with a floor so every source's ratio is estimated on its
#: own records rather than borrowed from its axis.
TARGET_RECORDS = 600_000
FLOOR_PER_SOURCE = 1_500
SEED = 20260912


def population() -> dict:
    return json.loads((RESULTS / "population.json").read_text())


def manifest() -> dict:
    m = json.loads((CACHE / "manifest.json").read_text())
    return {s["slug"]: s for s in m["sources"]}


def allocation() -> dict[str, int]:
    pop = population()
    n = pop["n"]
    return {slug: max(FLOOR_PER_SOURCE, round(TARGET_RECORDS * v["records"] / n))
            for slug, v in pop["sources"].items()}
