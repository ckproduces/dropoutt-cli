#!/usr/bin/env python3
"""Build hashed 8-gram contamination indices from public benchmarks.

Only benchmarks marked ``shippable`` in ``data/benchmarks.json`` are built, which
means only those whose Hub licence permits redistribution. HellaSwag, WinoGrande,
PIQA, BBH and OpenBookQA carry no licence tag at all, so we do not ship indices
for them; users can build those locally with ``dropoutt index-eval``.

What lands on disk is hashed n-grams and instance sizes. The benchmark text is
not stored and cannot be recovered from the index.

    python tools/build_contamination_index.py --out src/dropoutt/data/contamination
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dropoutt.contamination import BenchmarkIndex  # noqa: E402
from dropoutt.registry_data import shippable_benchmarks  # noqa: E402


def load_rows(hf_id: str, config: str | None, split: str, limit: int | None = None):
    """Fetch a benchmark split without pulling in the `datasets` library."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield row


def row_text(row: dict, question_fields: list[str], answer_field: str | None) -> str:
    parts: list[str] = []
    for field in question_fields:
        value = row.get(field)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif isinstance(value, dict):
            parts.extend(str(v) for v in value.values())
    if answer_field:
        value = row.get(answer_field)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(p for p in parts if p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("src/dropoutt/data/contamination"))
    ap.add_argument("--only", nargs="*", help="Restrict to these benchmark ids.")
    ap.add_argument("--limit", type=int, default=None, help="Cap instances per benchmark.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    targets = shippable_benchmarks()
    if args.only:
        targets = [b for b in targets if b["id"] in set(args.only)]

    built, failed = [], []
    for bench in targets:
        name = bench["id"]
        try:
            idx = BenchmarkIndex(
                name=name, n_instances=0,
                license=bench.get("license"),
                source=f'{bench["hf_id"]}:{bench.get("config") or "-"}:{bench["eval_split"]}',
            )
            n = 0
            for row in load_rows(bench["hf_id"], bench.get("config"),
                                 bench["eval_split"], args.limit):
                text = row_text(row, bench["question_fields"], bench.get("answer_field"))
                if not text.strip():
                    continue
                idx.add_instance(n, text)
                n += 1
            idx.n_instances = n
            dest = args.out / f"{name}.idx"
            idx.save(dest)
            size_kb = dest.stat().st_size / 1024
            print(f"  {name:<16} {n:>6,} instances  {len(idx.postings):>8,} grams  "
                  f"{size_kb:>7.0f} KB  [{bench.get('license')}]")
            built.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<16} FAILED: {type(exc).__name__}: {exc}")
            failed.append(name)

    print(f"\nbuilt {len(built)}, failed {len(failed)}")
    if failed:
        print("failed:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
