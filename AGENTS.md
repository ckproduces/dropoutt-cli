# AGENTS.md

## Cursor Cloud specific instructions

`dropoutt` is a CPU-only Python CLI (no server, no web app). The "application" is
the `dropoutt` command; running it means invoking the CLI against a data folder.

### Environment
- Python 3.12 (system) with a project venv at `.venv`. Activate with
  `source .venv/bin/activate` before running anything.
- Dependencies are installed as an editable install with all extras: the update
  script runs `pip install -e ".[dev]"`, which pulls in `[all]` (tokenizer,
  parquet, zstd, lid, atlas, fast) plus `pytest`, `ruff`, `mypy`, `scikit-learn`.
- CI (`.github/workflows/ci.yml`) uses plain `pip`, not `uv`, even though a
  `uv.lock` is present. Match CI: use the venv + pip.

### Lint / type / test / build (from README "Develop" and CI)
- Lint: `ruff check .`
- Type check: `mypy` (config in `pyproject.toml`, checks `src/dropoutt`).
- Tests: `python -m pytest -q`. Run with `DROPOUTT_OFFLINE=1 HF_HUB_OFFLINE=1`
  (as CI does) to keep tests from touching the network.
- Build: `python -m build && twine check dist/*` (`build`/`twine` are not in
  `[dev]`; install them ad hoc if you need to test packaging).

### Running the app (hello world)
- `DROPOUTT_OPEN=0 dropoutt scan ./test-data --no-atlas` scans the bundled
  sample corpus and writes `test-data/.dropoutt/{report.html,report.md,findings.jsonl,fingerprint.json}`.
- Add `--model qwen3 --seq-len 4096 --target sft` to exercise the tokenizer/mask
  checks and exit-code gating (downloads a tokenizer from HF on first use).
- `dropoutt doctor` reports which optional extras are installed.

### Non-obvious gotchas
- Always set `DROPOUTT_OPEN=0` in this headless/cloud environment; otherwise a
  scan tries to open the HTML report in a desktop browser. (`--no-open` also works.)
- The atlas coverage map downloads a ~500 MB static embedder on first use. Use
  `--no-atlas` for a fast, fully offline scan; run `dropoutt fetch` first if you
  want atlas checks. Tests skip atlas-dependent paths when it is unavailable.
- Scans print a harmless "sending unauthenticated requests to the HF Hub"
  warning when a tokenizer/atlas asset is resolved; it is not an error.
- Generated `.dropoutt/` output and `.venv/` are gitignored.
