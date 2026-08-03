# Running on clusters, and offline

This is built to run on arbitrary HPC and cloud clusters, not one particular
one. That rules out assuming a modern glibc, a writable home directory, network
access at run time, or a specific Python patch version.

## Installing

Install from the checkout (or a built wheel):

```bash
python3 -m venv .venv
# Unix / macOS
.venv/bin/pip install -e '.[all]'
# Windows
.venv\Scripts\pip install -e ".[all]"
```

If the login node cannot build some of the optional wheels, drop `[all]` and add
extras one at a time. `dropoutt doctor` reports what is missing and what its
absence costs.

Cache directory (tokenizers, atlas embedder): `DROPOUTT_CACHE`, else
`XDG_CACHE_HOME/dropoutt`, else `%LOCALAPPDATA%\dropoutt` on Windows or
`~/.cache/dropoutt` elsewhere. Falls back to the system temp directory when
that location is not writable.

## Invoking it

The console script lands in the environment's `bin/`, which module systems and
batch schedulers frequently leave off `PATH`, and on a shared cluster you often
cannot change that. Running the package as a module works from any interpreter
that can import it:

```bash
python -m dropoutt --version
python -m dropoutt scan /scratch/$USER/data --offline
```

Every command accepts both forms. Use `python -m dropoutt` in batch scripts,
because it is the invocation that always works.

## The dependency rule

> The core scan requires NumPy but not `tokenizers`. Token-dependent checks are
> an optional extra, and every other compiled component is either optional or an
> accelerator with a fallback.

| dependency | role | what happens without it |
| --- | --- | --- |
| `typer`, `rich`, `jinja2`, `numpy` | CLI, output, report, maths | required; all pure Python except numpy |
| `tokenizers` | token counting, template rendering | token-dependent checks skip with an unlock hint |
| `orjson` | JSONL parsing | falls back to stdlib `json`; slower, identical results |
| `fasttext-langdetect` | language identification | falls back to a small character-profile detector; **less accurate**, and every finding it produces is marked low-trust |
| `model2vec` | atlas embeddings | atlas coverage is reported as skipped |
| `pyarrow` | Parquet, Arrow IPC, Feather, ORC | these columnar files are reported as unreadable with an install hint |
| `zstandard` | `.zst` input | the file is reported as unreadable with an install hint; gzip, bzip2, and xz use the standard library |

The language fallback is the only one where the difference is quality rather
than speed, which is why its results are labelled rather than silently
substituted.

Check what you have:

```bash
dropoutt doctor
```

## Known wheel constraints

`pyarrow` 25 and `numpy` 2.5 ship `manylinux_2_28` only. On an older CentOS-era
login node, `pip install pyarrow` falls back to a source build and fails. That is
why columnar formats are an optional extra and why the package declares `numpy>=1.24`
rather than pinning to the newest.

`fasttext-predict`, which `fasttext-langdetect` depends on, has cp312 wheels but
none for cp313 and later. If your cluster runs a newer Python, install without
the `lid` extra and the fallback detector takes over.

## Offline operation

Air-gapped compute nodes are normal. A scan reaches the network in exactly three
places:

1. the tokenizer for `--model`, or the five panel tokenizers when `--model` is
   not given
2. that model's `tokenizer_config.json`, which carries the chat template
3. the atlas embedding model

`dropoutt fetch` pulls all three ahead of time. Run it where there is egress.
The atlas artifact itself and the shipped contamination indices are inside the
package and are never downloaded.

### The two-node workflow

On the login node, with both cache variables pointing at shared storage:

```bash
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
export HF_HOME=/scratch/$USER/hf

dropoutt fetch --model qwen3     # the tokenizer you will scan against
dropoutt fetch                   # or: the whole panel, for scans with no --model
```

On the compute node, with the same two variables:

```bash
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
export HF_HOME=/scratch/$USER/hf

python -m dropoutt scan /scratch/$USER/data \
    --model qwen3 --seq-len 4096 --offline
```

Both variables matter, and this is the easy thing to get wrong. They hold
different files:

| file | variable that controls it | path |
| --- | --- | --- |
| tokenizer | `HF_HOME` | `$HF_HOME/hub` |
| `tokenizer_config.json`, so the chat template | `DROPOUTT_CACHE` | `$DROPOUTT_CACHE/hub` |
| atlas embedding model | `DROPOUTT_CACHE` | `$DROPOUTT_CACHE/embedder` |

`DROPOUTT_CACHE` on its own is not enough. The tokenizer is loaded through the
Hugging Face hub cache, which follows `HF_HOME`, so an offline run with only
`DROPOUTT_CACHE` set reports `could not load a tokenizer` and skips every
token-dependent check.

### What `--offline` resolves

`--offline` loads the tokenizer from the cache with `HF_HUB_OFFLINE=1` set for
the duration of the call, and the `tokenizer_config.json` with
`local_files_only=True`. It resolves from the cache rather than refusing, which
is the point: against a populated cache, an offline scan produces findings
identical to an online one, down to the token counts.

If no cached chat template is found the run says so before the report:

```
  note no cached chat template for Qwen/Qwen3-8B; token counts exclude template overhead
and loss-mask checks were skipped. Run `dropoutt fetch --model qwen3` on a node with
network.
```

That note matters. Without a template, records are counted as raw text, every
token number shifts, and the loss-mask checks do not run at all, so the numbers
must not be read as if the template had been applied.

The same offline flag gates atlas model loading. If its three files are absent
from `$DROPOUTT_CACHE/embedder`, coverage is reported as unavailable without a
connection attempt. The cache directory is always used for
`$DROPOUTT_CACHE/contamination`; it never writes into read-only site-packages.

`DROPOUTT_OFFLINE=1` and `HF_HUB_OFFLINE=1` are also honored by `scan` and
`init`. This prevents a missed command-line flag in a batch script from changing
the network contract.

### Local model directories

You can also point `--model` at a local directory holding `tokenizer.json` and
`tokenizer_config.json`. That path reads straight from disk and needs no network
and no cache at all.

```bash
python -m dropoutt scan ./data --model /scratch/$USER/models/qwen3-4b --offline
```

## Cache location

`DROPOUTT_CACHE` is resolved first, and used exactly as given. Only when it is
unset does the rest apply:

1. `DROPOUTT_CACHE`, used as-is
2. `$XDG_CACHE_HOME/dropoutt`
3. `~/.cache/dropoutt`
4. `<tempdir>/dropoutt-cache`, when the directory from 2 or 3 cannot be created
   or written to

The fourth case matters: compute nodes frequently mount `$HOME` read-only, and a
scanner that crashes on a cache write is a scanner that cannot run there. Note
that it is a fallback for the unset case only. If you set `DROPOUTT_CACHE` to a
path you cannot write, there is no fallback, which is the right behaviour for an
explicit setting.

```bash
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
```

## Inside a batch job

The HTML report is a single self-contained file with no server, no CDN and no
web fonts. That is deliberate: it can be produced inside a Slurm job, copied off
with `scp` when policy permits, and read by someone who has never installed the
tool.

```bash
#!/bin/bash
#SBATCH --job-name=dropoutt-scan
#SBATCH --time=00:30:00

export DROPOUTT_CACHE=/scratch/$USER/dropoutt
export HF_HOME=/scratch/$USER/hf

/scratch/$USER/dropoutt-cli/.venv/bin/python -m dropoutt \
    scan /scratch/$USER/data \
    --model /scratch/$USER/models/qwen3-4b \
    --seq-len 4096 \
    --offline \
    --no-evidence \
    --out /scratch/$USER/scan-out
```

Then apply the metadata policy described below before copying
`scan-out/report.html`.

### Output confidentiality

No telemetry or hosted service is used. Network access is limited to fetching
explicit model artifacts and the optional tokenizer panel described above.
That does not make every output artifact safe to export.

By default, terminal examples, `findings.jsonl`, `report.md`, and `report.html` contain
bounded excerpts and source locations. Detected PII is masked, but arbitrary
proprietary text and secrets outside the pattern catalog are not. Use:

```bash
python -m dropoutt scan /scratch/$USER/data \
    --offline \
    --no-evidence \
    --out /scratch/$USER/scan-out
```

The resulting findings and HTML report omit record excerpts and source
locations. They still contain dataset names, the scan root, aggregate
measurements, and hashes. Treat those as metadata under the VPC's data
classification policy. `fingerprint.json` never contains record excerpts but
contains the same metadata.

## Using it as a gate

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft --quiet
```

Exit code 10 means blocking findings under the declared target. Exit code 0
means the scan completed, findings or not. Exit code 1 means dropoutt itself
failed, which is a distinction CI needs and which most linters get wrong.

Without `--target` nothing ever blocks, because blocking asserts that something
is wrong for a purpose and no purpose was declared.
