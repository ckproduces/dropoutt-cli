# Running on clusters, and offline

This is built to run on arbitrary HPC and cloud clusters, not one particular
one. That rules out assuming a modern glibc, a writable home directory, network
access at run time, or a specific Python patch version.

## The dependency rule

> The core scan runs with no compiled dependency other than `tokenizers`, which
> is the only one that still ships `manylinux_2_17` and therefore installs on
> old cluster images. Everything faster is an accelerator with a fallback that
> produces the same answers.

| dependency | role | what happens without it |
| --- | --- | --- |
| `typer`, `rich`, `jinja2`, `numpy` | CLI, output, report, maths | required; all pure Python except numpy |
| `tokenizers` | token counting, template rendering | token-dependent checks skip with an unlock hint |
| `orjson` | JSONL parsing | falls back to stdlib `json`; slower, identical results |
| `rensa` | Rust MinHash | falls back to numpy MinHash; same permutation scheme and banding, so clusters agree |
| `fasttext-langdetect` | language identification | falls back to a small character-profile detector; **less accurate**, and every finding it produces is marked low-trust |
| `model2vec` | atlas embeddings | atlas coverage is reported as skipped |
| `pyarrow` | Parquet | `.parquet` files are reported as unreadable with an install hint |

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
why Parquet is an optional extra and why the package declares `numpy>=1.24`
rather than pinning to the newest.

`fasttext-predict`, which `fasttext-langdetect` depends on, has cp312 wheels but
none for cp313 and later. If your cluster runs a newer Python, install without
the `lid` extra and the fallback detector takes over.

## Offline operation

Air-gapped compute nodes are normal. The scan makes network calls in exactly two
places: resolving `--model` against the Hub, and fetching an embedding model for
the atlas.

```bash
# On the login node, where there is egress:
dropoutt scan ./data --model Qwen/Qwen3-8B   # populates the cache

# On the compute node:
dropoutt scan ./data --model Qwen/Qwen3-8B --offline
```

`--offline` disables every network call. A model that is not already cached
causes the token-dependent checks to skip with a clear reason rather than
hanging on a connection attempt.

You can also point `--model` at a local directory containing `tokenizer.json`
and `tokenizer_config.json`, which needs no network at all.

## Cache location

Resolved in this order:

1. `DROPOUTT_CACHE`
2. `$XDG_CACHE_HOME/dropoutt`
3. `~/.cache/dropoutt`
4. a temp directory, when the home directory is read-only

The fourth case matters: compute nodes frequently mount `$HOME` read-only, and a
scanner that crashes on a cache write is a scanner that cannot run there.

```bash
export DROPOUTT_CACHE=/scratch/$USER/dropoutt
```

## Inside a batch job

The HTML report is a single self-contained file with no server, no CDN and no
web fonts. That is deliberate: it can be produced inside a Slurm job, copied off
with `scp`, and read by someone who has never installed the tool.

```bash
#!/bin/bash
#SBATCH --job-name=dropoutt-scan
#SBATCH --time=00:30:00

export DROPOUTT_CACHE=/scratch/$USER/dropoutt
dropoutt scan /scratch/$USER/data \
    --model /scratch/$USER/models/qwen3-4b \
    --seq-len 4096 \
    --offline \
    --out /scratch/$USER/scan-out
```

Then `scp` `scan-out/report.html` to your laptop.

## Using it as a gate

```bash
dropoutt scan ./data --model qwen3 --seq-len 4096 --target sft --quiet
```

Exit code 10 means blocking findings under the declared target. Exit code 0
means the scan completed, findings or not. Exit code 1 means dropoutt itself
failed, which is a distinction CI needs and which most linters get wrong.

Without `--target` nothing ever blocks, because blocking asserts that something
is wrong for a purpose and no purpose was declared.
