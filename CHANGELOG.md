# Changelog

## 0.1.0

First release. Tier 0 and Tier 1 checks, a fingerprint, and a first atlas.

### Added

- **28 checks** across two tiers. Tier 0 is structural and needs no model; Tier 1
  is statistical. `dropoutt checks` lists them; `dropoutt checks <id>` explains
  one.
- **Zero-configuration scan.** Discovery, schema induction against ten known
  layouts, detection of files that are not training data at all, hypothesis-based
  profile inference, and a cross-tokenizer token budget when no `--model` is given.
- **Contamination scanning** using the Tülu 3 rule, against 10 bundled hashed
  8-gram indices for permissively licensed benchmarks. `dropoutt index-eval`
  builds an index from your own held-out set locally.
- **`atlas-lite-v0`**: 255 regions over 41,054 reference records, level-0
  taxonomy accuracy 0.879, region purity 0.830. Coverage is suppressed when the
  off-atlas rate exceeds 10%, and is reported per language.
- **Self-contained HTML report**, one file, no server and no CDN.
- **Three-valued exit codes**: 0 completed, 1 internal error, 2 usage, 10
  blocking findings under a declared target.
- Verified static registries: 22 benchmarks with their genuinely scorable eval
  splits, 12 chat-template families with literal delimiters, 37 models including
  the Turkish set, PII patterns with checksums.

### Policies enforced in code and covered by tests

- Nothing blocks a run whose purpose was never declared.
- No check recommends deleting data without a measured effect; every finding in
  this release is `unverified`.
- Cross-dataset overlap is directional.
- PII never reaches the report; a test fails if a planted secret appears.
- Anything that degraded says so.

### Known issues

See [docs/limitations.md](docs/limitations.md). The largest are that language
identification is the main source of false positives, scale above roughly a
million records is unprofiled, and the scan is single-process.
