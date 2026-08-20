"""Folder discovery: what is here, and where do datasets begin and end.

Dataset boundaries are read from the directory structure, never inferred from
content. A record knows which dataset it belongs to because of where it was
found. When the input is a single undifferentiated file there are no boundaries,
and the cross-dataset overlap matrix is simply not produced rather than invented.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import DatasetRef

#: Directories that never contain training data.
#:
#: The list grew after a user pointed a scan at their home directory. The walk
#: itself is fast — it is what happens to each file afterwards that is not — so
#: the cheapest possible fix is to never see the file. Everything here is a
#: place whose contents are generated, vendored or installed, and a `.json` or
#: `.md` inside one of them is a manifest, a lockfile or a changelog rather than
#: a training record.
SKIP_DIRS = {
    # Version control and editor state.
    ".git", ".hg", ".svn", ".idea", ".vscode", ".ipynb_checkpoints",
    # Language package trees and their caches.
    "node_modules", "__pycache__", ".venv", "venv", "site-packages",
    "bower_components", "vendor", ".bundle", ".gem", ".cargo", ".rustup",
    ".nuget", ".m2", ".gradle", ".ivy2", ".pub-cache", "Pods",
    # Tool caches.
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".cache", ".tox", ".nox",
    ".npm", ".yarn", ".pnpm-store", ".nvm", ".terraform", ".dropoutt",
    # Build output. Named exactly, and only at the level the walk finds them,
    # because a dataset called `build` is possible and a compiled `build/`
    # directory holding training data is not.
    ".next", ".nuxt", ".svelte-kit", ".parcel-cache", ".gradle-cache",
    # Environment managers that install thousands of packages under one root.
    "miniconda3", "anaconda3", "miniforge3", "pkgs",
    # Operating-system trees under a home directory. Every one of these is
    # gigabytes of application state, and none of it is a corpus.
    "Library", "Applications", "AppData", ".Trash", "$RECYCLE.BIN",
    "System Volume Information", "OneDriveTemp",
}

#: Extensions this tool can read, and what each one is.
#:
#: `.mds` (MosaicML Streaming) and `.tar` (WebDataset) are container formats —
#: one file holding many records in a layout of their own. See
#: :mod:`dropoutt.containers`.
DATA_SUFFIXES = {
    ".jsonl", ".ndjson", ".json", ".parquet", ".arrow", ".feather", ".orc",
    ".txt", ".md", ".csv", ".tsv", ".mds", ".tar",
}
COMPRESSED = {".gz", ".zst", ".bz2", ".xz"}

#: Filenames that indicate the directory is a Hugging Face dataset.
CARD_NAMES = {"README.md", "dataset_infos.json", "dataset_dict.json"}

#: Files whose extension is in :data:`DATA_SUFFIXES` and which are never
#: training data. Matched on the whole lowercased filename, so a dataset that
#: genuinely is called `train.json` is untouched.
#:
#: This is the other half of the home-directory fix. A developer's tree is full
#: of `package.json`, `tsconfig.json` and `CHANGELOG.md`, and each one costs a
#: sniff, a schema induction and its own single-file "dataset" in the report —
#: which is how a scan came back describing four thousand datasets of one record
#: each. Every name here is a fixed convention of some toolchain, not a guess
#: about content.
NOT_DATA_NAMES = {
    # JavaScript / TypeScript.
    "package.json", "package-lock.json", "tsconfig.json", "jsconfig.json",
    "composer.json", "composer.lock", "bun.lock", "deno.json", "deno.lock",
    "angular.json", "nx.json", "turbo.json", "biome.json", ".eslintrc.json",
    ".babelrc.json", ".prettierrc.json", "manifest.json", "lerna.json",
    "jest.config.json", "tslint.json",
    # Python / Rust / Go / Ruby packaging.
    "pipfile.lock", "poetry.lock", "uv.lock", "cargo.lock", "go.sum",
    "gemfile.lock", "pdm.lock", "renovate.json",
    # Model and tokenizer metadata that lives beside weights. These are read
    # deliberately elsewhere; as records they are one dict of hyperparameters.
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "preprocessor_config.json", "model_index.json",
    "adapter_config.json", "quantize_config.json", "chat_template.json",
    # Documentation conventions.
    "readme.md", "changelog.md", "contributing.md", "license.md", "code_of_conduct.md",
    "security.md", "authors.md", "notice.md", "history.md", "todo.md",
    "requirements.txt", "constraints.txt", "license.txt", "notice.txt",
    "authors.txt", "copying.txt", "install.txt", "manifest.in",
}


@dataclass(slots=True)
class FileRef:
    path: str
    rel: str
    size: int
    suffix: str
    compressed: bool = False

    @property
    def readable(self) -> bool:
        return self.suffix in DATA_SUFFIXES


@dataclass(slots=True)
class Discovery:
    root: str
    files: list[FileRef] = field(default_factory=list)
    datasets: list[DatasetRef] = field(default_factory=list)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)
    total_bytes: int = 0
    #: Dataset names produced by folding sharded siblings together.
    shard_families: list[str] = field(default_factory=list)
    #: Files whose extension matched but which were passed over, by reason.
    #: Counted rather than listed: on a home directory this is hundreds of
    #: thousands of files, and a list of them is not a report.
    passed_over: dict[str, int] = field(default_factory=dict)
    #: Directories not descended into, by name. Same reasoning.
    skipped_dirs: int = 0

    @property
    def format_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[f.suffix] = counts.get(f.suffix, 0) + 1
        return counts

    @property
    def single_file(self) -> bool:
        return len(self.files) == 1


#: A trailing shard number, with or without the Hugging Face ``-of-NNNNN`` tail.
#: ``responses_0001``, ``train-00003-of-00042`` and ``part.7`` all reduce to
#: their family name.
SHARD_SUFFIX = re.compile(r"^(?P<stem>.*?)[._-]\d+(?:[._-]of[._-]\d+)?$", re.IGNORECASE)

#: How many siblings must share a stem before they are treated as one sharded
#: dataset. Two files are a coincidence; three are a naming convention.
MIN_SHARD_FAMILY = 3


def _dataset_name(root: Path, file_path: Path) -> str:
    """Name the dataset a file belongs to.

    A file directly under the scan root belongs to a dataset named after the
    file. A file in a subdirectory belongs to a dataset named after the
    directory, so ``data/turkish-qa/train.jsonl`` and
    ``data/turkish-qa/valid.jsonl`` are one dataset with two files.

    Split-named files are folded into the parent, because train and validation
    of the same corpus are the same dataset for our purposes, and keeping them
    apart would make the overlap matrix report the obvious.
    """
    rel = file_path.relative_to(root)
    parts = rel.parts
    if len(parts) == 1:
        return rel.stem
    return "/".join(parts[:-1])


def _shard_family(name: str) -> str | None:
    """The family a sharded filename belongs to, or None if it is not sharded."""
    m = SHARD_SUFFIX.match(name)
    if not m:
        return None
    stem = m.group("stem").rstrip("._-")
    return stem or None


def _fold_shards(names: dict[str, list[Path]]) -> dict[str, str]:
    """Map each dataset name to the family it should be merged into.

    Sharded exports are the normal way large datasets arrive — ``train-00000-of
    -00042.parquet`` from the Hub, ``responses_0001.txt`` from a generation run —
    and naming each shard as its own dataset breaks more than cosmetics. Every
    per-dataset statistic is then computed over one shard, the cross-dataset
    overlap matrix compares a corpus against itself, and a folder of 251 shards
    reports 251 datasets of one record each.

    Only names that reduce to the same stem *and* appear at least
    ``MIN_SHARD_FAMILY`` times are folded, so ``train.jsonl`` and ``valid.jsonl``
    stay separate and a lone ``notes_2024.txt`` keeps its name.
    """
    families: dict[str, list[str]] = {}
    for name in names:
        family = _shard_family(name)
        if family:
            families.setdefault(family, []).append(name)

    mapping: dict[str, str] = {}
    for family, members in families.items():
        if len(members) < MIN_SHARD_FAMILY:
            continue
        # An existing dataset already using the family name absorbs the shards
        # rather than colliding with them.
        for member in members:
            mapping[member] = family
    return mapping


def discover(root: str, *, follow_symlinks: bool = False, max_files: int = 200_000) -> Discovery:
    """Walk a path and group what is found into datasets."""
    root_path = Path(root).resolve()
    disc = Discovery(root=str(root_path))

    if root_path.is_file():
        size = root_path.stat().st_size
        suffix, compressed = _classify(root_path)
        ref = FileRef(str(root_path), root_path.name, size, suffix, compressed)
        disc.files.append(ref)
        disc.total_bytes = size
        if size == 0:
            disc.empty_files.append(ref.rel)
        disc.datasets = [
            DatasetRef(name=root_path.stem, root=str(root_path.parent),
                       files=[str(root_path)], total_bytes=size)
        ]
        return disc

    # Collected first, grouped second: shard folding needs to see every sibling
    # before it can tell a naming convention from a coincidence.
    pending: list[tuple[str, Path, int]] = []
    count = 0
    limit_hit = False
    for dirpath, dirnames, filenames, dirfd in _walk(root_path, follow_symlinks):
        keep = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        disc.skipped_dirs += len(dirnames) - len(keep)
        dirnames[:] = sorted(keep)
        if limit_hit:
            # Nothing below this point can be read, so there is no reason to
            # keep walking into it. Recording one skip per remaining file used
            # to build a list with millions of entries in it, which cost more
            # memory than the scan it was describing.
            dirnames[:] = []
            continue
        # `index.json` beside a MosaicML shard is that shard's column schema,
        # not a record. It is only metadata in that company: an `index.json`
        # anywhere else is somebody's data and is read as such.
        local_cards = CARD_NAMES
        if any(n.lower().endswith(".mds") for n in filenames):
            local_cards = CARD_NAMES | {"index.json"}

        for name in sorted(filenames):
            if count >= max_files:
                limit_hit = True
                disc.skipped_files.append((name, "file limit reached"))
                break
            suffix, compressed = _classify_name(name)
            if suffix not in DATA_SUFFIXES:
                continue
            if name in local_cards:
                continue
            lowered = name.lower()
            if lowered in NOT_DATA_NAMES:
                disc.passed_over["known non-data filename"] = (
                    disc.passed_over.get("known non-data filename", 0) + 1
                )
                continue
            # One stat per candidate, taken relative to the directory handle
            # the walk already holds where the platform offers one. On a tree
            # with a hundred thousand candidates that is a hundred thousand
            # path resolutions saved.
            try:
                size = os.stat(name, dir_fd=dirfd).st_size if dirfd is not None \
                    else os.stat(os.path.join(dirpath, name)).st_size
            except OSError as exc:
                disc.skipped_files.append((os.path.join(dirpath, name), str(exc)))
                continue
            fp = Path(dirpath) / name
            rel = str(fp.relative_to(root_path))
            ref = FileRef(str(fp), rel, size, suffix, compressed)
            disc.files.append(ref)
            disc.total_bytes += size
            count += 1
            if size == 0:
                disc.empty_files.append(rel)

            pending.append((_dataset_name(root_path, fp), fp, size))

    folded = _fold_shards({name: [] for name, _, _ in pending})
    if folded:
        disc.shard_families = sorted(set(folded.values()))

    by_dataset: dict[str, DatasetRef] = {}
    for raw_name, fp, size in pending:
        ds_name = folded.get(raw_name, raw_name)
        ds = by_dataset.get(ds_name)
        if ds is None:
            ds = DatasetRef(name=ds_name, root=str(fp.parent))
            by_dataset[ds_name] = ds
        ds.files.append(str(fp))
        ds.total_bytes += size

    disc.datasets = sorted(by_dataset.values(), key=lambda d: d.name)
    _attach_cards(root_path, disc)
    return disc


def _walk(root: Path, follow_symlinks: bool):
    """``os.walk``, yielding the directory file descriptor where there is one.

    ``os.fwalk`` hands back an open descriptor per directory, which lets the
    size check below be a ``fstatat`` rather than a full path resolution. It
    does not exist on Windows, where the same loop runs through ``os.walk`` and
    ``dirfd`` is None.
    """
    fwalk = getattr(os, "fwalk", None)
    if fwalk is None:  # pragma: no cover - Windows
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            yield dirpath, dirnames, filenames, None
        return
    yield from fwalk(root, follow_symlinks=follow_symlinks)


def effective_suffix(path: str | Path) -> tuple[str, bool]:
    """``(".jsonl", True)`` for ``train.jsonl.zst``. The one place this is decided.

    Discovery, the shard planner and the reader dispatch all have to agree on
    what a file is, and each of them used to work it out from ``Path.suffix``
    separately. They disagreed on case: discovery matched ``.JSONL`` against a
    lowercase set and dropped the file, while the reader would have read it
    fine.
    """
    return _classify_name(os.path.basename(str(path)))


def _classify(path: Path) -> tuple[str, bool]:
    """Return the effective data suffix, seeing through compression."""
    return _classify_name(path.name)


def _classify_name(name: str) -> tuple[str, bool]:
    """The same, from a filename alone.

    Split by hand rather than through ``Path.suffixes``, which allocates a
    ``Path`` and a list per call. The walk asks this of every file it sees, so
    on a home directory it is called a million times and the allocation shows
    up in the profile ahead of the actual reading.
    """
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        return "", False
    suffix = "." + suffix.lower()
    if suffix in COMPRESSED:
        inner_stem, inner_dot, inner = stem.rpartition(".")
        if not inner_dot or not inner_stem:
            return "", True
        return "." + inner.lower(), True
    return suffix, False


def _attach_cards(root: Path, disc: Discovery) -> None:
    """Read licence and declared language from any dataset card found."""
    from .cards import parse_card

    for ds in disc.datasets:
        card = Path(ds.root) / "README.md"
        if not card.exists():
            continue
        try:
            meta = parse_card(card.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        ds.license = meta.get("license")
        ds.declared_language = meta.get("language", [])
