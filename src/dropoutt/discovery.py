"""Folder discovery: what is here, and where do datasets begin and end.

Dataset boundaries are read from the directory structure, never inferred from
content. A record knows which dataset it belongs to because of where it was
found. When the input is a single undifferentiated file there are no boundaries,
and the cross-dataset overlap matrix is simply not produced rather than invented.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import DatasetRef

#: Directories that never contain training data.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".dropoutt", ".ipynb_checkpoints", ".idea",
    ".vscode", "site-packages", ".cache",
}

DATA_SUFFIXES = {".jsonl", ".ndjson", ".json", ".parquet", ".txt", ".md", ".csv", ".tsv"}
COMPRESSED = {".gz", ".zst", ".bz2", ".xz"}

#: Filenames that indicate the directory is a Hugging Face dataset.
CARD_NAMES = {"README.md", "dataset_infos.json", "dataset_dict.json"}


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

    @property
    def format_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[f.suffix] = counts.get(f.suffix, 0) + 1
        return counts

    @property
    def single_file(self) -> bool:
        return len(self.files) == 1


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

    by_dataset: dict[str, DatasetRef] = {}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in sorted(filenames):
            if count >= max_files:
                disc.skipped_files.append((name, "file limit reached"))
                continue
            fp = Path(dirpath) / name
            suffix, compressed = _classify(fp)
            if suffix not in DATA_SUFFIXES:
                continue
            if name in CARD_NAMES:
                continue
            try:
                size = fp.stat().st_size
            except OSError as exc:
                disc.skipped_files.append((str(fp), str(exc)))
                continue

            rel = str(fp.relative_to(root_path))
            ref = FileRef(str(fp), rel, size, suffix, compressed)
            disc.files.append(ref)
            disc.total_bytes += size
            count += 1
            if size == 0:
                disc.empty_files.append(rel)

            ds_name = _dataset_name(root_path, fp)
            ds = by_dataset.get(ds_name)
            if ds is None:
                ds = DatasetRef(name=ds_name, root=str(fp.parent))
                by_dataset[ds_name] = ds
            ds.files.append(str(fp))
            ds.total_bytes += size

    disc.datasets = sorted(by_dataset.values(), key=lambda d: d.name)
    _attach_cards(root_path, disc)
    return disc


def _classify(path: Path) -> tuple[str, bool]:
    """Return the effective data suffix, seeing through compression."""
    suffixes = path.suffixes
    if suffixes and suffixes[-1] in COMPRESSED:
        inner = suffixes[-2] if len(suffixes) > 1 else ""
        return inner, True
    return path.suffix, False


def _attach_cards(root: Path, disc: Discovery) -> None:
    """Read licence and declared language from any dataset card found."""
    from .cards import parse_card  # noqa: PLC0415

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
