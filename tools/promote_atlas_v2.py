#!/usr/bin/env python3
"""Validate Atlas v2 release artifacts, bundle them, and smoke-test the CLI."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from atlas_sources import DEFAULT_RELEASE, DEFAULT_WORK  # noqa: E402

from dropoutt.atlas.apply import Atlas  # noqa: E402
from dropoutt.atlas.profiles import ATLAS_V2, ATLAS_V2_LITE, AtlasProfile  # noqa: E402

ARTIFACTS = {
    "atlas-v2.npz": ATLAS_V2,
    "atlas-v2-lite.npz": ATLAS_V2_LITE,
}
SUPPORT_FILES = ("atlas-v2-release-notes.json",)
L1_LABEL_FILES = {
    ATLAS_V2.version: ROOT / "tools" / "atlas-data" / "l1_labels_atlas-v2.json",
    ATLAS_V2_LITE.version: ROOT / "tools" / "atlas-data" / "l1_labels_atlas-v2-lite.json",
}
MIN_L2_REFERENCE_SUPPORT = 1_000


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def load_progress(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def wait_for_build(progress_path: Path, interval: float) -> None:
    while True:
        payload = load_progress(progress_path)
        if payload is not None:
            status = str(payload.get("status", "running"))
            overall = float(payload.get("overall_pct", 0.0))
            stage = str(payload.get("stage", "unknown"))
            print(f"waiting: {overall:.2f}% {stage}", flush=True)
            if status == "complete" and overall >= 100.0:
                return
            pid = int(payload.get("pid", 0) or 0)
            if pid and not process_alive(pid):
                raise RuntimeError(f"build process {pid} stopped before completion")
        else:
            print(f"waiting for {progress_path}", flush=True)
        time.sleep(interval)


def normalized_labels(meta: dict, n_l1: int) -> tuple[list[str], bool]:
    raw = meta.get("l1_labels")
    labels = list(raw) if isinstance(raw, list) else []
    changed = len(labels) != n_l1
    if changed:
        labels = [f"area {index}" for index in range(n_l1)]
    used: set[str] = set()
    output: list[str] = []
    for index, value in enumerate(labels):
        label = str(value).strip() or f"area {index}"
        candidate = label
        if candidate in used:
            candidate = f"{label} [L1 {index}]"
            changed = True
        used.add(candidate)
        output.append(candidate)
    return output, changed


def curated_labels(profile: AtlasProfile) -> tuple[list[str], str] | None:
    path = L1_LABEL_FILES.get(profile.version)
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("labels")
    if payload.get("atlas_version") != profile.version or not isinstance(raw, dict):
        raise ValueError(f"{path.name} does not describe {profile.version}")
    labels = [str(raw.get(str(index), "")).strip() for index in range(profile.n_l1)]
    if not all(labels) or len(raw) != profile.n_l1 or len(set(labels)) != profile.n_l1:
        raise ValueError(f"{path.name} must contain {profile.n_l1} unique labels")
    return labels, f"curated:{path.name}"


def normalize_artifact_metadata(path: Path, profile: AtlasProfile) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    meta = json.loads(str(payload["meta"][0]))
    curated = curated_labels(profile)
    if curated is None:
        labels, changed = normalized_labels(meta, profile.n_l1)
        source = str(meta.get("l1_labels_source", "automatic-contrastive-terms"))
        if changed and not source.endswith("+deduplicated"):
            source += "+deduplicated"
    else:
        labels, source = curated
        changed = meta.get("l1_labels") != labels or meta.get("l1_labels_source") != source
    if not changed:
        return
    meta["l1_labels"] = labels
    meta["l1_labels_source"] = source
    payload["meta"] = np.asarray([json.dumps(meta)])
    temporary = path.with_suffix(".normalized.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _write_npz_atomically(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".repair.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _remap_crosswalk(meta: dict, old_to_new: np.ndarray, keep: np.ndarray) -> None:
    """Keep the shared full/lite crosswalk valid after removing full L2 cells."""
    crosswalk = meta.get("crosswalk")
    if not isinstance(crosswalk, dict):
        return
    full_to_lite = crosswalk.get("full_to_lite")
    full_modal_share = crosswalk.get("full_modal_share")
    if isinstance(full_to_lite, list) and len(full_to_lite) == len(keep):
        crosswalk["full_to_lite"] = [
            value for value, retained in zip(full_to_lite, keep.tolist(), strict=True)
            if retained
        ]
    if isinstance(full_modal_share, list) and len(full_modal_share) == len(keep):
        crosswalk["full_modal_share"] = [
            value for value, retained in zip(full_modal_share, keep.tolist(), strict=True)
            if retained
        ]
    lite_to_full = crosswalk.get("lite_to_full")
    if isinstance(lite_to_full, list):
        remapped: list[int] = []
        for value in lite_to_full:
            cell = int(value)
            if cell < 0 or cell >= len(old_to_new):
                raise ValueError(f"crosswalk refers to invalid full L2 cell {cell}")
            remapped.append(int(old_to_new[cell]))
        crosswalk["lite_to_full"] = remapped


def merge_undersized_full_l2_cells(path: Path) -> dict | None:
    """Merge invalid tiny full-v2 L2 cells into supported siblings.

    The production build now prevents this state. This repair makes already
    released artifacts safe without re-embedding the unchanged reference corpus.
    """
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    centroids = np.asarray(payload["centroids"], dtype=np.float32)
    parents = np.asarray(payload["l1_parent"], dtype=np.int32)
    support = np.asarray(payload["region_size"], dtype=np.int64)
    old_n = len(centroids)
    undersized = np.flatnonzero(support < MIN_L2_REFERENCE_SUPPORT)
    if not len(undersized):
        return None

    keep = np.ones(old_n, dtype=bool)
    keep[undersized] = False
    target = np.arange(old_n, dtype=np.int32)
    for cell in undersized.tolist():
        candidates = np.flatnonzero((parents == parents[cell]) & keep)
        if not len(candidates):
            raise ValueError(f"cannot merge L2 cell {cell}: no supported L1 sibling")
        target[cell] = int(candidates[np.argmax(centroids[candidates] @ centroids[cell])])

    compact = np.full(old_n, -1, dtype=np.int32)
    compact[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    old_to_new = compact[target]
    new_n = int(keep.sum())
    weighted = np.zeros((new_n, centroids.shape[1]), dtype=np.float64)
    np.add.at(weighted, old_to_new, centroids * support[:, None])
    merged_support = np.bincount(old_to_new, weights=support, minlength=new_n).astype(np.int64)
    merged_centroids = weighted / np.maximum(merged_support[:, None], 1)
    merged_centroids /= np.linalg.norm(merged_centroids, axis=1, keepdims=True) + 1e-9

    for name, value in list(payload.items()):
        if name == "cell_similarity" or not value.ndim or value.shape[0] != old_n:
            continue
        payload[name] = value[keep]
    payload["centroids"] = merged_centroids.astype(np.float32)
    payload["region_size"] = merged_support.astype(payload["region_size"].dtype)
    payload["region_category"] = parents[keep].astype(payload["region_category"].dtype)
    payload["l1_parent"] = parents[keep].astype(payload["l1_parent"].dtype)
    if "distance_refs_support" in payload:
        payload["distance_refs_support"] = merged_support.astype(payload["distance_refs_support"].dtype)
    if "distance_refs_reliable" in payload:
        payload["distance_refs_reliable"] = merged_support >= MIN_L2_REFERENCE_SUPPORT
    if "l1_child_count" in payload:
        payload["l1_child_count"] = np.bincount(
            parents[keep], minlength=len(payload["l1_child_count"])
        ).astype(payload["l1_child_count"].dtype)
    if "cooccurrence_ids" in payload:
        ids = np.asarray(payload["cooccurrence_ids"], dtype=np.int32).copy()
        valid = (ids >= 0) & (ids < old_n)
        ids[valid] = old_to_new[ids[valid]]
        payload["cooccurrence_ids"] = ids.astype(payload["cooccurrence_ids"].dtype)
    if "cell_similarity" in payload:
        payload["cell_similarity"] = (merged_centroids @ merged_centroids.T).astype(np.float16)

    meta = json.loads(str(payload["meta"][0]))
    terms = meta.get("region_terms")
    if isinstance(terms, list) and len(terms) == old_n:
        meta["region_terms"] = [
            value for value, retained in zip(terms, keep.tolist(), strict=True) if retained
        ]
    meta["n_regions"] = new_n
    meta.setdefault("repairs", []).append({
        "kind": "merge_undersized_l2_cells",
        "minimum_support": MIN_L2_REFERENCE_SUPPORT,
        "merged_cells": undersized.tolist(),
    })
    _remap_crosswalk(meta, old_to_new, keep)
    payload["meta"] = np.asarray([json.dumps(meta)])
    _write_npz_atomically(path, payload)
    return {
        "old_to_new": old_to_new,
        "keep": keep,
        "new_n": new_n,
        "merged_cells": undersized.tolist(),
    }


def repair_release_artifacts(release: Path) -> None:
    """Repair known-invalid full-v2 singleton cells and its lite crosswalk."""
    result = merge_undersized_full_l2_cells(release / "atlas-v2.npz")
    if result is None:
        normalize_artifact_metadata(release / "atlas-v2.npz", ATLAS_V2)
        normalize_artifact_metadata(release / "atlas-v2-lite.npz", ATLAS_V2_LITE)
        return
    lite = release / "atlas-v2-lite.npz"
    with np.load(lite, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    lite_meta = json.loads(str(payload["meta"][0]))
    _remap_crosswalk(lite_meta, result["old_to_new"], result["keep"])
    payload["meta"] = np.asarray([json.dumps(lite_meta)])
    _write_npz_atomically(lite, payload)

    notes_path = release / "atlas-v2-release-notes.json"
    if notes_path.is_file():
        notes = json.loads(notes_path.read_text(encoding="utf-8"))
        notes.setdefault("artifacts", {}).setdefault("atlas-v2", {})["n_regions"] = result["new_n"]
        notes.setdefault("repairs", []).append({
            "kind": "merge_undersized_l2_cells",
            "merged_cells": result["merged_cells"],
            "remaining_l2_cells": result["new_n"],
        })
        temporary = notes_path.with_suffix(".repair.tmp.json")
        temporary.write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
        temporary.replace(notes_path)
    normalize_artifact_metadata(release / "atlas-v2.npz", ATLAS_V2)
    normalize_artifact_metadata(release / "atlas-v2-lite.npz", ATLAS_V2_LITE)


def validate_artifact(path: Path, profile: AtlasProfile) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "centroids", "region_category", "region_size", "l1_centroids",
            "norm_mean", "norm_pca", "idf_token_ids", "idf_log_probs", "meta",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path.name} missing arrays: {', '.join(sorted(missing))}")
        meta = json.loads(str(data["meta"][0]))
        centroids = np.asarray(data["centroids"])
        n_cells = len(centroids)
        parents = np.asarray(data["region_category"])
        support = np.asarray(data["region_size"])
        l1 = np.asarray(data["l1_centroids"])
        minimum_cells = profile.n_l1 * profile.l2_k_min
        maximum_cells = profile.n_l1 * profile.l2_k_max
        if centroids.ndim != 2 or centroids.shape[1] != profile.dim:
            raise ValueError(f"{path.name} has centroid shape {centroids.shape}")
        if not minimum_cells <= n_cells <= maximum_cells:
            raise ValueError(f"{path.name} has {n_cells} cells outside best-k bounds")
        if l1.shape != (profile.n_l1, profile.dim):
            raise ValueError(f"{path.name} has L1 shape {l1.shape}")
        if parents.shape != (n_cells,) or support.shape != (n_cells,):
            raise ValueError(f"{path.name} has invalid L2 metadata shapes")
        if int(parents.min()) < 0 or int(parents.max()) >= profile.n_l1:
            raise ValueError(f"{path.name} has invalid L1 parent ids")
        if not np.isfinite(centroids).all() or not np.isfinite(l1).all():
            raise ValueError(f"{path.name} contains non-finite centroids")
        stored = meta.get("profile")
        if stored != asdict(profile):
            raise ValueError(f"{path.name} profile does not match the runtime declaration")
        if meta.get("version") != profile.version:
            raise ValueError(f"{path.name} version metadata is incorrect")
        if int(meta.get("n_reference_records", -1)) != int(support.sum()):
            raise ValueError(f"{path.name} reference population does not match support")
        labels = meta.get("l1_labels")
        if not isinstance(labels, list) or len(labels) != profile.n_l1:
            raise ValueError(f"{path.name} has invalid L1 labels")
        if len(set(map(str, labels))) != profile.n_l1:
            raise ValueError(f"{path.name} has duplicate L1 labels")
        crosswalk = meta.get("crosswalk")
        if not isinstance(crosswalk, dict):
            raise ValueError(f"{path.name} has no full/lite crosswalk")
    atlas = Atlas.load(path)
    if atlas.dim != profile.dim or atlas.n_l1 != profile.n_l1:
        raise ValueError(f"{path.name} does not load with its declared geometry")
    return meta


def validate_pair(directory: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name, profile in ARTIFACTS.items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        normalize_artifact_metadata(path, profile)
        result[name] = validate_artifact(path, profile)
    hashes = {meta.get("corpus_hash") for meta in result.values()}
    populations = {int(meta.get("n_reference_records", -1)) for meta in result.values()}
    encoders = {meta.get("encoder_weight_hash") for meta in result.values()}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("full and lite do not share one corpus hash")
    if len(populations) != 1 or min(populations) <= 0:
        raise ValueError("full and lite do not share one reference population")
    if len(encoders) != 1 or None in encoders:
        raise ValueError("full and lite do not share one encoder hash")
    return result


def cli_smoke() -> None:
    executable = Path(sys.executable).parent / "dropoutt"
    if not executable.is_file():
        raise FileNotFoundError(f"dropoutt executable not found beside {sys.executable}")
    with tempfile.TemporaryDirectory(prefix="atlas-v2-cli-") as temporary:
        root = Path(temporary)
        corpus = root / "records.jsonl"
        rows = [
            {"text": " ".join([
                "Public scientific writing about ocean circulation, climate measurements, and reproducible observational methods."
            ] * 4)},
            {"text": " ".join([
                "An openly licensed software guide explaining database transactions, network services, and error recovery procedures."
            ] * 4)},
            {"text": " ".join([
                "A public history article discussing architecture, literature, civic institutions, and archival primary sources."
            ] * 4)},
        ]
        corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        for version in ("atlas-v2", "atlas-v2-lite"):
            command = [
                str(executable), "atlas", "--model", version, str(corpus),
                "--out", str(root / version), "--offline", "--sampling", "0",
                "--no-html", "--no-open", "--quiet",
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=300
            )
            if completed.returncode:
                output = (completed.stdout + "\n" + completed.stderr).strip()
                raise RuntimeError(f"CLI smoke failed for {version}:\n{output}")


def promote(release: Path, package_dir: Path) -> None:
    for name in (*ARTIFACTS, *SUPPORT_FILES):
        if not (release / name).is_file():
            raise FileNotFoundError(release / name)
    package_dir.mkdir(parents=True, exist_ok=True)
    stage = package_dir / f".atlas-v2-stage-{os.getpid()}"
    backup = package_dir / f".atlas-v2-backup-{os.getpid()}"
    stage.mkdir()
    backup.mkdir()
    names = [*ARTIFACTS, *SUPPORT_FILES, "atlas-v2-SHA256SUMS"]
    originally_present = {name for name in names if (package_dir / name).is_file()}
    try:
        for name in (*ARTIFACTS, *SUPPORT_FILES):
            shutil.copy2(release / name, stage / name)
        metadata = validate_pair(stage)
        checksum_lines = [f"{digest(stage / name)}  {name}" for name in ARTIFACTS]
        (stage / "atlas-v2-SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        for name in originally_present:
            shutil.copy2(package_dir / name, backup / name)
        for name in names:
            (stage / name).replace(package_dir / name)
        try:
            validate_pair(package_dir)
            cli_smoke()
        except Exception:
            for name in names:
                target = package_dir / name
                if name in originally_present:
                    (backup / name).replace(target)
                else:
                    target.unlink(missing_ok=True)
            raise
        population = next(iter(metadata.values()))["n_reference_records"]
        corpus_hash = next(iter(metadata.values()))["corpus_hash"]
        print(f"promoted both artifacts: records={population:,} corpus={corpus_hash}")
        print("CLI smoke passed for atlas-v2 and atlas-v2-lite")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--package-dir", type=Path, default=ROOT / "src" / "dropoutt" / "data" / "atlas"
    )
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--progress", type=Path, default=DEFAULT_WORK / "build-progress.json")
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.wait:
        wait_for_build(args.progress, args.interval)
    args.package_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.release / ".atlas-v2-promote.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        repair_release_artifacts(args.release)
        promote(args.release, args.package_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
