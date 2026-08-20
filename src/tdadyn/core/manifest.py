"""The registry of generated datasets, ``data/synthetic/manifest.json``.

The manifest is the only generated file kept in git: it records what each dataset
directory was built from, so that a rebuild with different parameters is visible in
the diff instead of being silently absorbed by ``data/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .contracts import FORMAT_VERSION, GroundTruth
from .io import GROUND_TRUTH_FILE, read_json, write_json
from .schema import jsonify

MANIFEST_FILE = "manifest.json"

_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class ManifestEntry:
    """One line of the registry.

    Parameters
    ----------
    dataset_id : str
        Directory name, which also encodes the generator parameters.
    family, generator : str
        Copied from the ground truth for readability of the registry itself.
    seed : int
        Seed of the dataset's stream.
    path : str
        Dataset directory relative to ``output_root``, POSIX separators.
    params : dict
        Generator arguments.
    arrays : dict of str to str
        File name to sha256 of the ``.npy`` file's bytes.
    truth_sha256 : str
        sha256 of the **canonical serialisation** of the ground truth, not of the file's
        bytes.  Hashing the bytes would make indentation, key order and float ``repr``
        part of the digest, so a cosmetic reformat would read as changed data.
    package_version : str
        Version of ``tdadyn`` that produced the dataset.
    generated_at : str
        UTC timestamp, ISO-8601.
    """

    dataset_id: str
    family: str
    generator: str
    seed: int
    path: str
    params: dict
    arrays: dict[str, str]
    truth_sha256: str
    package_version: str
    generated_at: str


def sha256_file(path: Path | str) -> str:
    """Hash a file's bytes.

    Parameters
    ----------
    path : pathlib.Path or str
        File to hash.

    Returns
    -------
    str
        Hex digest, matching what ``shasum -a 256`` reports.

    Notes
    -----
    The ``.npy`` file is hashed rather than ``ndarray.tobytes()``: the digest is then
    verifiable with ordinary command-line tools, and it also covers the header, so a
    changed dtype or shape cannot hide behind identical payload bytes.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def make_entry(
    dataset_dir: Path | str,
    output_root: Path | str,
    ground_truth: GroundTruth,
    *,
    generated_at: str | None = None,
) -> ManifestEntry:
    """Build a registry entry for a dataset that is already on disk.

    Parameters
    ----------
    dataset_dir : pathlib.Path or str
        Directory returned by :func:`tdadyn.core.io.save_dataset`.
    output_root : pathlib.Path or str
        Root the recorded path is relative to.
    ground_truth : GroundTruth
        Truth record of the dataset.
    generated_at : str, optional
        Override the timestamp; used by tests.

    Returns
    -------
    ManifestEntry
        Entry with a sha256 for every array file in the directory.
    """
    dataset_dir = Path(dataset_dir)
    output_root = Path(output_root)
    arrays = {
        file.name: sha256_file(file) for file in sorted(dataset_dir.glob("*.npy"))
    }
    if not arrays:
        raise ValueError(f"{dataset_dir} holds no arrays; nothing to register")
    if not (dataset_dir / GROUND_TRUTH_FILE).exists():
        raise ValueError(f"{dataset_dir} has no {GROUND_TRUTH_FILE}")

    relative = dataset_dir.resolve().relative_to(output_root.resolve())
    return ManifestEntry(
        dataset_id=dataset_dir.name,
        family=ground_truth.family,
        generator=ground_truth.generator,
        seed=int(ground_truth.seed),
        path=relative.as_posix(),
        params=jsonify("params", ground_truth.params),
        arrays=arrays,
        truth_sha256=truth_digest(read_json(dataset_dir / GROUND_TRUTH_FILE)),
        package_version=package_version(),
        generated_at=generated_at or _utc_now(),
    )


def entry_payload(entry: ManifestEntry | dict[str, Any]) -> dict[str, Any]:
    """Strip the fields that legitimately differ between two identical rebuilds.

    Parameters
    ----------
    entry : ManifestEntry or dict
        Entry to reduce.

    Returns
    -------
    dict
        The entry without ``generated_at`` and ``package_version``.

    Notes
    -----
    The manifest as a whole is not byte-reproducible — it records when it was built.
    "A rebuild reproduces the same hashes" is therefore checked on this reduced form.
    """
    payload = dict(asdict(entry) if isinstance(entry, ManifestEntry) else entry)
    payload.pop("generated_at", None)
    payload.pop("package_version", None)
    return payload


def truth_digest(payload: dict[str, Any]) -> str:
    """sha256 of a ground-truth record, over its canonical serialisation.

    Parameters
    ----------
    payload : dict
        Parsed ``ground_truth.json``.

    Returns
    -------
    str
        Hex digest of the record dumped with sorted keys and fixed separators.

    Notes
    -----
    Arrays are hashed by file bytes, deliberately (see :func:`sha256_file`); the truth is
    not.  Its file is pretty-printed, so byte hashing would fold indentation, key order and
    float ``repr`` into the digest, and a cosmetic reformat would read as changed data.
    Hashing the parsed object makes the digest track what the truth *says*.

    This is what makes "a rebuild reproduces the same hashes" cover the expected Betti
    numbers, the event log and the notes — the whole value of these datasets, and exactly
    what the array digests do not touch.

    No key of the record depends on the environment: there is no timestamp, no path and no
    package version among them.  Should one appear, it must be excluded here by an explicit
    list rather than dropped from the file.
    """
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_manifest(output_root: Path | str, entries: list[ManifestEntry]) -> Path:
    """Replace the manifest with exactly these entries.

    Parameters
    ----------
    output_root : pathlib.Path or str
        Root the manifest sits in.
    entries : list of ManifestEntry
        The complete registry after a full rebuild.

    Returns
    -------
    pathlib.Path
        Path of the written manifest.

    Notes
    -----
    A full rebuild replaces rather than accumulates.  :func:`update_manifest` never prunes,
    so after a configuration change it would keep rows for datasets that no longer exist,
    and "the rebuild reproduces the same hashes" would then compare against a registry
    describing a directory tree nobody has.
    """
    rows = sorted((asdict(entry) for entry in entries), key=lambda item: item["dataset_id"])
    path = manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"format_version": FORMAT_VERSION, "datasets": rows})
    return path


def manifest_path(output_root: Path | str) -> Path:
    """Return the path of the registry file under ``output_root``."""
    return Path(output_root) / MANIFEST_FILE


def read_manifest(output_root: Path | str) -> dict[str, Any]:
    """Read the registry, returning an empty one if the file does not exist.

    Parameters
    ----------
    output_root : pathlib.Path or str
        Root that holds ``manifest.json``.

    Returns
    -------
    dict
        Mapping with ``format_version`` and ``datasets``.
    """
    path = manifest_path(output_root)
    if not path.exists():
        return {"format_version": FORMAT_VERSION, "datasets": []}
    manifest = read_json(path)
    if not isinstance(manifest, dict) or "datasets" not in manifest:
        raise ValueError(f"{path} is not a manifest")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format_version is {manifest.get('format_version')!r}, "
            f"this package writes {FORMAT_VERSION}"
        )
    return manifest


def update_manifest(output_root: Path | str, entry: ManifestEntry) -> Path:
    """Insert or replace one entry and rewrite the registry.

    Parameters
    ----------
    output_root : pathlib.Path or str
        Root that holds ``manifest.json``.
    entry : ManifestEntry
        Entry to record; an existing entry with the same ``dataset_id`` is replaced.

    Returns
    -------
    pathlib.Path
        Path of the written registry.
    """
    manifest = read_manifest(output_root)
    datasets = [item for item in manifest["datasets"] if item["dataset_id"] != entry.dataset_id]
    datasets.append(asdict(entry))
    datasets.sort(key=lambda item: item["dataset_id"])
    manifest["datasets"] = datasets

    path = manifest_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return path


def package_version() -> str:
    """Return the installed version of the package, or ``"unknown"``."""
    try:
        return version("tdadyn")
    except PackageNotFoundError:
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
