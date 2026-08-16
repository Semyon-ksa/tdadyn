"""Writing and reading dataset directories.

Generators never write to disk themselves; saving goes through :func:`save_dataset`,
which writes a directory atomically.  ``docs/data.md`` forbids adding anything to an
existing dataset directory — a rebuild replaces it whole — so the write goes to a
temporary directory next to the target and is moved into place in one step.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    NetworkDataset,
    Dataset,
    DynamicGraphDataset,
    GroundTruth,
    PointCloudDataset,
    SeriesDataset,
)
from .naming import make_dataset_id
from .schema import ground_truth_from_json_dict, ground_truth_to_json_dict

GROUND_TRUTH_FILE = "ground_truth.json"

_ARRAY_FILES: dict[type, tuple[tuple[str, str, bool], ...]] = {
    # dataset type -> (attribute, file name, required)
    PointCloudDataset: (("points", "points.npy", True), ("weights", "weights.npy", False)),
    DynamicGraphDataset: (("snapshots", "snapshots.npy", True),),
    SeriesDataset: (("series", "series.npy", True), ("coupling", "coupling.npy", True)),
    NetworkDataset: (("weights", "weights.npy", True),),
}

# Load dispatches on the array files present, not on the family: a noisy point cloud
# and a noisy dynamic graph both carry family "noisy".
_LOAD_BY_MARKER: dict[str, type] = {
    "points.npy": PointCloudDataset,
    "snapshots.npy": DynamicGraphDataset,
    "series.npy": SeriesDataset,
}

_FALLBACK_MARKER = ("weights.npy", NetworkDataset)
"""Consulted only when none of the primary markers is present.

``weights.npy`` cannot be a primary marker: a point cloud may legitimately carry one
beside ``points.npy``, and the primary markers are mutually exclusive on purpose — two of
them in one directory means the directory is malformed.  A directory holding *only*
weights is a standalone network, which is how family 4 stores a jittered one
(``contracts.NetworkDataset``)."""


def dataset_arrays(dataset: Dataset) -> dict[str, np.ndarray]:
    """List the arrays a dataset stores, keyed by file name.

    Parameters
    ----------
    dataset : PointCloudDataset or DynamicGraphDataset or SeriesDataset
        Dataset to inspect.

    Returns
    -------
    dict of str to numpy.ndarray
        File name to array, omitting optional arrays that are absent.
    """
    try:
        spec = _ARRAY_FILES[type(dataset)]
    except KeyError:
        raise TypeError(f"unsupported dataset type {type(dataset).__name__}") from None
    arrays = {}
    for attribute, file_name, required in spec:
        array = getattr(dataset, attribute)
        if array is None:
            if required:
                raise ValueError(f"{type(dataset).__name__}.{attribute} is required")
            continue
        arrays[file_name] = array
    return arrays


def save_dataset(
    dataset: Dataset,
    root: Path | str,
    *,
    float_precision: int,
    overwrite: bool = False,
) -> Path:
    """Write a dataset to its own directory under ``root``.

    Parameters
    ----------
    dataset : PointCloudDataset or DynamicGraphDataset or SeriesDataset
        Dataset to store.
    root : pathlib.Path or str
        Family directory, e.g. ``data/synthetic/static_cloud``.
    float_precision : int
        Significant digits for floats in the directory name, from
        ``config.naming.float_precision``.
    overwrite : bool, default False
        Replace an existing dataset directory.  Without it an existing directory is
        an error, so that a rebuild with changed parameters cannot pass unnoticed.

    Returns
    -------
    pathlib.Path
        The dataset directory.
    """
    root = Path(root)
    dataset_id = make_dataset_id(dataset.ground_truth, float_precision=float_precision)
    target = root / dataset_id
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"{target} already exists; pass overwrite=True to rebuild it whole. "
            "Datasets are never updated in place."
        )

    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".tmp-{dataset_id}-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir()
        for file_name, array in dataset_arrays(dataset).items():
            np.save(staging / file_name, np.ascontiguousarray(array, dtype=np.float64))
        write_json(
            staging / GROUND_TRUTH_FILE,
            ground_truth_to_json_dict(dataset.ground_truth, dataset_id),
        )
        if target.exists():
            _remove_dataset_dir(target, root)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def load_dataset(path: Path | str, *, float_precision: int | None = None) -> Dataset:
    """Read a dataset directory back into its dataclass.

    Parameters
    ----------
    path : pathlib.Path or str
        Dataset directory.
    float_precision : int, optional
        When given, the identifier is recomputed from the loaded parameters and
        compared with the stored one, which catches a hand-edited ``params``.

    Returns
    -------
    PointCloudDataset or DynamicGraphDataset or SeriesDataset
        The dataset, with arrays identical to the ones that were written.
    """
    path = Path(path)
    ground_truth = load_ground_truth(path, float_precision=float_precision)

    present = [marker for marker in _LOAD_BY_MARKER if (path / marker).exists()]
    if len(present) > 1:
        raise ValueError(
            f"{path}: expected exactly one of {sorted(_LOAD_BY_MARKER)}, found {sorted(present)}"
        )
    if present:
        dataset_type = _LOAD_BY_MARKER[present[0]]
    elif (path / _FALLBACK_MARKER[0]).exists():
        dataset_type = _FALLBACK_MARKER[1]
    else:
        raise ValueError(
            f"{path}: expected one of {sorted(_LOAD_BY_MARKER)} or {_FALLBACK_MARKER[0]}, "
            "found none"
        )
    arrays = {}
    for attribute, file_name, required in _ARRAY_FILES[dataset_type]:
        file_path = path / file_name
        if file_path.exists():
            arrays[attribute] = np.load(file_path, allow_pickle=False)
        elif required:
            raise FileNotFoundError(f"{file_path} is missing")
    return dataset_type(ground_truth=ground_truth, **arrays)


def load_ground_truth(path: Path | str, *, float_precision: int | None = None) -> GroundTruth:
    """Read and validate ``ground_truth.json`` from a dataset directory.

    Parameters
    ----------
    path : pathlib.Path or str
        Dataset directory.
    float_precision : int, optional
        When given, the identifier is recomputed and compared with the stored one.

    Returns
    -------
    GroundTruth
        The validated record.
    """
    path = Path(path)
    payload = read_json(path / GROUND_TRUTH_FILE)
    ground_truth = ground_truth_from_json_dict(payload)

    stored_id = payload["dataset_id"]
    if stored_id != path.name:
        raise ValueError(
            f"{path}: directory name does not match dataset_id {stored_id!r}; "
            "the directory was renamed by hand"
        )
    if float_precision is not None:
        derived_id = make_dataset_id(ground_truth, float_precision=float_precision)
        if derived_id != stored_id:
            raise ValueError(
                f"{path}: dataset_id {stored_id!r} does not follow from the stored "
                f"params, which give {derived_id!r}"
            )
    return ground_truth


def write_json(path: Path | str, payload: Any) -> None:
    """Write JSON in the project's canonical form.

    Parameters
    ----------
    path : pathlib.Path or str
        Destination file.
    payload : Any
        JSON-serialisable object.

    Notes
    -----
    UTF-8 without escapes, sorted keys, two-space indent and a trailing newline, per
    ``docs/data.md``.  Sorted keys are what make a rebuild byte-identical.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def read_json(path: Path | str) -> Any:
    """Read a JSON file written by :func:`write_json`.

    Parameters
    ----------
    path : pathlib.Path or str
        File to read.

    Returns
    -------
    Any
        The parsed object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _remove_dataset_dir(target: Path, root: Path) -> None:
    """Delete a dataset directory, refusing anything that is not one.

    Guards a recursive delete: the path must sit directly under ``root`` and must
    look like a dataset, so a mistyped root cannot take an unrelated tree with it.
    """
    resolved_target = target.resolve()
    resolved_root = root.resolve()
    if resolved_target.parent != resolved_root:
        raise ValueError(f"refusing to remove {target}: it is not directly under {root}")
    if not (resolved_target / GROUND_TRUTH_FILE).exists():
        raise ValueError(
            f"refusing to remove {target}: it has no {GROUND_TRUTH_FILE} and is not a dataset"
        )
    shutil.rmtree(resolved_target)
