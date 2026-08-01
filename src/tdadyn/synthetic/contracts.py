"""Dataclasses that define the contract between generators, storage and tests.

Every generator returns one of the ``*Dataset`` containers defined here and nothing
in this module touches the filesystem; see ``docs/data.md`` for the specification.

The containers are ``frozen`` but declared with ``eq=False``.  Python's generated
``__eq__`` compares fields with ``==``, which raises ``ValueError: ambiguous truth
value`` on the ``numpy`` fields.  With ``eq=False`` comparison falls back to
identity, so a test that compares two distinct objects with ``==`` fails loudly
instead of silently succeeding.  Use :func:`dataset_equal` / :func:`ground_truth_equal`
for real comparison — they are bitwise on array data, as round-trip must be.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

FORMAT_VERSION = 1
"""Version of the on-disk ground-truth schema. Bumped whenever the schema changes."""

FAMILIES = ("static_cloud", "dynamic_graph", "series", "noisy", "null")
"""Dataset families from ``docs/task.md``; also the top-level directories on disk."""

EVENT_TYPES = ("birth", "death")
"""The only admissible homological events.  A barcode has no "split" or "merge":
those are graph-level narratives and belong in :attr:`Event.narrative`."""


@dataclass(frozen=True)
class Event:
    """A single homological event in a dynamic dataset.

    Parameters
    ----------
    snapshot : int
        Index of the snapshot at which the event happens.
    dim : int
        Homology dimension the event applies to.
    type : {'birth', 'death'}
        Direction of the event.  Graph-level wording such as "two components merge"
        must not appear here: a merge is the *death* of an :math:`H_0` class.
    feature_id : str
        Stable identifier of the feature, so that a birth and its death can be paired.
    narrative : str
        Human-readable description, free-form.
    """

    snapshot: int
    dim: int
    type: str
    feature_id: str
    narrative: str

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(
                f"Event.type must be one of {EVENT_TYPES}, got {self.type!r}. "
                "Graph-level wording belongs in Event.narrative."
            )
        _check_nonneg_int("Event.snapshot", self.snapshot)
        _check_nonneg_int("Event.dim", self.dim)
        if not self.feature_id:
            raise ValueError("Event.feature_id must be a non-empty string")


@dataclass(frozen=True, eq=False)
class GroundTruth:
    """Machine-readable truth for a generated dataset.

    ``dataset_id`` is deliberately absent: it is a function of
    ``(generator, params, seed)`` and is derived by :mod:`tdadyn.synthetic.naming`.
    Storing it as well would allow the identifier and the parameters to drift apart.

    Parameters
    ----------
    format_version : int
        Must equal :data:`FORMAT_VERSION`.
    family : str
        One of :data:`FAMILIES`.
    generator : str
        Name of the generating function, e.g. ``"torus"``.
    params : dict
        The actual keyword arguments of the generator call — not the whole config
        section.  Only these reproduce the dataset, and the directory name is built
        from them.
    seed : int
        Seed of the derived stream the dataset was generated with.
    coeff_field : int, optional
        Field of homology coefficients the expectations below refer to.  Required
        wherever the answer depends on it (see the Klein bottle in ``docs/method.md``).
    complex : str, optional
        Complex the truth is stated for, e.g. ``"flag"`` for thresholded graphs.
    expected_betti : dict of int to int, optional
        Family 1.  Keys must be the contiguous range ``0..max_dim``; a dimension with
        answer zero is stated explicitly, since a missing key is indistinguishable
        from a forgotten one.
    betti_curve : dict of int to list of int, optional
        Family 2, one value per snapshot.  Computed by the reference implementation,
        never by the main pipeline.
    events : list of Event, optional
        Family 2.  Independent of ``betti_curve``: the scheduler's intent, which the
        tests check against the curve.
    community_labels : numpy.ndarray, optional
        Family 3, community label per node per time.
    noise_level : dict, optional
        Family 4, the noise parameters that were applied.
    notes : str
        Free-form remarks, e.g. known sampling non-uniformity.
    """

    format_version: int
    family: str
    generator: str
    params: dict
    seed: int
    coeff_field: int | None = None
    complex: str | None = None
    expected_betti: dict[int, int] | None = None
    betti_curve: dict[int, list[int]] | None = None
    events: list[Event] | None = None
    community_labels: np.ndarray | None = None
    noise_level: dict | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError(
                f"GroundTruth.format_version is {self.format_version}, "
                f"this package writes {FORMAT_VERSION}"
            )
        if self.family not in FAMILIES:
            raise ValueError(f"GroundTruth.family must be one of {FAMILIES}, got {self.family!r}")
        if not self.generator:
            raise ValueError("GroundTruth.generator must be a non-empty string")
        if not isinstance(self.params, dict):
            raise TypeError(f"GroundTruth.params must be a dict, got {type(self.params).__name__}")
        _check_nonneg_int("GroundTruth.seed", self.seed)
        if self.coeff_field is not None and (not _is_int(self.coeff_field) or self.coeff_field < 2):
            raise ValueError(f"GroundTruth.coeff_field must be an int >= 2, got {self.coeff_field!r}")

        if self.expected_betti is not None:
            _check_dim_keys("GroundTruth.expected_betti", self.expected_betti)
            for dim, value in self.expected_betti.items():
                _check_nonneg_int(f"GroundTruth.expected_betti[{dim}]", value)

        if self.betti_curve is not None:
            _check_dim_keys("GroundTruth.betti_curve", self.betti_curve)
            lengths = set()
            for dim, curve in self.betti_curve.items():
                if not isinstance(curve, list):
                    raise TypeError(f"GroundTruth.betti_curve[{dim}] must be a list")
                for index, value in enumerate(curve):
                    _check_nonneg_int(f"GroundTruth.betti_curve[{dim}][{index}]", value)
                lengths.add(len(curve))
            if len(lengths) > 1:
                raise ValueError(
                    f"GroundTruth.betti_curve has different lengths per dimension: {sorted(lengths)}"
                )

        if self.events is not None:
            if not isinstance(self.events, list) or not all(
                isinstance(event, Event) for event in self.events
            ):
                raise TypeError("GroundTruth.events must be a list of Event")
            # An event outside the curve means the scheduler and the curve disagree,
            # which is a generator bug and must not reach disk.
            if self.betti_curve is not None:
                n_snapshots = len(next(iter(self.betti_curve.values())))
                for event in self.events:
                    if event.snapshot >= n_snapshots:
                        raise ValueError(
                            f"event at snapshot {event.snapshot} is outside the "
                            f"betti_curve of length {n_snapshots}"
                        )
                    if event.dim not in self.betti_curve:
                        raise ValueError(
                            f"event in dimension {event.dim} has no corresponding betti_curve"
                        )

        if self.community_labels is not None:
            if not isinstance(self.community_labels, np.ndarray):
                raise TypeError("GroundTruth.community_labels must be a numpy array")
            # Fixed at int64 so that the JSON round-trip cannot change the dtype:
            # labels are categorical, and a dtype change would break bitwise comparison.
            if self.community_labels.dtype != np.int64:
                raise TypeError(
                    f"GroundTruth.community_labels must be int64, got {self.community_labels.dtype}"
                )
        if self.noise_level is not None and not isinstance(self.noise_level, dict):
            raise TypeError("GroundTruth.noise_level must be a dict")
        if not isinstance(self.notes, str):
            raise TypeError("GroundTruth.notes must be a string")


@dataclass(frozen=True, eq=False)
class PointCloudDataset:
    """A point cloud sampled from a manifold, optionally with a weighted network.

    Parameters
    ----------
    points : numpy.ndarray
        Shape ``(n_points, dim)``, float64.
    ground_truth : GroundTruth
        Truth for this dataset.
    weights : numpy.ndarray, optional
        Shape ``(n_points, n_points)``, symmetric similarity matrix.
    """

    points: np.ndarray
    ground_truth: GroundTruth
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        _check_array("PointCloudDataset.points", self.points, ndim=2)
        if self.weights is not None:
            _check_array("PointCloudDataset.weights", self.weights, ndim=2)
            _check_square_symmetric("PointCloudDataset.weights", self.weights)
            if self.weights.shape[0] != self.points.shape[0]:
                raise ValueError(
                    f"weights is {self.weights.shape} but there are "
                    f"{self.points.shape[0]} points"
                )


@dataclass(frozen=True, eq=False)
class DynamicGraphDataset:
    """A sequence of weighted graph snapshots.

    Parameters
    ----------
    snapshots : numpy.ndarray
        Shape ``(T, n_nodes, n_nodes)``, float64, each slice symmetric.
    ground_truth : GroundTruth
        Truth for this dataset.
    """

    snapshots: np.ndarray
    ground_truth: GroundTruth

    def __post_init__(self) -> None:
        _check_array("DynamicGraphDataset.snapshots", self.snapshots, ndim=3)
        _check_square_symmetric("DynamicGraphDataset.snapshots", self.snapshots)


@dataclass(frozen=True, eq=False)
class SeriesDataset:
    """A multivariate time series together with its true coupling schedule.

    Parameters
    ----------
    series : numpy.ndarray
        Shape ``(n_channels, n_timepoints)``, float64.  Channel first — the fMRI
        module uses the opposite convention and converts at the module boundary.
    coupling : numpy.ndarray
        Shape ``(T_schedule, n_channels, n_channels)``, float64, each slice symmetric.
    ground_truth : GroundTruth
        Truth for this dataset.
    """

    series: np.ndarray
    coupling: np.ndarray
    ground_truth: GroundTruth

    def __post_init__(self) -> None:
        _check_array("SeriesDataset.series", self.series, ndim=2)
        _check_array("SeriesDataset.coupling", self.coupling, ndim=3)
        _check_square_symmetric("SeriesDataset.coupling", self.coupling)
        if self.coupling.shape[1] != self.series.shape[0]:
            raise ValueError(
                f"coupling is over {self.coupling.shape[1]} nodes but the series has "
                f"{self.series.shape[0]} channels"
            )


Dataset = PointCloudDataset | DynamicGraphDataset | SeriesDataset


def values_equal(left: Any, right: Any) -> bool:
    """Compare two values, treating arrays bitwise and recursing into containers.

    Parameters
    ----------
    left, right : Any
        Values to compare.

    Returns
    -------
    bool
        True if the values are equal.  Arrays must agree in dtype, shape and raw
        bytes: a round-trip through ``.npy`` is expected to be exact, so an
        approximate comparison would hide real losses.
    """
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return False
        if left.keys() != right.keys():
            return False
        return all(values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            return False
        if len(left) != len(right):
            return False
        return all(values_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def ground_truth_equal(left: GroundTruth, right: GroundTruth) -> bool:
    """Compare two ground-truth records field by field.

    Parameters
    ----------
    left, right : GroundTruth
        Records to compare.

    Returns
    -------
    bool
        True if every field is equal, arrays bitwise.
    """
    return _dataclass_equal(left, right, GroundTruth)


def dataset_equal(left: Dataset, right: Dataset) -> bool:
    """Compare two datasets including their ground truth.

    Parameters
    ----------
    left, right : PointCloudDataset or DynamicGraphDataset or SeriesDataset
        Datasets to compare; different types are never equal.

    Returns
    -------
    bool
        True if every array is bitwise identical and the truth records agree.
    """
    if type(left) is not type(right):
        return False
    return _dataclass_equal(left, right, type(left))


def _dataclass_equal(left: Any, right: Any, expected_type: type) -> bool:
    if not isinstance(left, expected_type) or not isinstance(right, expected_type):
        raise TypeError(
            f"expected two {expected_type.__name__} instances, got "
            f"{type(left).__name__} and {type(right).__name__}"
        )
    for field in fields(expected_type):
        a = getattr(left, field.name)
        b = getattr(right, field.name)
        if isinstance(a, GroundTruth) or isinstance(b, GroundTruth):
            if not (isinstance(a, GroundTruth) and isinstance(b, GroundTruth)):
                return False
            if not ground_truth_equal(a, b):
                return False
        elif not values_equal(a, b):
            return False
    return True


def _is_int(value: Any) -> bool:
    # numpy integers pass isinstance checks against np.integer, not int; bool is an
    # int subclass and must not sneak through as a count.
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _check_nonneg_int(name: str, value: Any) -> None:
    if not _is_int(value):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _check_dim_keys(name: str, mapping: dict) -> None:
    """Require homology-dimension keys to be the contiguous range starting at zero."""
    if not isinstance(mapping, dict):
        raise TypeError(f"{name} must be a dict keyed by homology dimension")
    for key in mapping:
        if not _is_int(key):
            raise TypeError(f"{name} has non-integer key {key!r}; JSON keys are converted on load")
    expected = set(range(len(mapping)))
    if set(mapping) != expected:
        raise ValueError(
            f"{name} keys must be the contiguous range {sorted(expected)}, got "
            f"{sorted(mapping)}. State a dimension explicitly even when the answer is 0."
        )


def _check_array(name: str, array: Any, *, ndim: int) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if array.dtype != np.float64:
        raise TypeError(f"{name} must be float64, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} axes, got shape {array.shape}")


def _check_square_symmetric(name: str, array: np.ndarray) -> None:
    """Check that the trailing two axes are square and exactly symmetric.

    Exact rather than approximate: every matrix in this package is built symmetric
    by construction, so a floating-point asymmetry is a bug, not a tolerance issue.
    """
    if array.shape[-1] != array.shape[-2]:
        raise ValueError(f"{name} must be square in its last two axes, got shape {array.shape}")
    transposed = np.swapaxes(array, -1, -2)
    if not np.array_equal(array, transposed):
        raise ValueError(f"{name} must be symmetric")
