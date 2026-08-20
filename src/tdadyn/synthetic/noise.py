"""Family 4: controlled noise applied to the reference datasets.

The level is the truth here — each function takes one number from ``config.noise`` and the
dataset records which mechanism was applied at which level.  Two properties are
load-bearing and are checked by the tests rather than assumed:

* **The draw happens even at level zero.**  ``static_clouds._add_ambient_noise`` sets the
  precedent: consuming the same numbers regardless of the level keeps the underlying
  object bitwise identical across levels, so the clean dataset really is the same object
  the noisy ones are compared against.  Skipping the draw at zero would shift every
  subsequent stream and quietly make "level 0" a different sample.
* **Edges are counted on the upper triangle.**  ``docs/method.md`` §5 names the double
  count as a silent source of error in densities and in the :math:`p^{\\pm}` fractions;
  a fraction computed over the full matrix is understated by half.

The edge mechanisms take the dataset's own ``edge_rule``.  For a metric dataset the
snapshot holds distances, so *adding* an edge means pushing a distance **below**
:math:`\\varepsilon`, not raising a weight — the opposite arithmetic from a similarity
dataset (``docs/method.md`` §2.3).
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.contracts import (
    EDGE_RULES,
    FORMAT_VERSION,
    DynamicGraphDataset,
    GroundTruth,
    NetworkDataset,
    PointCloudDataset,
)
from tdadyn.core.naming import register_name_order
from tdadyn.core.rng import base_seed_of

NOISE_KINDS = ("point_jitter", "weight_jitter", "spurious_edges", "missing_edges")
"""The four mechanisms of ``docs/method.md`` §4, in the order they are listed there."""

for _kind in NOISE_KINDS:
    # The base dataset's full identifier goes into the name, not just its generator name:
    # a noisy set built from a torus with a different R must not land in the same
    # directory, and only the full id carries R (docs/data.md, 1.1).
    register_name_order(_kind, ("base", "level", "repeat"))
del _kind

NOISY_NOTES = (
    "Семейство 4: уровень шума — это истина набора. Разыгрывание происходит и при нулевом "
    "уровне, поэтому при level=0 набор побитово совпадает с исходным, а поток случайности "
    "сдвигается одинаково на всех уровнях. Доли p+ и p- считаются по верхнему треугольнику "
    "(docs/method.md, 5). Граница устойчивости — d_B <= 2*d_H, а не d_H (docs/method.md, 4.1)."
)


def make_jittered_cloud(
    base_points: np.ndarray,
    base_id: str,
    level: float,
    repeat: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Displace a family-1 cloud, storing the points alone.

    Parameters
    ----------
    base_points : numpy.ndarray
        Points of the clean dataset, shape ``(n_points, dim)``.
    base_id : str
        ``dataset_id`` of the clean dataset; recorded so the noisy set is reproducible.
    level : float
        The level from ``config.noise.point_jitter_sigma``.
    repeat : int
        Realisation index, part of the name so repeats do not overwrite each other.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points only, with ``family="noisy"`` and ``truth_kind="noise_level"``.

    Notes
    -----
    No network is attached, deliberately.  Once the points have moved, the network built
    from the *clean* ones no longer describes them — measured, the weights differ by up to
    0.36 at level 0.10 — and a directory holding both would be a pair that disagrees with
    itself.  The network is derivable from the stored points through the same calibrated
    path family 1 uses, so nothing is lost by leaving it out.
    """
    points = jitter_points(base_points, sigma=level, rng=rng)
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("point_jitter", base_id, level, repeat, rng),
    )


def make_jittered_network(
    base_weights: np.ndarray,
    base_id: str,
    level: float,
    weight_range: tuple[float, float],
    repeat: int,
    rng: np.random.Generator,
) -> NetworkDataset:
    """Perturb a weighted network, storing the weights alone.

    Parameters
    ----------
    base_weights : numpy.ndarray
        The clean network, square and symmetric.
    base_id : str
        ``dataset_id`` of the clean dataset the network was built from.
    level : float
        The level from ``config.noise.weight_jitter_sigma``.
    weight_range : tuple of float
        Admissible range to clip into.  Derived from the transform that produced the
        weights, never configured: for a Gaussian kernel it is :math:`(0, 1]` by
        definition of :math:`\\exp(-d^2/2\\sigma^2)`, and a configurable range would invite
        a value the kernel can never produce.
    repeat : int
        Realisation index.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    NetworkDataset
        Weights only, with ``family="noisy"`` and ``truth_kind="noise_level"``.

    Notes
    -----
    The points are not stored, for the mirror of the reason in :func:`make_jittered_cloud`:
    they are untouched while the weights are deliberately corrupted, so the pair on disk
    would again disagree.  The same rule covers the whole family — **only the perturbed
    array reaches disk, and the source is recovered through** ``noise_level["base"]``.
    """
    low, high = weight_range
    if not np.all((base_weights >= low) & (base_weights <= high)):
        raise ValueError(
            f"base weights fall outside the range {weight_range} the transform can "
            f"produce (observed {base_weights.min():.4f}..{base_weights.max():.4f}); "
            "the range is a property of the transform, so this means the wrong one"
        )
    weights = jitter_weights(base_weights, sigma=level, low=low, high=high, rng=rng)
    return NetworkDataset(
        weights=weights,
        ground_truth=_ground_truth("weight_jitter", base_id, level, repeat, rng),
    )


def make_noisy_snapshots(
    base_snapshots: np.ndarray,
    base_id: str,
    kind: str,
    level: float,
    edge_rule: str,
    edge_threshold: float,
    repeat: int,
    rng: np.random.Generator,
) -> DynamicGraphDataset:
    """Apply ``spurious_edges`` or ``missing_edges`` to a family-2 sequence.

    Parameters
    ----------
    base_snapshots : numpy.ndarray
        Snapshots of the clean dataset, shape ``(T, n, n)``.
    base_id : str
        ``dataset_id`` of the clean dataset.
    kind : {'spurious_edges', 'missing_edges'}
        Mechanism to apply.
    level : float
        :math:`p^+` or :math:`p^-`.
    edge_rule, edge_threshold : str, float
        Taken from the **base dataset's** ground truth, never assumed: the two scenarios
        of family 2 disagree about what a snapshot holds (``docs/method.md`` §2.3).
    repeat : int
        Realisation index.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    DynamicGraphDataset
        With ``family="noisy"`` and ``truth_kind="noise_level"``.

    Notes
    -----
    Every snapshot gets its own draw, so the noise is independent across time rather than
    a single perturbation repeated — the latter would leave the event log intact by
    construction and make the dataset useless as a stress test.
    """
    if kind not in ("spurious_edges", "missing_edges"):
        raise ValueError(f"kind must be spurious_edges or missing_edges, got {kind!r}")
    apply = add_spurious_edges if kind == "spurious_edges" else drop_edges

    noisy = np.empty_like(base_snapshots)
    for index, snapshot in enumerate(base_snapshots):
        noisy[index] = apply(
            snapshot,
            fraction=level,
            edge_rule=edge_rule,
            edge_threshold=edge_threshold,
            rng=rng,
        )
    return DynamicGraphDataset(
        snapshots=noisy,
        ground_truth=_ground_truth(
            kind, base_id, level, repeat, rng, edge_rule=edge_rule, edge_threshold=edge_threshold
        ),
    )


def _ground_truth(
    kind: str,
    base_id: str,
    level: float,
    repeat: int,
    rng: np.random.Generator,
    *,
    edge_rule: str | None = None,
    edge_threshold: float | None = None,
) -> GroundTruth:
    """Truth of a noisy dataset: which mechanism, at which level, over which base."""
    if not base_id:
        raise ValueError("base_id must be a non-empty dataset identifier")
    if not isinstance(repeat, (int, np.integer)) or isinstance(repeat, bool) or repeat < 0:
        raise ValueError(f"repeat must be a non-negative int, got {repeat!r}")
    return GroundTruth(
        format_version=FORMAT_VERSION,
        family="noisy",
        truth_kind="noise_level",
        generator=kind,
        params={"base": base_id, "level": float(level), "repeat": int(repeat)},
        seed=base_seed_of(rng),
        edge_rule=edge_rule,
        edge_threshold=edge_threshold,
        # String keys throughout: `jsonify` refuses anything else, and the round-trip
        # must return the same dict.
        noise_level={"kind": kind, "level": float(level), "base": base_id, "repeat": int(repeat)},
        notes=NOISY_NOTES,
    )


def jitter_points(points: np.ndarray, *, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Displace every point by an isotropic Gaussian in the ambient space.

    Parameters
    ----------
    points : numpy.ndarray
        Shape ``(n_points, dim)``.
    sigma : float
        Standard deviation of the displacement; may be 0.0.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    numpy.ndarray
        Displaced points, same shape and dtype.

    Notes
    -----
    The draw is made at every level, including zero, and then scaled.  At ``sigma = 0``
    the result is bitwise equal to the input while the stream advances exactly as it does
    at any other level.
    """
    _check_finite("points", points)
    _check_level("sigma", sigma)
    displacement = rng.normal(0.0, 1.0, size=points.shape)
    return points + displacement * sigma


def jitter_weights(
    weights: np.ndarray,
    *,
    sigma: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add Gaussian noise to edge weights and clip back into the admissible range.

    Parameters
    ----------
    weights : numpy.ndarray
        Square symmetric weight matrix.
    sigma : float
        Standard deviation of the additive noise.
    low, high : float
        Admissible range; ``docs/method.md`` §4 clips into it explicitly, because a weight
        outside it breaks the similarity-to-distance transforms downstream.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    numpy.ndarray
        Jittered weights, symmetric, with the diagonal left exactly as it was.

    Notes
    -----
    The diagonal is not an edge, so the noise does not touch it.  Zeroing it here would
    silently rewrite self-similarity — which a Gaussian kernel sets to 1 — and level zero
    would then stop reproducing the base.
    """
    _check_square("weights", weights)
    _check_level("sigma", sigma)
    if not high > low:
        raise ValueError(f"need low < high, got low={low}, high={high}")

    n_nodes = weights.shape[0]
    upper = np.triu_indices(n_nodes, 1)
    # Drawn on the upper triangle only, then mirrored: drawing on the full matrix would
    # consume twice the numbers and give the two halves independent noise.
    draw = rng.normal(0.0, 1.0, size=len(upper[0]))
    off_diagonal = np.zeros_like(weights)
    off_diagonal[upper] = np.clip(weights[upper] + draw * sigma, low, high)
    jittered = off_diagonal + off_diagonal.T
    np.fill_diagonal(jittered, np.diagonal(weights))
    return jittered


def add_spurious_edges(
    snapshot: np.ndarray,
    *,
    fraction: float,
    edge_rule: str,
    edge_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turn a fraction of the **absent** pairs into edges.

    Parameters
    ----------
    snapshot : numpy.ndarray
        Square symmetric matrix of similarities or distances.
    fraction : float
        :math:`p^+`, a fraction of the absent pairs — not of the present ones
        (``docs/method.md`` §4).
    edge_rule : {'weight_above', 'distance_within'}
        Which convention the snapshot follows.
    edge_threshold : float
        The threshold or radius the rule is applied at.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    numpy.ndarray
        The snapshot with ``round(fraction * n_absent)`` extra edges, symmetric.

    Notes
    -----
    The new values are drawn from the **empirical distribution of the existing** entries
    (``docs/method.md`` §4), not from a parametric law: a spurious edge should be
    indistinguishable from a real one by its weight alone.
    """
    present, absent, upper = _edge_partition(snapshot, edge_rule, edge_threshold)
    _check_level("fraction", fraction, maximum=1.0)

    n_new = int(round(fraction * absent.sum()))
    chosen = _choose(absent, n_new, rng)
    if present.sum() == 0 and n_new > 0:
        raise ValueError("no existing edges to draw spurious weights from")
    donor = snapshot[upper][present]
    values = rng.choice(donor, size=n_new, replace=True) if n_new else np.empty(0)
    return _rebuild(snapshot, upper, chosen, values)


def drop_edges(
    snapshot: np.ndarray,
    *,
    fraction: float,
    edge_rule: str,
    edge_threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Remove a fraction of the **present** edges.

    Parameters
    ----------
    snapshot : numpy.ndarray
        Square symmetric matrix of similarities or distances.
    fraction : float
        :math:`p^-`, a fraction of the present edges.
    edge_rule, edge_threshold : see :func:`add_spurious_edges`
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    numpy.ndarray
        The snapshot with ``round(fraction * n_present)`` edges removed, symmetric.

    Notes
    -----
    "Removed" is expressed in the dataset's own convention: a similarity drops to zero,
    a distance rises above the radius.  Writing zero into a *distance* matrix would create
    the edge instead of removing it.
    """
    present, absent, upper = _edge_partition(snapshot, edge_rule, edge_threshold)
    _check_level("fraction", fraction, maximum=1.0)

    n_dropped = int(round(fraction * present.sum()))
    chosen = _choose(present, n_dropped, rng)
    if edge_rule == "weight_above":
        values = np.zeros(n_dropped)
    else:
        # Push the distance clear of the radius. The spacing keeps the removed pairs from
        # sitting exactly on the threshold, where a non-strict comparison would keep them.
        span = float(np.max(snapshot)) if snapshot.size else edge_threshold
        values = np.full(n_dropped, max(span, edge_threshold) * 2.0 + 1.0)
    return _rebuild(snapshot, upper, chosen, values)


def edge_count(snapshot: np.ndarray, *, edge_rule: str, edge_threshold: float) -> int:
    """Number of edges, counted on the upper triangle (``docs/method.md`` §5)."""
    present, _, _ = _edge_partition(snapshot, edge_rule, edge_threshold)
    return int(present.sum())


def _edge_partition(
    snapshot: np.ndarray, edge_rule: str, edge_threshold: float
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Boolean masks of present and absent pairs over the upper triangle."""
    _check_square("snapshot", snapshot)
    if edge_rule not in EDGE_RULES:
        raise ValueError(f"edge_rule must be one of {EDGE_RULES}, got {edge_rule!r}")
    if not np.isfinite(edge_threshold):
        raise ValueError(f"edge_threshold must be finite, got {edge_threshold!r}")

    upper = np.triu_indices(snapshot.shape[0], 1)
    values = snapshot[upper]
    present = values > edge_threshold if edge_rule == "weight_above" else values <= edge_threshold
    return present, ~present, upper


def _choose(mask: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``count`` positions among those the mask allows, without replacement."""
    candidates = np.flatnonzero(mask)
    if count > len(candidates):
        raise ValueError(f"asked for {count} of {len(candidates)} available pairs")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(candidates, size=count, replace=False)


def _rebuild(
    snapshot: np.ndarray,
    upper: tuple[np.ndarray, np.ndarray],
    chosen: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Write the new values into a symmetric copy of the snapshot."""
    flat = snapshot[upper].copy()
    if len(chosen):
        flat[chosen] = values
    result = np.zeros_like(snapshot)
    result[upper] = flat
    result = result + result.T
    return result


def _check_square(name: str, matrix: np.ndarray) -> None:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(matrix).__name__}")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {matrix.shape}")
    _check_finite(name, matrix)


def _check_finite(name: str, array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} holds non-finite values")


def _check_level(name: str, value: float, *, maximum: float | None = None) -> None:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must not exceed {maximum}, got {value}")
