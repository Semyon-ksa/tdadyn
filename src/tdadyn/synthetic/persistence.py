"""Path A: persistent homology through ripser — the pipeline being validated.

This is the harness ``docs/roadmap.md`` S3 asks for: point cloud → distance matrix →
Vietoris–Rips filtration → count of significant bars.  It is what the generators of
family 1 are checked with, and deliberately separate from
:mod:`tdadyn.reference`, which must never import a persistence library.

Notes
-----
``ripser`` casts the distance matrix to float32, so bar endpoints come back rounded:
for the distance ``1.4756009149910707`` it reports a birth of ``1.4756009578704834``,
which is exactly ``float32`` of it.  That precision is irrelevant to bar lengths, but
it decides comparisons at a scale that coincides with a pairwise distance — hence
:func:`cross_check_scales` picks midpoints between distances rather than the distances
themselves, as ``docs/validation.md`` §1.1 requires.
"""

from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix of a point cloud.

    Parameters
    ----------
    points : numpy.ndarray
        Shape ``(n_points, dim)``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_points, n_points)`` matrix with a zero diagonal.
    """
    if not isinstance(points, np.ndarray):
        raise TypeError(f"points must be a numpy array, got {type(points).__name__}")
    if points.ndim != 2:
        raise ValueError(f"points must have shape (n_points, dim), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("points hold non-finite values")
    return squareform(pdist(points))


def persistence_diagrams(
    distances: np.ndarray, *, max_homology_dim: int, coeff_field: int
) -> dict[int, np.ndarray]:
    """Persistence diagrams of the Vietoris–Rips filtration.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    max_homology_dim : int
        Highest homology dimension to compute.
    coeff_field : int
        Prime field of coefficients, passed straight to ripser.  It must reach the
        computation: a ``coeff_field`` that is read from the config and then dropped is
        the bug the Klein bottle test of ``docs/validation.md`` §3 is designed to catch.

    Returns
    -------
    dict of int to numpy.ndarray
        Dimension to an array of ``(birth, death)`` pairs; deaths may be ``inf``.

    Notes
    -----
    The filtration is never truncated (``thresh=inf``).  A cut would turn every class
    still alive at the cut into an infinite bar, and infinite bars always count as
    significant — so truncating would quietly inflate the Betti numbers.
    """
    _check_distances(distances)
    if not isinstance(max_homology_dim, (int, np.integer)) or isinstance(max_homology_dim, bool):
        raise TypeError(f"max_homology_dim must be an int, got {type(max_homology_dim).__name__}")
    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")
    if not isinstance(coeff_field, (int, np.integer)) or coeff_field < 2:
        raise ValueError(f"coeff_field must be an int >= 2, got {coeff_field!r}")

    diagrams = ripser(
        distances,
        distance_matrix=True,
        maxdim=int(max_homology_dim),
        coeff=int(coeff_field),
        thresh=np.inf,
    )["dgms"]
    return {dim: np.asarray(diagram, dtype=np.float64) for dim, diagram in enumerate(diagrams)}


def significant_betti(
    distances: np.ndarray,
    *,
    max_homology_dim: int,
    coeff_field: int,
    criterion: str,
    significance_ratio: float | None = None,
    gap_noise_floor_ratio: float | None = None,
    gap_min_ratio: float | None = None,
) -> dict[int, int]:
    """Count the significant bars per dimension — the persistent Betti numbers.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    max_homology_dim : int
        Highest homology dimension to report.
    coeff_field : int
        Prime field of coefficients.
    criterion : {'gap', 'diam_ratio'}
        Which rule to apply, from ``topology_check.criterion``.
    significance_ratio : float, optional
        Required by ``diam_ratio``.
    gap_noise_floor_ratio, gap_min_ratio : float, optional
        Required by ``gap``.

    Returns
    -------
    dict of int to int
        Number of significant bars for every dimension from 0 to ``max_homology_dim``.

    Notes
    -----
    There are deliberately no defaults for the parameters: a threshold that can be
    supplied per call is a threshold that will eventually be tuned per dataset, which
    ``docs/validation.md`` §2 forbids.

    ``diam_ratio`` is the old rule — persistence at least a fixed fraction of
    :math:`\\mathrm{diam}(X)` — kept only so that the two can be compared.  It is known
    to be unfit (``docs/method.md`` §1.2a): it loses the cavity of the torus and invents
    loops on the sphere, and the two ranges overlap in units of the diameter.

    ``gap`` reads the spectrum of persistences instead.  In every dimension the bars are
    sorted descending, dust below ``gap_noise_floor_ratio`` of the diameter is set aside,
    and the cut is made at the largest multiplicative step — **including** the step from
    the last surviving bar down into the dust.  That last step is what the calibration
    showed to be essential: two disjoint circles leave two nearly equal bars with no gap
    between them, and a rule that only looks at gaps *between* survivors reports no loops
    at all.  An infinite bar is always significant, which keeps the one infinite bar of
    :math:`H_0` in the count of connected components.
    """
    diagrams = persistence_diagrams(
        distances, max_homology_dim=max_homology_dim, coeff_field=coeff_field
    )
    diameter = float(distances.max())

    if criterion == "diam_ratio":
        if significance_ratio is None or not 0.0 < significance_ratio < 1.0:
            raise ValueError(
                f"criterion 'diam_ratio' needs significance_ratio in (0, 1), "
                f"got {significance_ratio!r}"
            )
        counter = lambda bars: _count_by_diameter(bars, significance_ratio * diameter)
    elif criterion == "gap":
        if gap_noise_floor_ratio is None or gap_min_ratio is None:
            raise ValueError(
                "criterion 'gap' needs gap_noise_floor_ratio and gap_min_ratio"
            )
        counter = lambda bars: _count_by_gap(
            bars, diameter, gap_noise_floor_ratio, gap_min_ratio
        )
    else:
        raise ValueError(f"unknown criterion {criterion!r}; expected 'gap' or 'diam_ratio'")

    return {
        dim: counter(diagram) for dim, diagram in diagrams.items() if dim <= max_homology_dim
    }


def betti_at_scale(diagrams: dict[int, np.ndarray], scale: float) -> dict[int, int]:
    """Betti numbers of the complex at one scale, read off the barcode.

    Parameters
    ----------
    diagrams : dict of int to numpy.ndarray
        Output of :func:`persistence_diagrams`.
    scale : float
        The scale :math:`\\varepsilon`.

    Returns
    -------
    dict of int to int
        :math:`\\beta_k(\\varepsilon) = \\#\\{(b, d) : b \\le \\varepsilon < d\\}`.

    Notes
    -----
    The interval is half-open, as ``docs/validation.md`` §1.1 fixes it: at
    :math:`\\varepsilon = d` the class is already dead.  This function exists only to
    put paths A and B into comparable form and takes no part in counting significant
    bars — the two questions are different and must not be mixed.
    """
    if not np.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale!r}")
    return {
        dim: int(((diagram[:, 0] <= scale) & (diagram[:, 1] > scale)).sum()) if diagram.size else 0
        for dim, diagram in diagrams.items()
    }


def cross_check_scales(distances: np.ndarray, n_scales: int) -> np.ndarray:
    """Scales for comparing paths A and B, spread over the whole filtration.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    n_scales : int
        How many scales to return, from ``topology_check.cross_check_n_scales``.

    Returns
    -------
    numpy.ndarray
        Increasing scales, each the midpoint between two adjacent distinct pairwise
        distances.

    Notes
    -----
    Midpoints rather than the distances themselves: at a distance the simplex enters the
    complex, and the comparison would come down to floating-point equality between a
    float64 threshold and ripser's float32 barcode.  The midpoints are taken evenly by
    index over the sorted list, so the range from an almost discrete complex to the
    complete graph is covered.  The choice must not depend on the Betti numbers that come
    out — picking scales until the table appears would be fitting the truth.
    """
    _check_distances(distances)
    if not isinstance(n_scales, (int, np.integer)) or isinstance(n_scales, bool):
        raise TypeError(f"n_scales must be an int, got {type(n_scales).__name__}")
    if n_scales < 1:
        raise ValueError(f"n_scales must be >= 1, got {n_scales}")

    n_points = distances.shape[0]
    distinct = np.unique(distances[np.triu_indices(n_points, k=1)])
    if distinct.size < 2:
        raise ValueError("need at least two distinct pairwise distances to form a midpoint")
    midpoints = (distinct[:-1] + distinct[1:]) / 2
    if midpoints.size < n_scales:
        raise ValueError(
            f"asked for {n_scales} scales but only {midpoints.size} midpoints exist; "
            "use a larger subsample"
        )
    positions = np.linspace(0, midpoints.size - 1, n_scales).round().astype(int)
    return midpoints[positions]


def _split_bars(diagram: np.ndarray) -> tuple[np.ndarray, int]:
    """Finite persistences sorted descending, plus the count of infinite bars."""
    if diagram.size == 0:
        return np.empty(0), 0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    infinite = ~np.isfinite(lifetimes)
    return np.sort(lifetimes[~infinite])[::-1], int(infinite.sum())


def _count_by_diameter(diagram: np.ndarray, threshold: float) -> int:
    finite, n_infinite = _split_bars(diagram)
    return n_infinite + int((finite >= threshold).sum())


def _count_by_gap(
    diagram: np.ndarray,
    diameter: float,
    gap_noise_floor_ratio: float,
    gap_min_ratio: float,
) -> int:
    """Cut the spectrum at its largest multiplicative step, dust included."""
    finite, n_infinite = _split_bars(diagram)
    floor = gap_noise_floor_ratio * diameter
    above = finite[finite >= floor]
    if above.size == 0:
        return n_infinite

    below = finite[finite < floor]
    # The step down into the dust is part of the search: a spectrum of k features
    # followed by dust has its real gap at the very end, not between the features.
    # With no dust at all the floor itself stands in, so that the final step stays a
    # finite ratio and is judged by the same threshold as every other one — a zero
    # sentinel would make it infinite and sweep the whole spectrum in as features.
    sentinel = float(below[0]) if below.size else floor
    extended = np.concatenate([above, [sentinel]])
    positive = extended[1:] > 0
    ratios = np.where(positive, extended[:-1] / np.where(positive, extended[1:], 1.0), np.inf)

    best = int(np.argmax(ratios))
    if ratios[best] < gap_min_ratio:
        return n_infinite
    return n_infinite + best + 1


def _check_distances(distances: np.ndarray) -> None:
    if not isinstance(distances, np.ndarray):
        raise TypeError(f"distances must be a numpy array, got {type(distances).__name__}")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError(f"distances must be a square matrix, got shape {distances.shape}")
    if not np.isfinite(distances).all():
        raise ValueError("distances hold non-finite values")
    if not np.array_equal(distances, distances.T):
        raise ValueError("distances must be symmetric")
    if (distances < 0).any():
        raise ValueError("distances must be non-negative")
    if np.any(np.diagonal(distances) != 0):
        raise ValueError("distances must have a zero diagonal")
