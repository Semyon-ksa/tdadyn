"""Point cloud to weighted network and back — the bridge to task 3.

A Gaussian kernel turns a cloud into a fully connected weighted network, which is the
same kind of object as a functional connectome.  The k-NN route instead approximates the
**intrinsic** metric of the manifold through shortest paths (the Isomap construction).

Three things here decide the answer quietly, and all three are settled in
``docs/method.md`` §1.3:

* **The kernel width must match the scale of the data.**  At an absolute width of 0.5 and
  a torus diameter of 5.71, 60% of the weights fall below 1e-6 and the geometry collapses
  before any inverse transform is applied.  ``sqrt_neg_log`` does not notice, being
  exactly the inverse of the kernel; ``1 - w`` has nothing to undo it with and works on
  data already destroyed.  The width is therefore a fraction of the *median* pairwise
  distance, and :func:`collapsed_weight_fraction` is the guard.
* **k-NN is not a complexity device.**  Shortest paths give a dense matrix, so the Rips
  complex over it is exactly as large as over the Euclidean distances.  Its value is the
  intrinsic metric.
* **k works in a window.**  Below it the graph falls apart; above it short-circuiting sets
  in — chords cutting across the holes — and the intrinsic metric degenerates into the
  ambient one.  :func:`geodesic_diameter_ratio` is the cheap guard for the upper end.

Monotone inverse transforms preserve the *structure* of the diagram but distort
persistences, so the significance criterion is not invariant under them.  Neither
transform is forbidden: ``1 - w`` is the standard choice for connectomes.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

COLLAPSE_FLOOR = 1e-6
"""Weights below this are numerically indistinguishable from zero; the fraction of them
is what tells a healthy kernel from a degenerate one."""


def kernel_width(distances: np.ndarray, *, median_fraction: float) -> float:
    """Kernel width as a fraction of the median pairwise distance.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    median_fraction : float
        From ``network_from_cloud.kernel_sigma_median_fraction``.

    Returns
    -------
    float
        The width :math:`\\sigma`.

    Notes
    -----
    The median rather than the diameter: the diameter is set by the single farthest pair
    and, on an elongated object, says little about the typical scale the kernel has to
    resolve.
    """
    if median_fraction <= 0:
        raise ValueError(f"median_fraction must be > 0, got {median_fraction}")
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    if upper.size == 0:
        raise ValueError("need at least two points to measure a median distance")
    return float(median_fraction * np.median(upper))


def gaussian_kernel(distances: np.ndarray, *, sigma: float) -> np.ndarray:
    """Similarity matrix :math:`w_{ij} = \\exp(-d_{ij}^2 / 2\\sigma^2)`.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    sigma : float
        Kernel width, normally from :func:`kernel_width`.

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix with ones on the diagonal.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    # Symmetric by construction, but make it exactly so: the distance validator demands
    # bitwise symmetry and floating-point exp of a symmetric matrix is symmetric only
    # because the inputs are identical, which is worth not relying on.
    return (weights + weights.T) / 2


def collapsed_weight_fraction(weights: np.ndarray, *, floor: float = COLLAPSE_FLOOR) -> float:
    """Fraction of off-diagonal weights that have collapsed to numerical zero.

    Parameters
    ----------
    weights : numpy.ndarray
        Similarity matrix.
    floor : float, default 1e-6
        Below this a weight carries no geometry.

    Returns
    -------
    float
        The fraction, to be compared with ``max_collapsed_weight_fraction``.

    Notes
    -----
    This is the guard for the mistake that ``docs/method.md`` §1.3 calls the main source
    of error here.  It is deliberately a *number*, not an exception: the decision of what
    is acceptable belongs to the caller and to the config.
    """
    upper = weights[np.triu_indices(weights.shape[0], k=1)]
    if upper.size == 0:
        return 0.0
    return float((upper < floor).mean())


def distances_from_similarity(
    weights: np.ndarray, *, transform: str, sigma: float | None = None
) -> np.ndarray:
    """Turn similarities back into distances with a monotonically decreasing function.

    Parameters
    ----------
    weights : numpy.ndarray
        Symmetric similarity matrix.
    transform : {'sqrt_neg_log', 'one_minus'}
        ``sqrt_neg_log`` is :math:`\\sqrt{-2\\sigma^2\\log w}`, the exact inverse of the
        Gaussian kernel, and needs ``sigma``.  ``one_minus`` is :math:`1 - w` and is what
        task 3 will use on correlation matrices as :math:`d = 1 - |r|`.
    sigma : float, optional
        Required by ``sqrt_neg_log``.  Without it the recovered distances come out
        divided by sigma — the topology survives a uniform rescaling, but the numbers
        stop being comparable with the original cloud.

    Returns
    -------
    numpy.ndarray
        Symmetric distance matrix with a zero diagonal.

    Notes
    -----
    Monotonicity keeps the order in which simplices enter the filtration, and therefore
    the structure of the diagram.  It does **not** keep persistences: :math:`\\varphi(d) -
    \\varphi(b) \\ne \\varphi(d - b)`, and the significance criterion compares exactly
    those.  See ``docs/method.md`` §1.3.
    """
    if transform == "sqrt_neg_log":
        if sigma is None or sigma <= 0:
            raise ValueError("transform 'sqrt_neg_log' needs a positive sigma")
        safe = np.clip(weights, np.finfo(np.float64).tiny, 1.0)
        distances = np.sqrt(-2.0 * sigma**2 * np.log(safe))
    elif transform == "one_minus":
        distances = 1.0 - weights
    else:
        raise ValueError(
            f"unknown transform {transform!r}; expected 'sqrt_neg_log' or 'one_minus'"
        )
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    return distances


def knn_graph(distances: np.ndarray, *, k: int, symmetrize: str) -> np.ndarray:
    """Weighted k-nearest-neighbour graph with Euclidean edge weights.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric distance matrix.
    k : int
        Number of neighbours per node, excluding the node itself.
    symmetrize : {'union', 'mutual'}
        A k-NN graph is directed by nature.  ``union`` keeps an edge when either endpoint
        counts the other as a neighbour and is what the project adopts: it connects the
        graph at a smaller ``k`` and so widens the working window from below.  ``mutual``
        requires both and is kept as a negative control.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix of edge weights, zero where there is no edge.
    """
    n_nodes = distances.shape[0]
    if not 1 <= k < n_nodes:
        raise ValueError(f"k must lie in [1, {n_nodes - 1}], got {k}")

    order = np.argsort(distances, axis=1)
    directed = np.zeros_like(distances)
    rows = np.repeat(np.arange(n_nodes), k)
    columns = order[:, 1 : k + 1].ravel()
    directed[rows, columns] = distances[rows, columns]

    if symmetrize == "union":
        return np.maximum(directed, directed.T)
    if symmetrize == "mutual":
        both = (directed > 0) & (directed.T > 0)
        return np.where(both, directed, 0.0)
    raise ValueError(f"unknown symmetrize {symmetrize!r}; expected 'union' or 'mutual'")


def knn_geodesic_distances(
    distances: np.ndarray, *, k: int, symmetrize: str, require_connected: bool = True
) -> np.ndarray:
    """Shortest-path distances along the k-NN graph — the Isomap construction.

    Parameters
    ----------
    distances : numpy.ndarray
        Square symmetric matrix of ambient distances.
    k : int
        Neighbours per node.
    symmetrize : {'union', 'mutual'}
        How the directed k-NN graph is made undirected.
    require_connected : bool, default True
        Raise when the graph falls apart.  A disconnected graph leaves some geodesic
        distances infinite, and substituting a large number for them silently feeds
        rubbish to the persistence computation.

    Returns
    -------
    numpy.ndarray
        Dense symmetric matrix of geodesic distances with a zero diagonal.

    Notes
    -----
    The result is **dense**: this construction buys an intrinsic metric, not a smaller
    complex (``docs/method.md`` §1.3).

    ``scipy.sparse.csgraph.shortest_path`` does not return a bitwise symmetric matrix —
    paths are found in different orders in the two directions — so the result is
    symmetrised explicitly.  Measured: without it the distance validator rejects the
    matrix.
    """
    graph = knn_graph(distances, k=k, symmetrize=symmetrize)
    n_components, _ = connected_components(csr_matrix(graph), directed=False)
    if n_components > 1 and require_connected:
        raise ValueError(
            f"the {symmetrize} {k}-NN graph falls into {n_components} components; "
            "geodesic distances would be infinite. Raise k, or switch symmetrize to "
            "'union', rather than substituting a large number for infinity."
        )

    geodesic = shortest_path(csr_matrix(graph), directed=False)
    geodesic = np.minimum(geodesic, geodesic.T)
    np.fill_diagonal(geodesic, 0.0)
    return geodesic


def geodesic_diameter_ratio(geodesic: np.ndarray, ambient: np.ndarray) -> float:
    """Ratio of the geodesic diameter to the ambient one — the short-circuit guard.

    Parameters
    ----------
    geodesic : numpy.ndarray
        Geodesic distance matrix.
    ambient : numpy.ndarray
        Euclidean distance matrix of the same cloud.

    Returns
    -------
    float
        The ratio, to be compared with ``min_geodesic_diameter_ratio``.

    Notes
    -----
    On a curved manifold the intrinsic diameter exceeds the ambient one.  As ``k`` grows,
    chords appear that cut across the holes and the ratio slides towards 1: measured on
    the torus, from 7.72/5.71 at ``k = 8`` to 5.72/5.71 at ``k = 250``, where the graph is
    all but complete.  Approaching 1 means the metric has degenerated, and this costs no
    persistent homology to check.
    """
    ambient_diameter = float(ambient.max())
    if ambient_diameter <= 0:
        raise ValueError("the ambient diameter is zero")
    return float(geodesic.max()) / ambient_diameter


def is_monotone_decreasing(transform: str, *, sigma: float | None = None,
                           n_samples: int = 1000) -> bool:
    """Check that a similarity-to-distance transform is monotonically decreasing.

    Parameters
    ----------
    transform : {'sqrt_neg_log', 'one_minus'}
        Which transform to check.
    sigma : float, optional
        Needed by ``sqrt_neg_log``.
    n_samples : int, default 1000
        Grid resolution over the similarities.

    Returns
    -------
    bool
        True if distance strictly decreases as similarity increases.

    Notes
    -----
    Monotonicity is what preserves the order in which simplices enter the filtration.
    Its violation changes the answer without any other symptom, which is why the property
    is checked rather than assumed.
    """
    similarities = np.linspace(1e-9, 1.0, n_samples)
    values = _transform_values(similarities, transform, sigma)
    return bool(np.all(np.diff(values) < 0))


def _transform_values(similarities: np.ndarray, transform: str, sigma: float | None) -> np.ndarray:
    """Apply a transform to a 1-D array of similarities, for the monotonicity check."""
    if transform == "sqrt_neg_log":
        if sigma is None or sigma <= 0:
            raise ValueError("transform 'sqrt_neg_log' needs a positive sigma")
        return np.sqrt(-2.0 * sigma**2 * np.log(np.clip(similarities, np.finfo(np.float64).tiny, 1.0)))
    if transform == "one_minus":
        return 1.0 - similarities
    raise ValueError(f"unknown transform {transform!r}")
