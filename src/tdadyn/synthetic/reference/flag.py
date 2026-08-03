"""Layer 2: the flag complex of a thresholded graph.

The truth of family 2 is stated for the flag (clique) complex of the thresholded
graph, because that is what the pipeline uses (``docs/method.md`` §2.3).  Every
clique of the graph is declared a simplex; nothing beyond the graph is introduced.

Two consequences drive this module:

* **Simplices are built up to dimension** :math:`k_{\\max}+1`.  Computing
  :math:`\\beta_2` needs 3-simplices, and truncating at dimension 2 would count every
  closed 2-chain as a cavity — the complete graph on four nodes would report
  :math:`\\beta_2 = 1` instead of 0.  There is deliberately **no** parameter for
  truncating: its only use would be to switch on the wrong answer.
* **All cliques of every size are enumerated, not only the maximal ones.**
  ``networkx.find_cliques`` returns maximal cliques and is unusable here: on the
  complete graph of four nodes it yields one tetrahedron with no edges and no
  triangles, the boundary matrices fail to line up, and :math:`\\beta_1` grows from
  0 to 3.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .homology import REFERENCE_COEFF_FIELD, SimplicialComplex, betti_numbers


def flag_complex(
    adjacency: np.ndarray,
    *,
    threshold: float,
    max_homology_dim: int,
    max_nodes: int,
) -> SimplicialComplex:
    """Build the flag complex of a weighted graph thresholded at ``threshold``.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square symmetric weight matrix of shape ``(n_nodes, n_nodes)``; the diagonal
        is ignored.
    threshold : float
        An edge is present where the weight is **strictly above** the threshold.  A
        weight equal to the threshold does not exceed it, and at ``threshold = 0`` the
        strict comparison keeps zero weights — that is, absent edges — out of the graph.
    max_homology_dim : int
        Highest homology dimension the complex must support.  Simplices are built up
        to dimension ``max_homology_dim + 1``.
    max_nodes : int
        Node limit, from ``topology_check.reference_max_nodes``.  This implementation
        is deliberately slow and is meant for small graphs only.

    Returns
    -------
    SimplicialComplex
        The flag complex, closed under faces by construction: every subset of a clique
        is itself a clique.
    """
    _check_adjacency(adjacency, max_nodes=max_nodes)
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold!r}")
    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")

    edges = np.asarray(adjacency) > threshold
    np.fill_diagonal(edges, False)

    n_nodes = edges.shape[0]
    max_simplex_size = max_homology_dim + 2  # dimension max_homology_dim + 1
    simplices: list[tuple[int, ...]] = [(node,) for node in range(n_nodes)]
    for size in range(2, max_simplex_size + 1):
        simplices.extend(
            candidate
            for candidate in combinations(range(n_nodes), size)
            if _is_clique(candidate, edges)
        )
    return SimplicialComplex.from_simplices(simplices)


def betti_numbers_of_graph(
    adjacency: np.ndarray,
    *,
    threshold: float,
    max_homology_dim: int,
    max_nodes: int,
    coeff_field: int = REFERENCE_COEFF_FIELD,
) -> dict[int, int]:
    """Betti numbers of the flag complex of a thresholded graph.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square symmetric weight matrix.
    threshold : float
        Edges are the entries strictly above this value.
    max_homology_dim : int
        Highest homology dimension to report.
    max_nodes : int
        Node limit, from ``topology_check.reference_max_nodes``.
    coeff_field : int, default 2
        Field of coefficients; only 2 is implemented.

    Returns
    -------
    dict of int to int
        :math:`\\beta_k` for every ``k`` from 0 to ``max_homology_dim``.

    Notes
    -----
    Isolated nodes are cliques of size one and therefore contribute their own
    connected components to :math:`\\beta_0` — relevant when a node temporarily loses
    all of its edges in a dynamic scenario.
    """
    complex = flag_complex(
        adjacency,
        threshold=threshold,
        max_homology_dim=max_homology_dim,
        max_nodes=max_nodes,
    )
    return betti_numbers(complex, max_homology_dim, coeff_field=coeff_field)


def _is_clique(candidate: tuple[int, ...], edges: np.ndarray) -> bool:
    """Check that every pair of the candidate vertices is adjacent."""
    return all(edges[left, right] for left, right in combinations(candidate, 2))


def _check_adjacency(adjacency: np.ndarray, *, max_nodes: int) -> None:
    """Validate the weight matrix a flag complex is built from."""
    if not isinstance(adjacency, np.ndarray):
        raise TypeError(f"adjacency must be a numpy array, got {type(adjacency).__name__}")
    if adjacency.ndim != 2:
        raise ValueError(f"adjacency must be 2-D, got shape {adjacency.shape}")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {adjacency.shape}")
    if not np.isfinite(adjacency).all():
        # NaN > threshold is quietly False, which would drop edges without a word.
        raise ValueError("adjacency holds non-finite values")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    if not isinstance(max_nodes, (int, np.integer)) or isinstance(max_nodes, bool):
        raise TypeError(f"max_nodes must be an int, got {type(max_nodes).__name__}")
    if adjacency.shape[0] > max_nodes:
        raise ValueError(
            f"the graph has {adjacency.shape[0]} nodes, above the reference limit of "
            f"{max_nodes} (topology_check.reference_max_nodes). The reference "
            "implementation is deliberately slow; keep the scenarios small."
        )
