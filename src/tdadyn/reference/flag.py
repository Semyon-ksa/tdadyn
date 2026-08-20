"""Layer 2: the flag complex of a graph.

The truth of family 2 is stated for the flag (clique) complex of the thresholded
graph, because that is what the pipeline uses (``docs/method.md`` §2.3).  Every
clique of the graph is declared a simplex; nothing beyond the graph is introduced.

**Thresholding happens elsewhere, and that is the point.**  This module takes a graph
that already exists — a 0/1 adjacency matrix — and never decides for itself which
weights count as edges.  It used to: it compared ``adjacency > threshold``, which is a
second implementation of ``weight_above`` wired in the similarity direction.  On a
metric dataset, where an edge is :math:`d_{ij} \\le \\varepsilon`, that spelling builds
the flag complex of the **complement** graph — connected, plausible, and wrong without
a single error.  The one implementation of the rule is
:func:`tdadyn.core.contracts.adjacency_from_snapshot`; the caller applies it first:

.. code-block:: python

    adjacency = adjacency_from_snapshot(snapshot, edge_rule=gt.edge_rule,
                                        edge_threshold=gt.edge_threshold)
    complex = flag_complex(adjacency, max_homology_dim=2, max_nodes=30)

A non-binary matrix is refused rather than thresholded at some default, because the
only way to get one here is to have skipped that step.

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
    max_homology_dim: int,
    max_nodes: int,
) -> SimplicialComplex:
    """Build the flag complex of a graph.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square symmetric **0/1** adjacency matrix of shape ``(n_nodes, n_nodes)``, as
        returned by :func:`tdadyn.core.contracts.adjacency_from_snapshot`; the diagonal
        is ignored.  Weights are not accepted: which weights are edges is decided by the
        dataset's ``edge_rule``, one layer down, and never here.
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
    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")

    edges = np.asarray(adjacency) != 0.0
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
    max_homology_dim: int,
    max_nodes: int,
    coeff_field: int = REFERENCE_COEFF_FIELD,
) -> dict[int, int]:
    """Betti numbers of the flag complex of a graph.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Square symmetric 0/1 adjacency matrix; see :func:`flag_complex`.
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
        # NaN compares False against anything, which would drop edges without a word.
        raise ValueError("adjacency holds non-finite values")
    off_diagonal = adjacency[~np.eye(adjacency.shape[0], dtype=bool)]
    if not np.isin(off_diagonal, (0.0, 1.0)).all():
        raise ValueError(
            "adjacency must be a 0/1 matrix, and this one is not — which means the "
            "thresholding step was skipped. Apply the dataset's own rule first: "
            "core.contracts.adjacency_from_snapshot(snapshot, edge_rule=..., "
            "edge_threshold=...). Thresholding here would be a second implementation "
            "of the edge rule and, on a metric dataset, would build the complement graph."
        )
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
