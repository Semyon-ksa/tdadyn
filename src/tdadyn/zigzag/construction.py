"""Snapshots into the diagram of complexes the zigzag runs over.

The construction is
:math:`K_0 \\hookrightarrow K_0 \\cup K_1 \\hookleftarrow K_1 \\hookrightarrow \\cdots`
(``docs/zigzag.md`` §2.1), indexed by :mod:`tdadyn.zigzag.indices`.

Two decisions live here and neither is a config key, because the invariant depends on both
(the S6 boundary: the config holds what can change without rewriting the argument).

**The edge rule is read from the dataset and applied by one implementation.**  Thresholding
goes through :func:`tdadyn.core.contracts.adjacency_from_snapshot` and nowhere else.  The
two rules point in opposite directions, and applying the similarity rule to a metric
dataset builds the flag complex of the *complement* graph — connected, plausible, and wrong
without a single error (``docs/method.md`` §2.3).

**The intermediate object is the union of the complexes, not the flag complex of the union
of the graphs** (:data:`UNION_RULE`, ``docs/zigzag.md`` §2.2).  Always
:math:`\\mathrm{Fl}(G_i) \\cup \\mathrm{Fl}(G_j) \\subseteq \\mathrm{Fl}(G_i \\cup G_j)`,
in general strictly: three edges of a triangle lying in different snapshots make a triangle
on the right and none on the left.  Such a triangle would kill a cycle at an intermediate
index and put into the barcode a birth-death pair matching no state of the data — and the
barcode would be self-consistent, so no test of it would object.
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.contracts import DynamicGraphDataset, adjacency_from_snapshot
from tdadyn.reference import SimplicialComplex, flag_complex

from .indices import CoarseIndex, index_set

UNION_RULE = "complex_union"
"""How :math:`K_{i+1/2}` is built: simplexwise union of :math:`K_i` and :math:`K_{i+1}`.

A module constant rather than a config key.  The alternative construction exists only
inside the test that measures the difference between the two; a key would invite switching
it and obtaining a barcode that is self-consistent and answers a different question.
Barcodes obtained under different conventions are not comparable, so the convention travels
with any barcode that gets published (``docs/zigzag.md`` §7).
"""


def snapshot_complex(
    snapshot: np.ndarray,
    *,
    edge_rule: str,
    edge_threshold: float,
    max_homology_dim: int,
    max_nodes: int,
) -> SimplicialComplex:
    """Build :math:`K_i`: the flag complex of one snapshot.

    Parameters
    ----------
    snapshot : numpy.ndarray
        One square symmetric matrix of similarities or of distances.
    edge_rule, edge_threshold : str, float
        Taken from the dataset's own ground truth; see
        :func:`tdadyn.core.contracts.adjacency_from_snapshot`.  They are never inferred
        from the family: ``cycle_lifecycle`` and ``geometric_loop`` share a family and
        disagree here.
    max_homology_dim : int
        Simplices are built up to dimension ``max_homology_dim + 1``.  One above the
        requested homology dimension is obligatory: without :math:`\\partial_{k+1}` the
        :math:`k`-cycles are systematically miscounted.
    max_nodes : int
        Node limit of the reference builder.

    Returns
    -------
    SimplicialComplex
        The complex of that snapshot.
    """
    adjacency = adjacency_from_snapshot(
        snapshot, edge_rule=edge_rule, edge_threshold=edge_threshold
    )
    return flag_complex(adjacency, max_homology_dim=max_homology_dim, max_nodes=max_nodes)


def union_complex(left: SimplicialComplex, right: SimplicialComplex) -> SimplicialComplex:
    """Build :math:`K_{i+1/2} = K_i \\cup K_{i+1}`, simplex by simplex.

    Parameters
    ----------
    left, right : SimplicialComplex
        The complexes of two consecutive snapshots.

    Returns
    -------
    SimplicialComplex
        Their union.  It is closed under faces because both arguments are, and the union
        of two face-closed collections is face-closed.
    """
    simplices = {simplex for source in (left, right) for dim in source.faces
                 for simplex in source.faces[dim]}
    return SimplicialComplex.from_simplices(simplices)


def zigzag_complexes(
    dataset: DynamicGraphDataset, *, max_homology_dim: int, max_nodes: int
) -> dict[CoarseIndex, SimplicialComplex]:
    """Build the whole diagram: one complex per coarse index.

    Parameters
    ----------
    dataset : DynamicGraphDataset
        A dynamic graph whose ground truth records ``edge_rule`` and ``edge_threshold``.
    max_homology_dim, max_nodes : int
        Passed through to :func:`snapshot_complex`.

    Returns
    -------
    dict of CoarseIndex to SimplicialComplex
        The :math:`2T - 1` complexes in index order.

    Raises
    ------
    ValueError
        If the ground truth records no edge rule.  Guessing one would read a metric
        dataset as the complement of its graph, silently.

    Notes
    -----
    Identical snapshots are **not** collapsed.  ``cycle_lifecycle`` repeats a state on
    purpose (``docs/method.md`` §2.1), and deduplicating by graph identity would erase the
    birth and death of :math:`H_0` that the repetition exists to exercise.
    """
    truth = dataset.ground_truth
    if truth.edge_rule is None or truth.edge_threshold is None:
        raise ValueError(
            f"dataset {truth.generator!r} records no edge_rule/edge_threshold, and the "
            "rule is never guessed: applying the similarity rule to a metric dataset "
            "builds the complement of the intended graph and does it quietly "
            "(docs/method.md 2.3)"
        )

    snapshots = [
        snapshot_complex(
            snapshot,
            edge_rule=truth.edge_rule,
            edge_threshold=truth.edge_threshold,
            max_homology_dim=max_homology_dim,
            max_nodes=max_nodes,
        )
        for snapshot in dataset.snapshots
    ]

    diagram: dict[CoarseIndex, SimplicialComplex] = {}
    for index in index_set(len(snapshots)):
        if index.is_snapshot:
            diagram[index] = snapshots[index.snapshot]
        else:
            left = index.left_snapshot
            diagram[index] = union_complex(snapshots[left], snapshots[left + 1])
    return diagram
