"""The reference implementation against complexes whose homology is known by hand.

Expectations come from the tables in ``docs/method.md`` §2.3 and ``docs/validation.md``
§4, never from the output of this code.  The checks are split by layer, as the docs
require: layer 1 takes explicit simplices, layer 2 takes a graph.

Graph constructors are written out with numpy rather than taken from networkx: the
data a trust anchor is tested on should not come from a library.  Following
``docs/method.md`` §0, the letter K is not used in identifiers — it means three different
things in this project — so the complete graph is ``complete_graph``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from tdadyn import reference
from tdadyn.reference import (
    SimplicialComplex,
    betti_numbers,
    betti_numbers_of_graph,
    boundary_matrix,
    flag_complex,
    rank_mod2,
)

EDGE = 1.0
MAX_NODES = 30
MAX_HOMOLOGY_DIM = 2

FLAG_ARGS = {
    "max_homology_dim": MAX_HOMOLOGY_DIM,
    "max_nodes": MAX_NODES,
}


# --- graph constructors ----------------------------------------------------------


def graph_from_edges(n_nodes: int, edges) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes))
    for left, right in edges:
        adjacency[left, right] = EDGE
        adjacency[right, left] = EDGE
    return adjacency


def complete_graph(n_nodes: int) -> np.ndarray:
    return graph_from_edges(n_nodes, combinations(range(n_nodes), 2))


def complete_multipartite_graph(part_sizes) -> np.ndarray:
    """Edges between parts only; ``[2, 2, 2]`` is the octahedron."""
    part = np.concatenate([[index] * size for index, size in enumerate(part_sizes)])
    return (part[:, None] != part[None, :]) * EDGE


def cycle_graph(n_nodes: int) -> np.ndarray:
    return graph_from_edges(n_nodes, [(i, (i + 1) % n_nodes) for i in range(n_nodes)])


def path_graph(n_nodes: int) -> np.ndarray:
    return graph_from_edges(n_nodes, [(i, i + 1) for i in range(n_nodes - 1)])


# --- layer 1: explicit complexes -------------------------------------------------

POINT = [(0,)]
TWO_POINTS = [(0,), (1,)]
SEGMENT = [(0,), (1,), (0, 1)]
TRIANGLE_SKELETON = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
FILLED_TRIANGLE = [*TRIANGLE_SKELETON, (0, 1, 2)]
HOLLOW_TETRAHEDRON = [
    *[(vertex,) for vertex in range(4)],
    *combinations(range(4), 2),
    *combinations(range(4), 3),
]
TWO_TRIANGLE_SKELETONS = [
    *TRIANGLE_SKELETON,
    *[(vertex,) for vertex in (3, 4, 5)],
    (3, 4),
    (3, 5),
    (4, 5),
]


@pytest.mark.parametrize(
    ("name", "simplices", "expected"),
    [
        ("point", POINT, {0: 1, 1: 0, 2: 0}),
        ("two disjoint points", TWO_POINTS, {0: 2, 1: 0, 2: 0}),
        ("segment", SEGMENT, {0: 1, 1: 0, 2: 0}),
        ("triangle as a 1-skeleton", TRIANGLE_SKELETON, {0: 1, 1: 1, 2: 0}),
        ("filled triangle", FILLED_TRIANGLE, {0: 1, 1: 0, 2: 0}),
        ("hollow tetrahedron as an explicit 2-skeleton", HOLLOW_TETRAHEDRON, {0: 1, 1: 0, 2: 1}),
        ("two triangle skeletons", TWO_TRIANGLE_SKELETONS, {0: 2, 1: 2, 2: 0}),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_homology_engine_on_hand_computed_complexes(name, simplices, expected):
    complex = SimplicialComplex.from_simplices(simplices)
    assert betti_numbers(complex, MAX_HOMOLOGY_DIM) == expected


def test_filling_the_triangle_kills_its_loop():
    # The same six simplices, one 2-cell apart: beta_1 goes from 1 to 0.
    skeleton = betti_numbers(SimplicialComplex.from_simplices(TRIANGLE_SKELETON), 1)
    filled = betti_numbers(SimplicialComplex.from_simplices(FILLED_TRIANGLE), 1)
    assert (skeleton[1], filled[1]) == (1, 0)


def test_a_missing_face_is_an_error_not_something_to_repair():
    # Completing the complex here would mask a builder that enumerates only maximal
    # cliques — exactly the bug the face-count test below exists to catch.
    with pytest.raises(ValueError, match="not closed under faces"):
        SimplicialComplex.from_simplices([(0,), (1,), (2,), (0, 1, 2)])


def test_malformed_simplices_are_rejected():
    with pytest.raises(ValueError, match="repeats a vertex"):
        SimplicialComplex.from_simplices([(0,), (0, 0)])
    with pytest.raises(ValueError, match="listed twice"):
        SimplicialComplex.from_simplices([(0,), (1,), (0, 1), (1, 0)])
    with pytest.raises(ValueError, match="empty simplex"):
        SimplicialComplex.from_simplices([()])
    with pytest.raises(TypeError, match="vertex labels must be ints"):
        SimplicialComplex.from_simplices([("a",)])


def test_only_z2_is_implemented():
    """At p != 2 the reference must refuse, not quietly answer for Z/2.

    ``docs/validation.md`` §1: at p = 3 only path A is available, and a test there is
    a consistency check rather than an independent confirmation.  A reference that
    silently ignored the field would make the two indistinguishable.
    """
    complex = SimplicialComplex.from_simplices(TRIANGLE_SKELETON)
    with pytest.raises(NotImplementedError, match="only over Z/2"):
        betti_numbers(complex, MAX_HOMOLOGY_DIM, coeff_field=3)


@pytest.mark.parametrize(
    ("matrix", "expected_rank"),
    [
        (np.zeros((3, 3), dtype=bool), 0),
        (np.eye(4, dtype=bool), 4),
        (np.ones((3, 3), dtype=bool), 1),
        # Third row is the XOR of the first two, so the rank is 2 over Z/2 — over the
        # reals these rows would be independent.
        (np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool), 2),
        (np.zeros((0, 5), dtype=bool), 0),
    ],
)
def test_rank_over_z2_on_hand_computed_matrices(matrix, expected_rank):
    assert rank_mod2(matrix) == expected_rank


def test_rank_does_not_modify_its_argument():
    matrix = np.array([[1, 1], [1, 0]], dtype=bool)
    before = matrix.copy()
    rank_mod2(matrix)
    assert np.array_equal(matrix, before)


def test_boundary_matrix_shapes_and_entries():
    complex = SimplicialComplex.from_simplices(FILLED_TRIANGLE)
    assert boundary_matrix(complex, 0).shape == (0, 3)  # partial_0 maps into zero
    assert boundary_matrix(complex, 1).shape == (3, 3)
    assert boundary_matrix(complex, 2).shape == (3, 1)
    # The single triangle has all three edges as faces.
    assert boundary_matrix(complex, 2).sum() == 3
    # Every edge has exactly two endpoints.
    assert boundary_matrix(complex, 1).sum(axis=0).tolist() == [2, 2, 2]


# --- layer 2: flag complexes -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "adjacency", "expected"),
    [
        ("complete graph on 3 nodes", complete_graph(3), {0: 1, 1: 0, 2: 0}),
        ("complete graph on 4 nodes", complete_graph(4), {0: 1, 1: 0, 2: 0}),
        ("octahedron", complete_multipartite_graph([2, 2, 2]), {0: 1, 1: 0, 2: 1}),
        ("4-cycle", cycle_graph(4), {0: 1, 1: 1, 2: 0}),
        ("5-cycle", cycle_graph(5), {0: 1, 1: 1, 2: 0}),
        ("path", path_graph(5), {0: 1, 1: 0, 2: 0}),
        ("two disjoint edges", graph_from_edges(4, [(0, 1), (2, 3)]), {0: 2, 1: 0, 2: 0}),
        ("three isolated nodes", graph_from_edges(3, []), {0: 3, 1: 0, 2: 0}),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_flag_complex_betti_numbers(name, adjacency, expected):
    assert betti_numbers_of_graph(adjacency, **FLAG_ARGS) == expected


def test_beta_1_of_a_flag_complex_is_not_the_cyclomatic_number():
    """The triangle has cyclomatic number 1 and beta_1 = 0.

    ``docs/method.md`` §2.3: the triangle gets filled by the flag construction, which
    kills the loop.  Computing beta_1 as m - n + c is the classic silent error.
    """
    adjacency = complete_graph(3)
    n_edges, n_nodes, n_components = 3, 3, 1
    assert n_edges - n_nodes + n_components == 1
    assert betti_numbers_of_graph(adjacency, **FLAG_ARGS)[1] == 0


def test_truncating_at_dimension_two_would_report_a_cavity_that_is_not_there():
    """Why the complex is built up to dimension max_homology_dim + 1.

    The flag complex of the complete graph on four nodes is a solid tetrahedron. Drop
    its single 3-simplex and the same code reports beta_2 = 1 — the wrong answer a
    truncating implementation would give, and the reason no truncation parameter
    exists in the API.
    """
    complete = flag_complex(complete_graph(4), **FLAG_ARGS)
    assert betti_numbers(complete, MAX_HOMOLOGY_DIM)[2] == 0

    truncated = SimplicialComplex.from_simplices(
        simplex for dim in range(3) for simplex in complete.simplices(dim)
    )
    assert betti_numbers(truncated, MAX_HOMOLOGY_DIM)[2] == 1


def test_face_counts_of_the_complete_graph_on_four_nodes():
    """Catches enumeration of maximal cliques only, which would give (4, 0, 0, 1)."""
    assert flag_complex(complete_graph(4), **FLAG_ARGS).face_counts() == (4, 6, 4, 1)


def test_the_octahedron_has_no_tetrahedra():
    # Its maximal cliques are triangles, so the cavity is genuine rather than an
    # artefact of stopping at dimension 2.
    complex = flag_complex(complete_multipartite_graph([2, 2, 2]), **FLAG_ARGS)
    assert complex.face_counts() == (6, 12, 8)


@pytest.mark.parametrize(
    "adjacency",
    [complete_graph(4), complete_multipartite_graph([2, 2, 2]), cycle_graph(5)],
    ids=["complete-4", "octahedron", "cycle-5"],
)
def test_every_subset_of_a_simplex_is_in_the_complex(adjacency):
    complex = flag_complex(adjacency, **FLAG_ARGS)
    present = {simplex for dim in range(complex.dimension + 1) for simplex in complex.simplices(dim)}
    for simplex in present:
        for size in range(1, len(simplex)):
            for subset in combinations(simplex, size):
                assert subset in present


def test_simplices_are_built_one_dimension_above_the_homology_dimension():
    complex = flag_complex(complete_graph(4), max_homology_dim=1, max_nodes=MAX_NODES)
    assert complex.dimension == 2  # triangles needed for beta_1, no tetrahedra


def test_a_weight_matrix_is_refused_instead_of_thresholded():
    """The edge rule lives in ``core.contracts`` and nowhere else.

    This module used to threshold its input at ``adjacency > threshold``, a second
    implementation of ``weight_above`` wired in the similarity direction.  Fed a metric
    dataset it built the flag complex of the *complement* graph — connected, plausible
    and wrong in silence.  The parameter is gone, so the only way a weight matrix can
    arrive here is that the caller skipped the thresholding step, and the refusal has to
    say so.  Whether the comparison itself is strict or not is checked where it now
    lives: ``test_dynamic_events.test_weight_above_is_strict_and_distance_within_is_not``.
    """
    weights = graph_from_edges(3, [(0, 1)]) * 0.7
    with pytest.raises(ValueError, match="0/1 matrix"):
        flag_complex(weights, **FLAG_ARGS)
    with pytest.raises(ValueError, match="adjacency_from_snapshot"):
        betti_numbers_of_graph(weights, **FLAG_ARGS)


def test_node_limit_is_enforced():
    with pytest.raises(ValueError, match="reference_max_nodes"):
        flag_complex(complete_graph(6), max_homology_dim=2, max_nodes=5)


@pytest.mark.parametrize(
    ("adjacency", "message"),
    [
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "symmetric"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "non-finite"),
        (np.zeros((2, 3)), "square"),
        (np.zeros(3), "2-D"),
    ],
)
def test_malformed_adjacency_is_rejected(adjacency, message):
    with pytest.raises(ValueError, match=message):
        flag_complex(adjacency, **FLAG_ARGS)


def test_the_reference_does_not_import_a_persistence_library():
    """Path B must share no code with path A, or it confirms nothing."""
    package_dir = Path(reference.__file__).parent
    modules = sorted(package_dir.glob("*.py"))
    assert modules, "no reference modules found"
    for path in modules:
        source = path.read_text(encoding="utf-8")
        for library in ("ripser", "gudhi", "persim"):
            assert f"import {library}" not in source, f"{path.name} imports {library}"
            assert f"from {library}" not in source, f"{path.name} imports from {library}"
