"""Snapshots into the diagram of complexes, and the union rule that shapes it."""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import FORMAT_VERSION, GroundTruth, adjacency_from_snapshot
from tdadyn.reference import flag_complex
from tdadyn.zigzag import (
    UNION_RULE,
    index_set,
    snapshot_complex,
    union_complex,
    zigzag_complexes,
)


def simplices_of(complex_):
    """Every simplex of a complex as one flat set, for set arithmetic in the tests."""
    return {simplex for dim in complex_.faces for simplex in complex_.faces[dim]}


def build_args(config):
    return {
        "max_homology_dim": config.zigzag.max_homology_dim,
        "max_nodes": config.topology_check.reference_max_nodes,
    }


def adjacencies(dataset):
    truth = dataset.ground_truth
    return [
        adjacency_from_snapshot(
            snapshot, edge_rule=truth.edge_rule, edge_threshold=truth.edge_threshold
        )
        for snapshot in dataset.snapshots
    ]


# --- the diagram ------------------------------------------------------------------


def test_the_diagram_has_one_complex_per_coarse_index(family_two_dataset, config):
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    n_snapshots = len(family_two_dataset.snapshots)
    assert len(diagram) == 2 * n_snapshots - 1
    assert tuple(diagram) == tuple(index_set(n_snapshots))


def test_the_snapshot_complexes_match_a_direct_build(family_two_dataset, config):
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    for index, adjacency in zip(index_set(len(family_two_dataset.snapshots))[::2],
                                adjacencies(family_two_dataset)):
        assert diagram[index] == flag_complex(adjacency, **build_args(config))


def test_the_intermediate_is_the_union_of_its_neighbours_and_nothing_more(
    family_two_dataset, config
):
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    indices = index_set(len(family_two_dataset.snapshots))
    for position in range(1, len(indices), 2):
        middle = simplices_of(diagram[indices[position]])
        left = simplices_of(diagram[indices[position - 1]])
        right = simplices_of(diagram[indices[position + 1]])
        assert middle == left | right


def test_the_complex_is_truncated_one_dimension_above_the_homology_dimension(
    family_two_dataset, config
):
    # docs/zigzag.md 1.2: without the (k+1)-simplices the k-cycles are systematically
    # miscounted, and above k+1 nothing is computed at all.
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    ceiling = config.zigzag.max_homology_dim + 1
    assert max(complex_.dimension for complex_ in diagram.values()) <= ceiling


def test_the_edge_rule_comes_from_the_truth_and_is_applied_once(cycle_lifecycle, config):
    """Applying the opposite rule gives a different diagram — so the rule is read, not fixed.

    ``snapshot_complex`` must not compare against the threshold itself: the two rules point
    in opposite directions and the wrong one builds the complement graph
    (``docs/method.md`` §2.3).  Here the check is that the rule actually reaches the
    construction, which is what a hard-wired comparison would break.
    """
    truth = cycle_lifecycle.ground_truth
    snapshot = cycle_lifecycle.snapshots[0]
    same = snapshot_complex(
        snapshot, edge_rule=truth.edge_rule, edge_threshold=truth.edge_threshold,
        **build_args(config),
    )
    other = "distance_within" if truth.edge_rule == "weight_above" else "weight_above"
    complement = snapshot_complex(
        snapshot, edge_rule=other, edge_threshold=truth.edge_threshold, **build_args(config)
    )
    assert same != complement


# --- the union rule ---------------------------------------------------------------


def test_the_union_rule_is_a_module_constant():
    # docs/zigzag.md 2.2: the invariant depends on it, so by the S6 boundary it is a
    # constant and not a config key. A key would invite switching it and getting a
    # self-consistent barcode that answers a different question.
    assert UNION_RULE == "complex_union"


def test_the_two_union_constructions_agree_on_every_transition(family_two_dataset, config):
    """The measurement of ``docs/zigzag.md`` §2.2, as a test rather than a claim.

    Always ``Fl(G_i) u Fl(G_j) <= Fl(G_i u G_j)``, in general strictly: three edges of a
    triangle lying in different snapshots make a triangle in the flag complex of the graph
    union and no triangle in the union of the complexes.  On these datasets the inclusion
    turns out to be an equality everywhere, which is recorded as a measurement — the choice
    of ``UNION_RULE`` rests on the principle (no simplex that exists in no snapshot), not on
    the difference being zero.
    """
    diagram = zigzag_complexes(family_two_dataset, **build_args(config))
    indices = index_set(len(family_two_dataset.snapshots))
    matrices = adjacencies(family_two_dataset)
    for position in range(1, len(indices), 2):
        left = position // 2
        flag_of_graph_union = flag_complex(
            np.maximum(matrices[left], matrices[left + 1]), **build_args(config)
        )
        extra = simplices_of(flag_of_graph_union) - simplices_of(diagram[indices[position]])
        assert extra == set(), f"transition {left} -> {left + 1}: {sorted(extra)[:5]}"


def test_the_agreement_above_is_not_vacuous(config):
    """Negative control for the previous test: a case where the constructions do differ.

    Without it that test would pass against an implementation that always returned an
    empty difference — including one that compared a complex with itself.
    """
    first, second = np.zeros((3, 3)), np.zeros((3, 3))
    first[0, 1] = first[1, 0] = 1.0
    second[0, 2] = second[2, 0] = 1.0
    second[1, 2] = second[2, 1] = 1.0

    args = {"max_homology_dim": 1, "max_nodes": config.topology_check.reference_max_nodes}
    complexes = [flag_complex(matrix, **args) for matrix in (first, second)]
    of_complexes = simplices_of(union_complex(*complexes))
    of_graph_union = simplices_of(flag_complex(np.maximum(first, second), **args))
    assert of_graph_union - of_complexes == {(0, 1, 2)}


# --- monotonicity, named transition by transition ---------------------------------


GAP_CROSSING_EDGES = {(0, 22), (0, 23), (1, 23)}
"""The chords across the arc gap of ``geometric_loop``, measured on the dataset."""


def edge_sets(dataset):
    n_nodes = dataset.snapshots.shape[1]
    upper = np.triu_indices(n_nodes, 1)
    return [
        {(int(a), int(b)) for a, b in zip(*upper) if matrix[a, b]}
        for matrix in adjacencies(dataset)
    ]


def test_every_cycle_lifecycle_transition_is_monotone(cycle_lifecycle):
    # Each step adds or removes a single edge, so both union constructions coincide
    # identically here (docs/zigzag.md 2.2).
    edges = edge_sets(cycle_lifecycle)
    for i in range(len(edges) - 1):
        assert edges[i] <= edges[i + 1] or edges[i + 1] <= edges[i], f"transition {i}"


def test_only_one_geometric_loop_transition_is_non_monotone(geometric_loop):
    """Named transition by transition, because the document used to name two.

    ``docs/zigzag.md`` §2.2 called both delta-transitions non-monotone.  Measured, only
    ``44 -> 45`` is: at ``19 -> 20`` the three chords across the gap vanish and nothing
    appears, so the step is a plain inclusion.  The asymmetry is the radius, not delta —
    at ``R = 1.258`` the three-step chord misses ``eps`` in both configurations, at
    ``R = 1.203`` it fits in the open one and is lost when the gap closes.
    """
    edges = edge_sets(geometric_loop)
    non_monotone = [
        i
        for i in range(len(edges) - 1)
        if not (edges[i] <= edges[i + 1] or edges[i + 1] <= edges[i])
    ]
    assert non_monotone == [44]

    assert edges[19] - edges[20] == GAP_CROSSING_EDGES
    assert edges[20] - edges[19] == set()
    assert edges[45] - edges[44] == GAP_CROSSING_EDGES
    assert len(edges[44] - edges[45]) == 21


# --- negative controls of the construction ----------------------------------------


BREATHING_SNAPSHOTS = (4, 43, 56)
"""``geometric_loop``: the graph changes here and no event happens (``method.md`` §2.1a)."""


def test_the_breathing_radius_changes_the_complex(geometric_loop, config):
    """The strongest negative control of the metric dataset, and it needs the change to be real.

    A detector that fires whenever the complex changes must have something to fire on;
    a test over a range would pass on a dataset whose complex never moved at all.
    """
    diagram = zigzag_complexes(geometric_loop, **build_args(config))
    indices = index_set(len(geometric_loop.snapshots))
    for snapshot in BREATHING_SNAPSHOTS:
        before, after = indices[2 * (snapshot - 1)], indices[2 * snapshot]
        assert diagram[before] != diagram[after], f"snapshot {snapshot}"


def test_the_repeated_state_is_not_collapsed(cycle_lifecycle, config):
    """Snapshots 41-47 and 55-59 are byte-identical, and the diagram keeps them apart.

    The repetition is deliberate (``method.md`` §2.1): the pipeline must not deduplicate
    states by graph identity, or the birth and death of :math:`H_0` at 48 and 55 collapse
    into nothing.
    """
    diagram = zigzag_complexes(cycle_lifecycle, **build_args(config))
    indices = index_set(len(cycle_lifecycle.snapshots))
    assert diagram[indices[2 * 41]] == diagram[indices[2 * 55]]
    assert len(diagram) == 2 * len(cycle_lifecycle.snapshots) - 1


def test_relabelling_the_nodes_gives_the_same_shape(cycle_lifecycle, config):
    """Vertex labels are not randomised in the data, so the invariance is a test.

    The complexes are not equal — the labels differ — but the diagram has the same length
    and the same face counts everywhere, which is what a construction blind to the order
    ``0, 1, ..., 23`` must produce.
    """
    rng = np.random.default_rng(20260201)
    order = rng.permutation(cycle_lifecycle.snapshots.shape[1])
    relabelled = cycle_lifecycle.snapshots[:, order][:, :, order]
    permuted = type(cycle_lifecycle)(
        snapshots=relabelled, ground_truth=cycle_lifecycle.ground_truth
    )

    original = zigzag_complexes(cycle_lifecycle, **build_args(config))
    shuffled = zigzag_complexes(permuted, **build_args(config))
    assert len(original) == len(shuffled)
    for index in original:
        assert original[index].face_counts() == shuffled[index].face_counts()


def test_a_dataset_without_an_edge_rule_is_refused(cycle_lifecycle, config):
    """The rule is a property of the dataset and is never guessed.

    Every dataset shipped today records it, so this is a guard rather than a case that
    occurs: a ``truth_kind`` that does not require ``edge_rule`` is admissible by the
    schema, and defaulting to one of the two rules would read a metric dataset as the
    complement of its graph without a word (``docs/method.md`` §2.3).
    """
    truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="dynamic_graph",
        truth_kind="none",
        generator="ruleless",
        params={"n_nodes": 24},
        seed=0,
    )
    without = type(cycle_lifecycle)(snapshots=cycle_lifecycle.snapshots, ground_truth=truth)
    with pytest.raises(ValueError, match="edge_rule"):
        zigzag_complexes(without, **build_args(config))
