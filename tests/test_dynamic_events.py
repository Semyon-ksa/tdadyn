"""Family 2: the ``cycle_lifecycle`` scenario and its event log.

The expectations below are transcribed from ``docs/method.md`` §2.1 and §2.2a and from
``docs/data.md`` §4.  They are an **input**: if a test does not converge, the fault is in
the generator, never in this table (``docs/data.md`` §4).  Nothing here is imported from
:mod:`tdadyn.synthetic.dynamic_events` — a test that took its expectation from the module
under test would confirm itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import (
    adjacency_from_snapshot,
    dataset_equal,
    load_dataset,
    make_dataset_id,
    rng_for,
    save_dataset,
)
from tdadyn import reference
from tdadyn.synthetic import make_cycle_lifecycle

from conftest import build_dynamic_graph

EXPECTED_PHASES = (
    (0, 12, (1, 0, 0)),
    (12, 27, (1, 1, 0)),
    (27, 41, (1, 2, 0)),
    (41, 48, (1, 1, 0)),
    (48, 55, (2, 1, 0)),
    (55, 60, (1, 1, 0)),
)
"""``(first snapshot, one past the last, (beta_0, beta_1, beta_2))`` per phase."""

EXPECTED_EVENTS = (
    (12, 1, "birth", "h1_a"),
    (27, 1, "birth", "h1_b"),
    (41, 1, "death", "h1_a"),
    (48, 0, "birth", "h0_b"),
    (55, 0, "death", "h0_b"),
)
"""The log from ``docs/data.md`` §4.  The death at 41 is paired with the birth at 12,
not with the one at 27; that is forced, not chosen (``docs/method.md`` §2.2a)."""

EXPECTED_DATASET_ID = (
    "cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201"
)

REMOVED_EDGE = (6, 7)
"""The edge cut at snapshot 41, on the arc the chord separated."""


@pytest.fixture
def dynamic_graph(config):
    return build_dynamic_graph("cycle_lifecycle", config)


def reference_args(config):
    """Arguments the reference implementation takes for a snapshot of this family."""
    return {
        "max_homology_dim": config.topology_check.max_homology_dim,
        "max_nodes": config.topology_check.reference_max_nodes,
    }


def betti_tuple(betti):
    return tuple(betti[dim] for dim in sorted(betti))


def jumps_of(betti_curve):
    """Every change of the curve as ``(snapshot, dim, "birth" | "death")``.

    ``docs/method.md`` §2.1 fixes the convention: ``snapshot: k`` means that :math:`G_k`
    already carries the change, so a jump belongs to the transition ``k-1 -> k``.  A step
    of two counts as two events, even though this scenario has none.
    """
    jumps = []
    for dim, values in betti_curve.items():
        for index in range(1, len(values)):
            delta = values[index] - values[index - 1]
            kind = "birth" if delta > 0 else "death"
            jumps.extend([(index, dim, kind)] * abs(delta))
    return sorted(jumps)


# --- shape of the data ------------------------------------------------------------


def test_snapshots_have_the_declared_shape_and_dtype(dynamic_graph, config):
    events = config.dynamic_events
    assert dynamic_graph.snapshots.shape == (
        events.n_snapshots,
        events.n_nodes,
        events.n_nodes,
    )
    assert dynamic_graph.snapshots.dtype == np.float64


def test_snapshots_are_symmetric_with_zero_diagonal(dynamic_graph):
    snapshots = dynamic_graph.snapshots
    assert np.array_equal(snapshots, np.swapaxes(snapshots, -1, -2))
    diagonals = np.diagonal(snapshots, axis1=1, axis2=2)
    assert np.array_equal(diagonals, np.zeros_like(diagonals))


def test_weights_take_only_two_values(dynamic_graph, config):
    events = config.dynamic_events
    assert set(np.unique(dynamic_graph.snapshots).tolist()) == {0.0, events.edge_weight}
    # The threshold has to fall strictly between them, or the graph is empty or complete.
    assert 0.0 <= events.threshold < events.edge_weight


def test_no_snapshot_contains_a_triangle(dynamic_graph, config):
    # docs/method.md 2.1 requires this: a triangle would let the flag complex fill a
    # cycle, and the logged event would not be the event that happened. Making it a test
    # turns the agreement of beta_1 with the cyclomatic number here into a consequence of
    # the construction rather than a coincidence.
    adjacency = (dynamic_graph.snapshots > config.dynamic_events.threshold).astype(np.int64)
    for index, edges in enumerate(adjacency):
        closed_walks = np.trace(edges @ edges @ edges)  # six per triangle
        assert closed_walks == 0, f"snapshot {index} contains a triangle"


def test_the_repeated_state_is_present_in_the_data(dynamic_graph, config):
    # docs/method.md 2.1: restoring (3, 4) returns the graph to the state it held between
    # the edge removal and the split, so five distinct graphs span sixty snapshots. The
    # repeat is deliberate and must survive into the data.
    snapshots = dynamic_graph.snapshots
    assert np.array_equal(snapshots[41], snapshots[55])
    assert len({snapshot.tobytes() for snapshot in snapshots}) == 5
    lengths = {len(values) for values in dynamic_graph.ground_truth.betti_curve.values()}
    assert lengths == {config.dynamic_events.n_snapshots}


# --- reproducibility --------------------------------------------------------------


def test_the_generator_is_deterministic(config):
    first = build_dynamic_graph("cycle_lifecycle", config)
    second = build_dynamic_graph("cycle_lifecycle", config)
    assert dataset_equal(first, second)


def test_the_global_seed_does_not_leak(config, dynamic_graph):
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(build_dynamic_graph("cycle_lifecycle", config), dynamic_graph)


def test_the_dataset_survives_a_round_trip(dynamic_graph, config, tmp_path):
    path = save_dataset(
        dynamic_graph, tmp_path, float_precision=config.naming.float_precision
    )
    assert dataset_equal(load_dataset(path), dynamic_graph)


def test_the_dataset_id_matches_the_documented_name(dynamic_graph, config):
    # docs/data.md 1.1: the five schedule keys collapse into one token, and every one of
    # them still has to reach the name — two schedules must not share a directory.
    name = make_dataset_id(
        dynamic_graph.ground_truth, float_precision=config.naming.float_precision
    )
    assert name == EXPECTED_DATASET_ID


@pytest.mark.parametrize("n_nodes", [13, 14])
def test_a_node_count_that_degrades_the_scenario_is_refused(config, n_nodes):
    # The chord ends are fixed node indices, so a short enough graph quietly stops being
    # the scenario the truth describes: at 13 nodes the chord coincides with the closing
    # edge and adding it is a no-op, at 14 it closes a triangle. Both leave beta_1 at 1
    # while the log still claims a birth, which is exactly the plausible-looking dataset
    # with a wrong truth that docs/validation.md opens with.
    with pytest.raises(ValueError, match="cycle_lifecycle"):
        build_dynamic_graph("cycle_lifecycle", config, n_nodes=n_nodes)


def test_a_field_other_than_z2_is_refused(config):
    # The reference is Z/2 only (docs/method.md 1.2); coeff_field must not arrive as 3
    # and be quietly computed over Z/2 anyway.
    with pytest.raises(ValueError, match="Z/2"):
        build_dynamic_graph("cycle_lifecycle", config, coeff_field=3)


# --- the truth --------------------------------------------------------------------


def test_the_event_log_matches_the_documented_schedule(dynamic_graph):
    events = dynamic_graph.ground_truth.events
    actual = tuple(
        (event.snapshot, event.dim, event.type, event.feature_id) for event in events
    )
    assert actual == EXPECTED_EVENTS


def test_every_death_pairs_with_an_earlier_birth(dynamic_graph):
    born_at = {}
    for event in dynamic_graph.ground_truth.events:
        key = (event.dim, event.feature_id)
        if event.type == "birth":
            assert key not in born_at, f"{key} is born twice"
            born_at[key] = event.snapshot
        else:
            assert key in born_at, f"{key} dies without a birth"
            assert born_at[key] < event.snapshot


def test_the_curve_jumps_exactly_where_the_events_say(dynamic_graph):
    # The core of the session. events and betti_curve are two independent sources: the
    # first is the scheduler's intent, the second is computed by the reference. Comparing
    # multisets of (position, dimension, direction) catches an unscheduled jump, an event
    # without a jump, and a position off by one — the last being invisible inside family 2
    # and fatal in Task 2 (docs/method.md 2.1).
    ground_truth = dynamic_graph.ground_truth
    from_events = sorted(
        (event.snapshot, event.dim, event.type) for event in ground_truth.events
    )
    assert jumps_of(ground_truth.betti_curve) == from_events


def test_the_declared_curve_matches_the_expected_phases(dynamic_graph):
    curve = dynamic_graph.ground_truth.betti_curve
    for start, stop, expected in EXPECTED_PHASES:
        for index in range(start, stop):
            actual = tuple(curve[dim][index] for dim in sorted(curve))
            assert actual == expected, f"snapshot {index}"


@pytest.mark.slow
@pytest.mark.parametrize(
    ("start", "stop", "expected"), EXPECTED_PHASES, ids=lambda value: str(value)
)
def test_each_phase_has_the_expected_betti_numbers(
    dynamic_graph, config, start, stop, expected
):
    snapshots = dynamic_graph.snapshots
    for index in range(start, stop):
        assert np.array_equal(snapshots[index], snapshots[start]), f"snapshot {index}"
    betti = reference.betti_numbers_of_graph(snapshots[start], **reference_args(config))
    assert betti_tuple(betti) == expected


@pytest.mark.slow
def test_the_curve_matches_the_snapshots_on_disk(dynamic_graph, config, tmp_path):
    # The generator memoises the reference by snapshot; recomputing without the memo, from
    # the arrays that actually reached the disk, is the guard on both that shortcut and on
    # a curve taken from the scheduler's state instead of from the data.
    path = save_dataset(
        dynamic_graph, tmp_path, float_precision=config.naming.float_precision
    )
    loaded = load_dataset(path)
    recomputed = {dim: [] for dim in sorted(loaded.ground_truth.betti_curve)}
    for snapshot in loaded.snapshots:
        betti = reference.betti_numbers_of_graph(snapshot, **reference_args(config))
        for dim in recomputed:
            recomputed[dim].append(betti[dim])
    assert recomputed == loaded.ground_truth.betti_curve


@pytest.mark.slow
def test_the_dying_class_is_the_one_born_at_the_closure(dynamic_graph, config):
    """The pairing of ``docs/method.md`` §2.2a, in its canonical part.

    Over Z/2 let ``c`` be the 24-cycle and ``c2`` the cycle carried by the arc
    ``12..23..0`` together with the chord.  After the removal of ``(6, 7)`` the surviving
    cycle space is one-dimensional and contains ``c2`` but not ``c``, so the class born at
    the closure is the one that dies.  The identity of a class is *not* asserted: a basis
    of :math:`H_1` of the theta graph is not canonical.
    """
    threshold = config.dynamic_events.threshold
    theta = dynamic_graph.snapshots[40]
    broken = dynamic_graph.snapshots[41]
    assert theta[REMOVED_EDGE] > threshold
    assert broken[REMOVED_EDGE] <= threshold

    complex_theta = reference.flag_complex(theta, **reference_args(config))
    edges = complex_theta.simplices(1)
    position = {edge: index for index, edge in enumerate(edges)}
    boundary = reference.boundary_matrix(complex_theta, 1).astype(np.int64)

    def chain(pairs):
        vector = np.zeros(len(edges), dtype=np.int64)
        for pair in pairs:
            vector[position[tuple(sorted(pair))]] = 1
        return vector

    cycle_24 = chain([(node, node + 1) for node in range(23)] + [(23, 0)])
    c2 = chain(
        [(node, node + 1) for node in range(12, 23)] + [(23, 0), (0, 12)]
    )
    assert not (boundary @ cycle_24 % 2).any(), "c is not a cycle of the theta graph"
    assert not (boundary @ c2 % 2).any(), "c2 is not a cycle of the theta graph"

    # Z_1(G_41) sits inside C_1(G_40) as the chains supported off the removed edge.
    assert cycle_24[position[REMOVED_EDGE]] == 1
    assert c2[position[REMOVED_EDGE]] == 0
    # ... and c2 spans all of it, so nothing else could have survived instead.
    assert reference.betti_numbers_of_graph(broken, **reference_args(config))[1] == 1


@pytest.mark.slow
def test_relabelling_the_nodes_changes_nothing(dynamic_graph, config):
    # docs/method.md 2.1: the labels are not randomised in the data, so that snapshots stay
    # readable while debugging. The invariance is checked here instead.
    permutation = rng_for(config.seed, "relabel_check").permutation(
        config.dynamic_events.n_nodes
    )
    permuted = dynamic_graph.snapshots[:, permutation][:, :, permutation]
    curve = {dim: [] for dim in sorted(dynamic_graph.ground_truth.betti_curve)}
    for snapshot in permuted:
        betti = reference.betti_numbers_of_graph(snapshot, **reference_args(config))
        for dim in curve:
            curve[dim].append(betti[dim])
    assert curve == dynamic_graph.ground_truth.betti_curve


# --- geometric_loop: the metric scenario (docs/method.md 2.1a) ----------------------

LOOP_PHASES = ((0, 20, (1, 1, 0)), (20, 45, (1, 0, 0)), (45, 60, (1, 1, 0)))
"""Radius breathing carries no events; only the arc gap does."""

LOOP_EVENTS = ((20, 1, "death", "h1_loop"), (45, 1, "birth", "h1_loop"))

LOOP_BREATHING_SNAPSHOTS = (4, 43, 56)
"""Snapshots where the radius alone changes the graph, with no event attached.

Measured: 72 -> 48 edges at 4, 45 -> 66 at 43, 48 -> 72 at 56.  These are the negative
control of the dataset, and the test names them rather than sweeping the range: a range
sweep would also pass on a dataset whose complex never changed at all.
"""


@pytest.fixture
def geometric_loop(config):
    return build_dynamic_graph("geometric_loop", config)


def test_the_closed_arc_is_the_uniform_circle(geometric_loop, config):
    # docs/method.md 2.1a: delta = 0 would put points 0 and n-1 on top of each other, and
    # beta would not notice — (1,1,0) either way. The vertex doubling is visible only in
    # the geometry, so that is what is checked.
    loop = config.dynamic_events.scenarios.geometric_loop
    closed = geometric_loop.snapshots[0]
    off_diagonal = closed[~np.eye(loop.n_points, dtype=bool)]
    expected_step = 2 * np.sin(np.pi / loop.n_points) * loop.radius_schedule[0]
    assert off_diagonal.min() == pytest.approx(expected_step, rel=1e-9)
    assert off_diagonal.min() > 0.1


def test_the_closed_arc_has_the_face_counts_the_document_states(geometric_loop, config):
    """``docs/method.md`` §2.1a states (24, 72, 72, 24) for the closed arc; here it is.

    The point is not the numbers but the path to them.  The reference used to threshold
    its own input with ``adjacency > threshold`` — the similarity rule — and this dataset
    is metric, so that path built the complex of the **complement** graph.  Had the table
    been measured that way it would have been measured on the wrong graph, and nothing in
    the suite would have said so.  Measured through the one implementation of the edge
    rule, it agrees: the complement gives (24, 204, 728, 990), which is nowhere near.

    This also pins the 72 triangles behind the remark that closes §2.1a: the cyclomatic
    number here is 72 - 24 + 1 = 49 while beta_1 is 1, a discrepancy of 49 on real data.
    """
    loop = config.dynamic_events.scenarios.geometric_loop
    truth = geometric_loop.ground_truth
    adjacency = adjacency_from_snapshot(
        geometric_loop.snapshots[0],
        edge_rule=truth.edge_rule,
        edge_threshold=truth.edge_threshold,
    )
    complex_ = reference.flag_complex(
        adjacency,
        max_homology_dim=config.topology_check.max_homology_dim,
        max_nodes=config.topology_check.reference_max_nodes,
    )

    # The edge count is computed from the geometry rather than copied out of the table:
    # on the closed arc the points sit at angular step 2*pi/n on a circle of radius R, so
    # the chord across j steps is 2*R*sin(j*pi/n) and each point is joined to the
    # `reach` neighbours on either side whose chord fits inside eps.
    n_nodes, radius = loop.n_points, loop.radius_schedule[0]
    reach = sum(
        1
        for step in range(1, n_nodes // 2 + 1)
        if 2 * radius * np.sin(step * np.pi / n_nodes) <= loop.rips_epsilon
    )
    assert reach == 3
    assert complex_.face_counts() == (n_nodes, n_nodes * reach, 72, 24)

    # The remark that closes §2.1a: 72 triangles, so beta_1 is 1 where the cyclomatic
    # number is 49. This is what the metric scenario adds over `cycle_lifecycle`, which
    # has no triangles at all and where the two therefore agree by construction.
    n_edges = n_nodes * reach
    assert n_edges - n_nodes + 1 == 49
    assert truth.betti_curve[1][0] == 1


def test_the_loop_dataset_id_stays_readable(geometric_loop, config):
    # docs/data.md 1.1: the name exists so the directory says what is in it. Six raw
    # parameter names would defeat that, hence the abbreviations and the sched group.
    name = make_dataset_id(
        geometric_loop.ground_truth, float_precision=config.naming.float_precision
    )
    assert name == "geometric_loop_N24_T60_eps0.8_gap1.2_R1-1.4-1_sched20-45_seed20260201"


def test_the_snapshots_hold_distances_not_similarities(geometric_loop):
    ground_truth = geometric_loop.ground_truth
    assert ground_truth.edge_rule == "distance_within"
    snapshots = geometric_loop.snapshots
    assert np.array_equal(np.diagonal(snapshots, axis1=1, axis2=2), np.zeros((len(snapshots), snapshots.shape[1])))
    assert (snapshots >= 0.0).all()


def test_the_loop_curve_matches_the_expected_phases(geometric_loop):
    curve = geometric_loop.ground_truth.betti_curve
    for start, stop, expected in LOOP_PHASES:
        for index in range(start, stop):
            assert tuple(curve[dim][index] for dim in sorted(curve)) == expected, index


def test_the_loop_event_log_matches_the_schedule(geometric_loop):
    events = geometric_loop.ground_truth.events
    assert tuple((e.snapshot, e.dim, e.type, e.feature_id) for e in events) == LOOP_EVENTS


def test_the_loop_curve_jumps_exactly_where_the_events_say(geometric_loop):
    ground_truth = geometric_loop.ground_truth
    from_events = sorted((e.snapshot, e.dim, e.type) for e in ground_truth.events)
    assert jumps_of(ground_truth.betti_curve) == from_events


def test_the_breathing_radius_changes_the_graph_without_an_event(geometric_loop, config):
    # The strongest negative control in the dataset: the complex genuinely changes and no
    # event follows. Named snapshots, not a range — see LOOP_BREATHING_SNAPSHOTS.
    ground_truth = geometric_loop.ground_truth
    scheduled = {event.snapshot for event in ground_truth.events}
    curve = ground_truth.betti_curve

    def graph_at(index):
        return adjacency_from_snapshot(
            geometric_loop.snapshots[index],
            edge_rule=ground_truth.edge_rule,
            edge_threshold=ground_truth.edge_threshold,
        )

    for index in LOOP_BREATHING_SNAPSHOTS:
        assert index not in scheduled
        assert not np.array_equal(graph_at(index), graph_at(index - 1)), f"snapshot {index}"
        before = tuple(curve[dim][index - 1] for dim in sorted(curve))
        after = tuple(curve[dim][index] for dim in sorted(curve))
        assert before == after, f"snapshot {index} moved beta without an event"


def test_the_loop_has_four_distinct_graphs(geometric_loop):
    ground_truth = geometric_loop.ground_truth
    graphs = {
        adjacency_from_snapshot(
            snapshot,
            edge_rule=ground_truth.edge_rule,
            edge_threshold=ground_truth.edge_threshold,
        ).tobytes()
        for snapshot in geometric_loop.snapshots
    }
    assert len(graphs) == 4


def test_every_snapshot_stays_inside_both_regime_bounds(geometric_loop, config):
    # Connectivity below, the S^1 -> S^3 transition above. Checked on the data, not only
    # in the config, because the radius is interpolated and the config sees only knots.
    loop = config.dynamic_events.scenarios.geometric_loop
    epsilon = loop.rips_epsilon
    n_snapshots = len(geometric_loop.snapshots)
    radii = np.interp(
        np.arange(n_snapshots),
        np.linspace(0, n_snapshots - 1, len(loop.radius_schedule)),
        loop.radius_schedule,
    )
    # The radius is interpolated, so the config sees only the knots; the binding values
    # are the extremes actually reached.
    assert epsilon > 2 * np.pi * radii.max() / loop.n_points
    assert epsilon < radii.min() * np.sqrt(3.0)


@pytest.mark.slow
def test_the_loop_beta_1_is_far_from_the_cyclomatic_number(geometric_loop, config):
    # Unlike cycle_lifecycle, this complex is full of triangles, so the distinction of
    # docs/method.md 2.3 shows up on real data rather than on a 3-node fixture.
    ground_truth = geometric_loop.ground_truth
    adjacency = adjacency_from_snapshot(
        geometric_loop.snapshots[0],
        edge_rule=ground_truth.edge_rule,
        edge_threshold=ground_truth.edge_threshold,
    )
    n_nodes = adjacency.shape[0]
    n_edges = int(adjacency.sum() // 2)
    cyclomatic = n_edges - n_nodes + 1
    assert ground_truth.betti_curve[1][0] == 1
    assert cyclomatic > 40


@pytest.mark.slow
def test_the_loop_curve_matches_the_snapshots_on_disk(geometric_loop, config, tmp_path):
    path = save_dataset(geometric_loop, tmp_path, float_precision=config.naming.float_precision)
    loaded = load_dataset(path)
    ground_truth = loaded.ground_truth
    recomputed = {dim: [] for dim in sorted(ground_truth.betti_curve)}
    for snapshot in loaded.snapshots:
        adjacency = adjacency_from_snapshot(
            snapshot,
            edge_rule=ground_truth.edge_rule,
            edge_threshold=ground_truth.edge_threshold,
        )
        betti = reference.betti_numbers_of_graph(
            adjacency,
            max_homology_dim=config.topology_check.max_homology_dim,
            max_nodes=config.topology_check.reference_max_nodes,
        )
        for dim in recomputed:
            recomputed[dim].append(betti[dim])
    assert recomputed == ground_truth.betti_curve


# --- the edge rule -----------------------------------------------------------------


def test_weight_above_is_strict_and_distance_within_is_not():
    # docs/method.md 2.3: a weight equal to the threshold does not exceed it, while a
    # distance equal to epsilon is inside the radius of inclusion.
    matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    above = adjacency_from_snapshot(matrix, edge_rule="weight_above", edge_threshold=0.5)
    within = adjacency_from_snapshot(matrix, edge_rule="distance_within", edge_threshold=0.5)
    assert above[0, 1] == 0.0
    assert within[0, 1] == 1.0


def test_the_two_rules_give_complementary_graphs():
    # Why the rule must be recorded rather than inferred: read a metric dataset with the
    # similarity convention and you get the complement — connected, plausible, wrong.
    rng = rng_for(1, "edge_rule_check")
    matrix = rng.random((6, 6))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0.0)
    above = adjacency_from_snapshot(matrix, edge_rule="weight_above", edge_threshold=0.5)
    within = adjacency_from_snapshot(matrix, edge_rule="distance_within", edge_threshold=0.5)
    off_diagonal = ~np.eye(6, dtype=bool)
    assert np.array_equal(above[off_diagonal] + within[off_diagonal], np.ones(30))
    assert np.array_equal(np.diagonal(above), np.zeros(6))
    assert np.array_equal(np.diagonal(within), np.zeros(6))


def test_an_unknown_edge_rule_is_refused():
    with pytest.raises(ValueError, match="edge_rule must be one of"):
        adjacency_from_snapshot(np.zeros((2, 2)), edge_rule="nearest", edge_threshold=0.5)


@pytest.mark.slow
def test_the_recorded_edge_rule_is_the_one_that_reproduces_the_curve(dynamic_graph, config):
    # The field is load-bearing, not decorative: applying the other rule to the same
    # snapshots yields a different curve. Without this the rule could be recorded wrongly
    # and nothing inside the package would notice.
    ground_truth = dynamic_graph.ground_truth
    topology = config.topology_check

    def curve_under(rule):
        result = {dim: [] for dim in sorted(ground_truth.betti_curve)}
        for snapshot in dynamic_graph.snapshots:
            adjacency = adjacency_from_snapshot(
                snapshot, edge_rule=rule, edge_threshold=ground_truth.edge_threshold
            )
            betti = reference.betti_numbers_of_graph(
                adjacency,
                max_homology_dim=topology.max_homology_dim,
                max_nodes=topology.reference_max_nodes,
            )
            for dim in result:
                result[dim].append(betti[dim])
        return result

    assert curve_under(ground_truth.edge_rule) == ground_truth.betti_curve
    other = "distance_within" if ground_truth.edge_rule == "weight_above" else "weight_above"
    assert curve_under(other) != ground_truth.betti_curve


def test_flag_beta_1_is_not_the_cyclomatic_number(config):
    # docs/method.md 2.3: the triangle is filled, so beta_1 is 0 while the cyclomatic
    # number m - n + c is 1. Counting cycles instead of computing homology is the classic
    # silent error this family has to be safe from.
    weight = config.dynamic_events.edge_weight
    triangle = np.array(
        [[0.0, weight, weight], [weight, 0.0, weight], [weight, weight, 0.0]]
    )
    betti = reference.betti_numbers_of_graph(triangle, **reference_args(config))
    assert betti[1] == 0
    n_nodes, n_edges, n_components = 3, 3, 1
    assert n_edges - n_nodes + n_components == 1
