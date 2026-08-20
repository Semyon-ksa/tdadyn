"""Family 2: graph snapshots carrying a schedule of homological events.

The scenario follows ``docs/method.md`` §2.1 step by step, and the pairing of its births
with its deaths follows §2.2a.  Two properties of the construction are load-bearing and
are checked by the tests rather than assumed here:

* **No snapshot contains a triangle.**  A triangle would let the flag complex fill a
  cycle, and the event that reached the log would not be the event that happened.
* **The state at the end repeats the state before the split.**  Restoring the edge
  returns the graph exactly where it was, so five distinct graphs span sixty snapshots.
  That repeat is deliberate: it checks that a consumer does not collapse states by graph
  identity and report one event where there are two.

The Betti curve is computed by the reference implementation **from the snapshots that
were written**, never from the scheduler's own state.  The two are meant to be
independent sources: ``events`` is the intent, ``betti_curve`` is what the data says.
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.contracts import (
    FORMAT_VERSION,
    DynamicGraphDataset,
    Event,
    GroundTruth,
    adjacency_from_snapshot,
)
from tdadyn.core.naming import register_name_order
from tdadyn.core.rng import base_seed_of
from tdadyn.reference import REFERENCE_COEFF_FIELD, betti_numbers_of_graph

register_name_order(
    "cycle_lifecycle",
    [
        "n_nodes",
        "n_snapshots",
        "edge_weight",
        "threshold",
        (
            "sched",
            [
                "path_closes_at",
                "chord_added_at",
                "edge_removed_at",
                "component_splits_at",
                "component_merges_at",
            ],
        ),
    ],
)

CHORD_ENDS = (0, 12)
"""The chord added mid-scenario.  It splits the 24-cycle into two cycles of length 13,
so neither side is a triangle and the flag complex stays one-dimensional."""

REMOVED_EDGE = (6, 7)
"""Cut to kill a class of :math:`H_1`.  It lies on the arc the chord separated, and the
graph stays connected: 0..6 and 7..12 remain as pendant paths."""

SPLIT_EDGE = (3, 4)
"""Cut to split off a component, then restored.  It lies on a pendant path, so
:math:`\\beta_0` moves while :math:`\\beta_1` does not."""

_MIN_ARC_EDGES = 3
"""Shortest arc the chord may cut off.  Two edges plus the chord is a triangle, which the
flag complex fills; one edge means the chord *is* the closing edge."""

CYCLE_LIFECYCLE_NOTES = (
    "Сценарий детерминирован, rng не используется (docs/method.md, 2.1). Гибель H1 "
    "спарена с рождением при замыкании пути, а не с рождением от хорды: разложение "
    "зигзаг-модуля на интервалы однозначно (docs/method.md, 2.2a). Снимки 41-47 и 55-59 "
    "совпадают побайтово — повтор состояния намеренный."
)


def make_cycle_lifecycle(
    n_nodes: int,
    n_snapshots: int,
    edge_weight: float,
    threshold: float,
    path_closes_at: int,
    chord_added_at: int,
    edge_removed_at: int,
    component_splits_at: int,
    component_merges_at: int,
    coeff_field: int,
    max_homology_dim: int,
    reference_max_nodes: int,
    rng: np.random.Generator,
) -> DynamicGraphDataset:
    """Build the scheduled life cycle of a loop on a path of nodes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes; the path runs ``0, 1, ..., n_nodes - 1``.
    n_snapshots : int
        Length of the sequence.
    edge_weight : float
        Weight written on a present edge; an absent edge is 0.0.
    threshold : float
        Edges enter the flag complex where the weight is strictly above this value.
    path_closes_at : int
        Snapshot at which the path closes into a cycle — a birth in :math:`H_1`.
    chord_added_at : int
        Snapshot at which the chord divides the cycle — a second birth in :math:`H_1`.
    edge_removed_at : int
        Snapshot at which an edge of the divided arc is cut — a death in :math:`H_1`.
    component_splits_at : int
        Snapshot at which a pendant path is cut — a birth in :math:`H_0`.
    component_merges_at : int
        Snapshot at which that cut is restored — a death in :math:`H_0`.
    coeff_field : int
        Field of coefficients; only 2 is available, see Notes.
    max_homology_dim : int
        Highest homology dimension of the curve, from ``topology_check``.
    reference_max_nodes : int
        Node limit of the reference implementation.
    rng : numpy.random.Generator
        Required by the generator contract and **not used**: the scenario is
        deterministic.  Only the base seed is read, so that the recorded seed cannot
        disagree with the stream the dataset was addressed by.

    Returns
    -------
    DynamicGraphDataset
        Snapshots of shape ``(n_snapshots, n_nodes, n_nodes)`` with a ground truth
        holding both the event log and the Betti curve.

    Notes
    -----
    ``coeff_field`` must be 2: the reference implementation is deliberately not
    generalised (``docs/method.md`` §1.2), and computing over Z/2 while the truth claims
    another field would be the silent failure the explicit check exists to prevent.

    A snapshot index in the event log means that this snapshot **already** carries the
    change and its predecessor does not (``docs/method.md`` §2.1), so a jump of the curve
    belongs to the transition into it.
    """
    _check_common(
        n_nodes=n_nodes,
        n_snapshots=n_snapshots,
        coeff_field=coeff_field,
        max_homology_dim=max_homology_dim,
    )
    _check_cycle_lifecycle_arguments(
        n_nodes=n_nodes, edge_weight=edge_weight, threshold=threshold
    )
    schedule = (
        path_closes_at,
        chord_added_at,
        edge_removed_at,
        component_splits_at,
        component_merges_at,
    )
    _check_schedule(schedule, n_snapshots)

    snapshots = _build_snapshots(n_nodes, n_snapshots, edge_weight, schedule)
    betti_curve = _betti_curve(
        snapshots,
        edge_rule="weight_above",
        edge_threshold=threshold,
        max_homology_dim=max_homology_dim,
        reference_max_nodes=reference_max_nodes,
        coeff_field=coeff_field,
    )
    params = {
        "n_nodes": int(n_nodes),
        "n_snapshots": int(n_snapshots),
        "edge_weight": float(edge_weight),
        "threshold": float(threshold),
        "path_closes_at": int(path_closes_at),
        "chord_added_at": int(chord_added_at),
        "edge_removed_at": int(edge_removed_at),
        "component_splits_at": int(component_splits_at),
        "component_merges_at": int(component_merges_at),
    }
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="dynamic_graph",
        truth_kind="betti_curve",
        generator="cycle_lifecycle",
        params=params,
        seed=base_seed_of(rng),
        coeff_field=int(coeff_field),
        complex="flag",
        # Similarities, so an edge is a weight strictly above the threshold. Stated
        # rather than assumed: geometric_loop shares this family and reads the other way.
        edge_rule="weight_above",
        edge_threshold=float(threshold),
        betti_curve=betti_curve,
        events=_events(schedule),
        notes=CYCLE_LIFECYCLE_NOTES,
    )
    return DynamicGraphDataset(snapshots=snapshots, ground_truth=ground_truth)


register_name_order(
    "geometric_loop",
    [
        "n_points",
        "n_snapshots",
        "rips_epsilon",
        "gap_angle",
        "radius_schedule",
        ("sched", ["gap_opens_at", "gap_closes_at"]),
    ],
)

GEOMETRIC_LOOP_NOTES = (
    "Метрический сценарий: в snapshots.npy лежат РАССТОЯНИЯ, ребро при d <= eps "
    "(edge_rule=distance_within, docs/method.md 2.3). Закрытое состояние дуги — "
    "delta = 2*pi/n, а не ноль: при нуле точки 0 и n-1 совпадают, а beta этого не "
    "замечает (docs/method.md 2.1a). Дыхание радиуса меняет граф, но не beta; "
    "события даёт только разрыв дуги."
)


def make_geometric_loop(
    n_points: int,
    n_snapshots: int,
    rips_epsilon: float,
    gap_angle: float,
    radius_schedule: tuple[float, ...],
    gap_opens_at: int,
    gap_closes_at: int,
    coeff_field: int,
    max_homology_dim: int,
    reference_max_nodes: int,
    rng: np.random.Generator,
) -> DynamicGraphDataset:
    """Build a breathing circle whose arc opens and closes on schedule.

    Parameters
    ----------
    n_points : int
        Number of points on the circle.
    n_snapshots : int
        Length of the sequence.
    rips_epsilon : float
        Connection radius: an edge wherever the distance is at most this.  It must stay
        inside the regime window of ``docs/method.md`` §2.1a, which the config checks.
    gap_angle : float
        Angular width of the open gap, in radians.
    radius_schedule : sequence of float
        Radii to interpolate through, piecewise-linearly, with knots spread evenly over
        the sequence.  The breathing carries no events.
    gap_opens_at : int
        Snapshot at which the arc opens — a death in :math:`H_1`.
    gap_closes_at : int
        Snapshot at which it closes again — a birth in :math:`H_1`.
    coeff_field : int
        Field of coefficients; only 2 is available.
    max_homology_dim : int
        Highest homology dimension of the curve.
    reference_max_nodes : int
        Node limit of the reference implementation.
    rng : numpy.random.Generator
        Required by the generator contract and not used: the scenario is deterministic.

    Returns
    -------
    DynamicGraphDataset
        Snapshots of shape ``(n_snapshots, n_points, n_points)`` holding **distances**,
        with a ground truth whose ``edge_rule`` is ``"distance_within"``.

    Notes
    -----
    The closed arc is :math:`\\delta = 2\\pi/n`, not zero.  With
    :math:`\\theta_j = j(2\\pi-\\delta)/(n-1)`, zero puts :math:`\\theta_{n-1}` on top of
    :math:`\\theta_0`; substituting :math:`2\\pi/n` makes the step exactly :math:`2\\pi/n`,
    that is the uniform circle.  Betti numbers do not distinguish the two, so the
    degenerate variant would sit in the data unnoticed — the geometry is what catches it.
    """
    _check_common(
        n_nodes=n_points,
        n_snapshots=n_snapshots,
        coeff_field=coeff_field,
        max_homology_dim=max_homology_dim,
    )
    if rips_epsilon <= 0.0 or not np.isfinite(rips_epsilon):
        raise ValueError(f"rips_epsilon must be positive and finite, got {rips_epsilon!r}")
    if gap_angle <= 0.0 or gap_angle >= 2 * np.pi:
        raise ValueError(f"gap_angle must lie in (0, 2*pi), got {gap_angle!r}")
    if len(radius_schedule) < 2:
        raise ValueError("radius_schedule needs at least two knots to interpolate between")
    _check_schedule((gap_opens_at, gap_closes_at), n_snapshots)

    closed_gap = 2 * np.pi / n_points
    if gap_angle <= closed_gap:
        raise ValueError(
            f"gap_angle ({gap_angle}) does not exceed the closed state 2*pi/n "
            f"({closed_gap}); the arc would never open"
        )

    radii = np.interp(
        np.arange(n_snapshots),
        np.linspace(0.0, n_snapshots - 1, len(radius_schedule)),
        np.asarray(radius_schedule, dtype=np.float64),
    )
    snapshots = np.zeros((n_snapshots, n_points, n_points), dtype=np.float64)
    for index in range(n_snapshots):
        gap = gap_angle if gap_opens_at <= index < gap_closes_at else closed_gap
        snapshots[index] = _circle_distances(n_points, radii[index], gap)

    betti_curve = _betti_curve(
        snapshots,
        edge_rule="distance_within",
        edge_threshold=rips_epsilon,
        max_homology_dim=max_homology_dim,
        reference_max_nodes=reference_max_nodes,
        coeff_field=coeff_field,
    )
    params = {
        "n_points": int(n_points),
        "n_snapshots": int(n_snapshots),
        "rips_epsilon": float(rips_epsilon),
        "gap_angle": float(gap_angle),
        "radius_schedule": [float(value) for value in radius_schedule],
        "gap_opens_at": int(gap_opens_at),
        "gap_closes_at": int(gap_closes_at),
    }
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="dynamic_graph",
        truth_kind="betti_curve",
        generator="geometric_loop",
        params=params,
        seed=base_seed_of(rng),
        coeff_field=int(coeff_field),
        complex="flag",
        edge_rule="distance_within",
        edge_threshold=float(rips_epsilon),
        betti_curve=betti_curve,
        events=[
            Event(gap_opens_at, 1, "death", "h1_loop", "дуга разрывается, петля исчезает"),
            Event(gap_closes_at, 1, "birth", "h1_loop", "дуга смыкается, петля возвращается"),
        ],
        notes=GEOMETRIC_LOOP_NOTES,
    )
    return DynamicGraphDataset(snapshots=snapshots, ground_truth=ground_truth)


def _circle_distances(n_points: int, radius: float, gap: float) -> np.ndarray:
    """Pairwise distances between points spread evenly over an arc of ``2*pi - gap``."""
    angles = np.arange(n_points) * (2 * np.pi - gap) / (n_points - 1)
    points = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    # Exact symmetry and an exact zero diagonal: DynamicGraphDataset compares bitwise.
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    return distances


def _build_snapshots(
    n_nodes: int, n_snapshots: int, edge_weight: float, schedule: tuple[int, ...]
) -> np.ndarray:
    """Write the six phases of ``docs/method.md`` §2.1 into a snapshot array."""
    closes, chord, removed, splits, merges = schedule
    edges = {(node, node + 1) for node in range(n_nodes - 1)}

    snapshots = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=np.float64)
    for index in range(n_snapshots):
        if index == closes:
            _add(edges, (0, n_nodes - 1))
        elif index == chord:
            _add(edges, CHORD_ENDS)
        elif index == removed:
            _remove(edges, REMOVED_EDGE)
        elif index == splits:
            _remove(edges, SPLIT_EDGE)
        elif index == merges:
            _add(edges, SPLIT_EDGE)
        for left, right in edges:
            # Both cells are written explicitly: DynamicGraphDataset demands exact
            # symmetry, and building one triangle and mirroring it invites a drift.
            snapshots[index, left, right] = edge_weight
            snapshots[index, right, left] = edge_weight
    return snapshots


def _add(edges: set[tuple[int, int]], edge: tuple[int, int]) -> None:
    """Add a scheduled edge, refusing a no-op.

    ``set.add`` and ``set.discard`` are silent when there is nothing to do, and a
    scheduled change that quietly does nothing leaves the event log claiming a birth or a
    death the data never had — the same silent-wrong-truth failure the node count is
    checked against.
    """
    if edge in edges:
        raise ValueError(f"edge {edge} is already present; the scheduled change is a no-op")
    edges.add(edge)


def _remove(edges: set[tuple[int, int]], edge: tuple[int, int]) -> None:
    """Remove a scheduled edge, refusing a no-op.  See :func:`_add`."""
    if edge not in edges:
        raise ValueError(f"edge {edge} is absent; the scheduled change is a no-op")
    edges.remove(edge)


def _betti_curve(
    snapshots: np.ndarray,
    *,
    edge_rule: str,
    edge_threshold: float,
    max_homology_dim: int,
    reference_max_nodes: int,
    coeff_field: int,
) -> dict[int, list[int]]:
    """Betti numbers of every snapshot, from the reference implementation.

    The snapshot is turned into a graph by :func:`adjacency_from_snapshot` and never by
    comparing against the threshold here: the two edge rules point in opposite directions
    and applying the wrong one yields the complement of the intended graph
    (``docs/method.md`` §2.3).  The reference then works on a 0/1 matrix, where the
    similarity convention is the correct one whichever rule produced it.

    Identical snapshots are computed once.  The scenarios hold a handful of distinct
    graphs over sixty snapshots, and the reference is deliberately slow, so the memo is
    what keeps these generators usable outside the slow tests.  It cannot change the
    answer — the same graph has the same Betti numbers — and the curve still carries one
    entry per snapshot; a repeated state lives in the data, not in the computation.
    """
    curve: dict[int, list[int]] = {dim: [] for dim in range(max_homology_dim + 1)}
    seen: dict[bytes, dict[int, int]] = {}
    for snapshot in snapshots:
        key = snapshot.tobytes()
        betti = seen.get(key)
        if betti is None:
            adjacency = adjacency_from_snapshot(
                snapshot, edge_rule=edge_rule, edge_threshold=edge_threshold
            )
            betti = betti_numbers_of_graph(
                adjacency,
                max_homology_dim=max_homology_dim,
                max_nodes=reference_max_nodes,
                coeff_field=coeff_field,
            )
            seen[key] = betti
        for dim, values in curve.items():
            values.append(int(betti[dim]))
    return curve


def _events(schedule: tuple[int, ...]) -> list[Event]:
    """The scheduler's intent, independent of the Betti curve.

    The death of :math:`H_1` is paired with the birth at the closure of the path, not
    with the birth from the chord.  This is forced by the interval decomposition of the
    zigzag module and is not a bookkeeping choice; the argument, and the counter-example
    of removing the chord instead, are in ``docs/method.md`` §2.2a.
    """
    closes, chord, removed, splits, merges = schedule
    return [
        Event(closes, 1, "birth", "h1_a", "путь замыкается в цикл"),
        Event(chord, 1, "birth", "h1_b", "хорда делит цикл на два"),
        Event(removed, 1, "death", "h1_a", "удалено ребро дуги, отрезанной хордой"),
        Event(splits, 0, "birth", "h0_b", "от висячего пути отделяется компонента"),
        Event(merges, 0, "death", "h0_b", "компоненты сливаются обратно"),
    ]


def _check_common(*, n_nodes: int, n_snapshots: int, coeff_field: int, max_homology_dim: int) -> None:
    """Validate what every family-2 scenario needs, whatever its edge rule."""
    if coeff_field != REFERENCE_COEFF_FIELD:
        raise ValueError(
            f"coeff_field is {coeff_field!r}, but the Betti curve of family 2 is computed "
            "by the reference implementation, which works only over Z/2 "
            "(docs/method.md, 1.2). Computing over Z/2 while the truth claims another "
            "field would be a silent error."
        )
    if n_nodes < 3:
        raise ValueError(f"n_nodes must be >= 3, got {n_nodes}")
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    if max_homology_dim < 0:
        raise ValueError(f"max_homology_dim must be >= 0, got {max_homology_dim}")


def _check_cycle_lifecycle_arguments(
    *, n_nodes: int, edge_weight: float, threshold: float
) -> None:
    """Validate what is specific to the combinatorial scenario."""
    # The chord and the two cut edges are fixed node indices, so a graph that is too
    # short quietly stops being the scenario the truth describes. The binding constraint
    # is not that the indices exist but that the chord divides the cycle into two arcs of
    # at least three edges each: at two the chord closes a triangle, at one it coincides
    # with the closing edge and adding it is a no-op. Either way beta_1 never reaches 2
    # while the log still claims a birth.
    low, high = CHORD_ENDS
    arcs = (high - low, n_nodes - high + low)
    if min(arcs) < _MIN_ARC_EDGES:
        raise ValueError(
            f"n_nodes={n_nodes} leaves the cycle_lifecycle chord {CHORD_ENDS} with arcs "
            f"of {arcs} edges; each needs at least {_MIN_ARC_EDGES}, otherwise the chord "
            "closes a triangle or duplicates the closing edge and the second birth in H1 "
            "never happens"
        )
    if max(max(REMOVED_EDGE), max(SPLIT_EDGE)) >= n_nodes:
        raise ValueError(
            f"n_nodes={n_nodes} is too small for the cycle_lifecycle cuts "
            f"{REMOVED_EDGE} and {SPLIT_EDGE}"
        )
    if not np.isfinite(edge_weight) or edge_weight <= 0.0:
        raise ValueError(f"edge_weight must be a positive finite number, got {edge_weight!r}")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(f"threshold must be a non-negative finite number, got {threshold!r}")
    if threshold >= edge_weight:
        raise ValueError(
            f"threshold ({threshold}) must be below edge_weight ({edge_weight}), "
            "otherwise no edge ever enters the complex"
        )


def _check_schedule(schedule: tuple[int, ...], n_snapshots: int) -> None:
    """The schedule must be strictly increasing, start at 1 and fit in the sequence."""
    if schedule[0] < 1:
        # A jump belongs to the transition k-1 -> k, and snapshot 0 has no predecessor.
        raise ValueError(
            f"the first scheduled event is at snapshot {schedule[0]}; it must be at "
            "least 1, since an event on snapshot 0 has no preceding snapshot"
        )
    for earlier, later in zip(schedule, schedule[1:]):
        if later <= earlier:
            raise ValueError(
                f"the schedule must be strictly increasing, got {schedule}; events "
                "sharing a snapshot cannot be told apart in the Betti curve"
            )
    if schedule[-1] >= n_snapshots:
        raise ValueError(
            f"the last scheduled event is at snapshot {schedule[-1]}, outside "
            f"n_snapshots ({n_snapshots})"
        )
