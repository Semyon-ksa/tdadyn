"""Family 3: the dynamic stochastic block model.

Its truth is the partition and the schedule, and **nothing topological**.  That is not a
gap to be filled later: merging two communities is not a homological event at this
density, and :math:`\\beta_0` here tracks a stray isolated vertex rather than the block
structure (``docs/method.md`` §3.1).  The dataset lays out like a dynamic graph and
carries ``truth_kind: partition``; the tests below check the partition, not a barcode.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import dataset_equal, load_dataset, save_dataset

from conftest import build_dynamic_sbm

EXPECTED_COMMUNITY_COUNTS = ((0, 20, 4), (20, 40, 3), (40, 60, 4))
"""``(first snapshot, one past the last, number of communities)``.

Four blocks, two of them merge at 20, the merge is undone at 40 — so consecutive
partitions are related by refinement, as ``docs/method.md`` §3.1a requires of a
merge/split schedule.
"""


@pytest.fixture
def sbm(config):
    return build_dynamic_sbm(config)


def test_snapshots_have_the_declared_shape_and_dtype(sbm, config):
    settings = config.dynamic_sbm
    assert sbm.snapshots.shape == (settings.n_snapshots, settings.n_nodes, settings.n_nodes)
    assert sbm.snapshots.dtype == np.float64


def test_snapshots_are_symmetric_with_a_zero_diagonal(sbm):
    snapshots = sbm.snapshots
    assert np.array_equal(snapshots, np.swapaxes(snapshots, -1, -2))
    diagonals = np.diagonal(snapshots, axis1=1, axis2=2)
    assert np.array_equal(diagonals, np.zeros_like(diagonals))


def test_snapshots_are_unweighted(sbm):
    assert set(np.unique(sbm.snapshots).tolist()) <= {0.0, 1.0}


def test_the_truth_is_the_partition_and_carries_no_barcode(sbm, config):
    # docs/method.md 3.1: the rejected alternatives are recorded there so this does not
    # get "fixed" into a measured Betti curve later.
    ground_truth = sbm.ground_truth
    assert ground_truth.family == "dynamic_graph"
    assert ground_truth.truth_kind == "partition"
    assert ground_truth.betti_curve is None
    assert ground_truth.events is None
    assert ground_truth.community_labels.dtype == np.int64
    assert ground_truth.community_labels.shape == (
        config.dynamic_sbm.n_snapshots,
        config.dynamic_sbm.n_nodes,
    )


def test_the_schedule_merges_and_then_splits(sbm):
    labels = sbm.ground_truth.community_labels
    for start, stop, expected in EXPECTED_COMMUNITY_COUNTS:
        for index in range(start, stop):
            assert len(set(labels[index].tolist())) == expected, f"snapshot {index}"


def test_consecutive_partitions_are_related_by_refinement(sbm):
    # docs/method.md 3.1a: otherwise the transition is a re-cut, not a merge or a split.
    labels = sbm.ground_truth.community_labels
    for index in range(1, len(labels)):
        earlier, later = labels[index - 1], labels[index]
        if np.array_equal(earlier, later):
            continue
        coarser, finer = (earlier, later) if _refines(later, earlier) else (later, earlier)
        assert _refines(finer, coarser), f"snapshots {index - 1} and {index} are a re-cut"


def test_the_labels_describe_the_snapshots_they_sit_beside(sbm, config):
    # The link between truth and data: blocks that the labels call one community must be
    # denser than pairs the labels separate. Without this the partition could be any
    # array of the right shape.
    settings = config.dynamic_sbm
    labels = sbm.ground_truth.community_labels
    off_diagonal = ~np.eye(settings.n_nodes, dtype=bool)
    for index in (0, settings.merge_at, settings.n_snapshots - 1):
        same = (labels[index][:, None] == labels[index][None, :]) & off_diagonal
        within = sbm.snapshots[index][same].mean()
        between = sbm.snapshots[index][off_diagonal & ~same].mean()
        assert within > between, f"snapshot {index}: {within:.3f} vs {between:.3f}"


def test_the_generator_is_deterministic(config):
    assert dataset_equal(build_dynamic_sbm(config), build_dynamic_sbm(config))


def test_the_global_seed_does_not_leak(config, sbm):
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(build_dynamic_sbm(config), sbm)


def test_a_different_seed_gives_different_edges(config, sbm):
    assert not dataset_equal(build_dynamic_sbm(config, seed=7), sbm)


def test_the_dataset_survives_a_round_trip(sbm, config, tmp_path):
    path = save_dataset(sbm, tmp_path, float_precision=config.naming.float_precision)
    assert dataset_equal(load_dataset(path), sbm)


def test_merging_communities_is_not_a_homological_event(sbm, config):
    """``docs/method.md`` §3.1, as a test so the claim stays measured rather than asserted.

    The invariant is stated on the graph with isolated vertices removed, and deliberately
    not as "beta_0 is the same before and after the merge".  Measured over sixty
    snapshots, :math:`\\beta_0` takes both 1 and 2 on every calibration seed — not because
    communities separate, but because a node occasionally draws no edge at all
    (expected 3.5 such vertices per dataset, measured 2, 3 and 2).  Pinning raw
    :math:`\\beta_0` across the merge would therefore be pinning the absence of a stray
    vertex on two particular snapshots.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    settings = config.dynamic_sbm
    labels = sbm.ground_truth.community_labels
    off_diagonal = ~np.eye(settings.n_nodes, dtype=bool)

    def components_ignoring_isolated(snapshot):
        adjacency = snapshot > 0.5
        connected = adjacency.sum(axis=1) > 0
        sub = adjacency[np.ix_(connected, connected)]
        return connected_components(csr_matrix(sub), directed=False)[0]

    for index in range(settings.n_snapshots):
        count = components_ignoring_isolated(sbm.snapshots[index])
        assert count == 1, f"snapshot {index}: {count} components once isolates are dropped"

    # ... and the reason the merge cannot matter: the blocks were never separate.
    for index in (settings.merge_at - 1, settings.merge_at):
        same = (labels[index][:, None] == labels[index][None, :]) & off_diagonal
        cross_edges = int(sbm.snapshots[index][off_diagonal & ~same].sum() // 2)
        assert cross_edges > 5, f"snapshot {index}: only {cross_edges} cross-block edges"


def _refines(finer: np.ndarray, coarser: np.ndarray) -> bool:
    """True if every block of ``finer`` sits inside a single block of ``coarser``."""
    return all(
        len(set(coarser[finer == block].tolist())) == 1 for block in set(finer.tolist())
    )
