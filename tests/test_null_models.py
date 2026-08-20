"""Family 4: null models.

These datasets do not claim zero features — they let a false-positive rate be measured.
A random graph honestly has cycles at every workable density, and the significance
criterion has a non-zero false-positive rate on random clouds — 1.5% at the configured
size (``docs/method.md`` §1.2a, §4.2).  What is asserted here is a statement about
the *construction*: no schedule, an empty event log, and reproducibility.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn import reference
from tdadyn.core import dataset_equal, load_dataset, save_dataset
from tdadyn.synthetic import (
    cycle_lifecycle_density,
    edge_count,
    pairwise_distances,
    significant_betti,
)

from tdadyn.synthetic import build_dynamic_graph, null_edge_probability

from conftest import build_null_cloud, build_null_graph


def er_null(config, *, repeat=0):
    """The ER null over a ``cycle_lifecycle`` built once, as the CLI does."""
    return build_null_graph(config, build_dynamic_graph("cycle_lifecycle", config), repeat=repeat)

CLEAN = (1, 0, 0)
"""What "no features" looks like: one component, nothing above it.  The infinite bar of
:math:`H_0` is always significant, so zero everywhere is not the right expectation."""

DENSITY_TOLERANCE = 0.13
"""How far the mean edge count may sit from the target density, in relative terms.

Binomial: 4.59 edges of spread per snapshot, so 0.59 on a mean over sixty — 2.6% of the
23-edge target.  The bound is five of those, and the worst deviation observed over five
realisations is 4.8%.  Set from the sampling distribution rather than from the observed
spread, so it does not tighten to whatever this seed happened to give."""


@pytest.fixture
def null_cloud(config):
    return build_null_cloud(config)


@pytest.fixture
def null_graph(config):
    return er_null(config)


def measured_betti(points, config):
    topology = config.topology_check
    betti = significant_betti(
        pairwise_distances(points),
        max_homology_dim=2,
        coeff_field=topology.coeff_field,
        criterion=topology.criterion,
        gap_noise_floor_ratio=topology.gap_noise_floor_ratio,
        gap_min_ratio=topology.gap_min_ratio,
    )
    return tuple(betti[dim] for dim in range(3))


# --- the truth record ---------------------------------------------------------------


@pytest.mark.parametrize("which", ["cloud", "graph"])
def test_the_truth_is_an_empty_event_log(config, which):
    dataset = build_null_cloud(config) if which == "cloud" else er_null(config)
    ground_truth = dataset.ground_truth
    assert ground_truth.family == "null"
    assert ground_truth.truth_kind == "none"
    # The empty list is the claim. Absent would mean "not stated"; empty means "stated to
    # be nothing", and only the second is a truth a null model can offer.
    assert ground_truth.events == []
    assert ground_truth.betti_curve is None
    assert ground_truth.expected_betti is None
    assert ground_truth.community_labels is None


@pytest.mark.parametrize("which", ["cloud", "graph"])
def test_the_dataset_survives_a_round_trip(config, which, tmp_path):
    dataset = build_null_cloud(config) if which == "cloud" else er_null(config)
    path = save_dataset(dataset, tmp_path, float_precision=config.naming.float_precision)
    assert dataset_equal(load_dataset(path), dataset)


@pytest.mark.parametrize("which", ["cloud", "graph"])
def test_the_generator_is_deterministic(config, which):
    build = build_null_cloud if which == "cloud" else er_null
    assert dataset_equal(build(config), build(config))


@pytest.mark.parametrize("which", ["cloud", "graph"])
def test_the_global_seed_does_not_leak(config, which):
    build = build_null_cloud if which == "cloud" else er_null
    expected = build(config)
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(build(config), expected)


@pytest.mark.parametrize("which", ["cloud", "graph"])
def test_realisations_differ(config, which):
    build = build_null_cloud if which == "cloud" else er_null
    assert not dataset_equal(build(config, repeat=0), build(config, repeat=1))


# --- the ER null (docs/method.md 4.2) -----------------------------------------------


def test_the_er_null_matches_the_density_of_cycle_lifecycle(config):
    base = build_dynamic_graph("cycle_lifecycle", config)
    probability = null_edge_probability(config, base)
    null = build_null_graph(config, base)
    events = config.dynamic_events
    expected_edges = probability * events.n_nodes * (events.n_nodes - 1) / 2
    truth = null.ground_truth
    counts = [
        edge_count(snapshot, edge_rule=truth.edge_rule, edge_threshold=truth.edge_threshold)
        for snapshot in null.snapshots
    ]
    # Binomial around the target: the mean over sixty snapshots lands close, one snapshot
    # need not.
    assert abs(np.mean(counts) - expected_edges) < DENSITY_TOLERANCE * expected_edges


def test_the_er_snapshots_are_independent(config, null_graph):
    # Drawn per snapshot, not one perturbation repeated: a constant sequence would have
    # no fluctuation, and the fluctuation is exactly the null distribution task 2 needs.
    assert len({snapshot.tobytes() for snapshot in null_graph.snapshots}) == len(null_graph.snapshots)


def test_the_er_snapshots_are_symmetric_and_unweighted(config, null_graph):
    snapshots = null_graph.snapshots
    assert np.array_equal(snapshots, np.swapaxes(snapshots, -1, -2))
    assert np.array_equal(
        np.diagonal(snapshots, axis1=1, axis2=2), np.zeros(snapshots.shape[:2])
    )
    assert set(np.unique(snapshots).tolist()) <= {0.0, config.dynamic_events.edge_weight}


@pytest.mark.slow
def test_the_er_null_is_not_topologically_empty(config, null_graph):
    """The claim ``docs/method.md`` §4.2 rests on, kept as a measurement.

    At the density of ``cycle_lifecycle`` a random graph carries cycles; a test demanding
    ``(1,0,0)`` per snapshot would be demanding the impossible.  Asserting the *presence*
    of that structure keeps the rationale honest: if a future change made the null clean,
    this test fails and §4.2 has to be revisited rather than quietly left wrong.
    """
    topology = config.topology_check
    curves = [
        reference.betti_numbers_of_graph(
            snapshot,
            threshold=null_graph.ground_truth.edge_threshold,
            max_homology_dim=topology.max_homology_dim,
            max_nodes=topology.reference_max_nodes,
        )
        for snapshot in null_graph.snapshots[:10]
    ]
    assert any(betti[1] > 0 for betti in curves), "the ER null came out clean; see method.md 4.2"


# --- the false-positive rate of the criterion (docs/method.md 1.2a) ------------------


@pytest.mark.slow
def test_the_false_positive_rate_stays_under_the_measured_bound(config):
    """The rate is measured, not asserted to be zero.

    On the generator's own stream the rate is 1.5% at the configured size (3 misses in
    200 trials, 95% upper bound 4.3%); on the three calibration seeds specifically it is
    0 of 60.  The bound sits above the confidence bound rather than above that observed
    zero: sixty trials cannot establish a zero, and a threshold set from one would assert
    more than the measurement supports (``docs/method.md`` §1.2a).

    Deterministic given the seeds, so this cannot flake — a different result means the
    criterion, the generator or a dependency changed.
    """
    bound = config.null_models.max_false_positive_rate
    trials = misses = 0
    for seed in config.topology_check.calibration_seeds:
        for repeat in range(config.null_models.n_realizations):
            from tdadyn.core import rng_for
            from tdadyn.synthetic import make_uniform_cloud

            settings = config.null_models.uniform_cloud
            cloud = make_uniform_cloud(
                settings.n_points, settings.dim, repeat, rng_for(seed, "uniform_cloud", repeat)
            )
            trials += 1
            misses += measured_betti(cloud.points, config) != CLEAN
    rate = misses / trials
    assert rate <= bound, f"{misses}/{trials} = {rate:.3f} exceeds {bound}"
