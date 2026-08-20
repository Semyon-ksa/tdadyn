"""Topological checks for family 1, split into the two tests of ``docs/validation.md`` §1.1.

The two must not be conflated:

* **Test 1** takes the full cloud, runs path A only, and compares with the expectation
  table.  Path B cannot participate — it is limited to ``reference_max_nodes``, and on a
  small sample the gaps between neighbours are random, so a scale giving the table answer
  may not exist at all.  Demanding the table from a small cloud would mean hunting for a
  scale until it converges, which is fitting the truth.
* **Test 2** takes a subsample and asks a different question: do the two implementations
  agree about the *same* complex?  Whatever they return, they must return it together.
  The table answer will not appear at most scales, and that is not an error.

Expectations below are transcribed from the table in ``docs/method.md`` §1.2.  Changing
them is allowed only together with a mathematical justification added to that document
in the same commit (``docs/validation.md`` §2); if a test fails, the cause is the number of points,
the noise level, the uniformity of the sampling or the parametrisation — never the
significance threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import adjacency_from_snapshot
from tdadyn.reference import betti_numbers_of_graph
from tdadyn.synthetic import (
    betti_at_scale,
    cross_check_scales,
    pairwise_distances,
    persistence_diagrams,
    significant_betti,
)

from conftest import STATIC_GENERATORS, build_static_cloud

EXPECTED_BETTI = {
    "circle": {0: 1, 1: 1, 2: 0},
    "two_circles": {0: 2, 1: 2, 2: 0},
    "figure_eight": {0: 1, 1: 2, 2: 0},
    "torus": {0: 1, 1: 2, 2: 1},
    "sphere": {0: 1, 1: 0, 2: 1},
    "klein_bottle": {0: 1, 1: 2, 2: 1},
    "hopf_link": {0: 2, 1: 2, 2: 0},
}


# --- Test 1: the full cloud against the table, path A only -----------------------


def measure(dataset, config, *, coeff_field=None):
    """Run path A with the calibrated criterion, every parameter from the config."""
    topology = config.topology_check
    return significant_betti(
        pairwise_distances(dataset.points),
        max_homology_dim=topology.max_homology_dim,
        coeff_field=topology.coeff_field if coeff_field is None else coeff_field,
        criterion=topology.criterion,
        significance_ratio=topology.significance_ratio,
        gap_noise_floor_ratio=topology.gap_noise_floor_ratio,
        gap_min_ratio=topology.gap_min_ratio,
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", STATIC_GENERATORS)
@pytest.mark.parametrize("clean", [True, False], ids=["sigma-zero", "sigma-config"])
def test_significant_bars_match_the_expectation_table(name, clean, config):
    sigma = 0.0 if clean else config.static_clouds.noise_sigma
    dataset = build_static_cloud(name, config, noise_sigma=sigma)
    assert measure(dataset, config) == EXPECTED_BETTI[name]


@pytest.mark.slow
@pytest.mark.parametrize("name", STATIC_GENERATORS)
def test_the_dataset_declares_what_the_pipeline_finds(name, config):
    """The truth stored with the dataset and the truth the pipeline recovers agree."""
    dataset = build_static_cloud(name, config)
    assert measure(dataset, config) == dataset.ground_truth.expected_betti


@pytest.mark.slow
def test_klein_bottle_over_z2_needs_both_paths(config):
    """Over Z/2 the torsion loop is visible: (1, 2, 1), confirmed by paths A and B.

    ``docs/validation.md`` §3: at p = 2 the double check applies in full — path A here,
    path B through the cross-check test below.
    """
    dataset = build_static_cloud("klein_bottle", config)
    assert measure(dataset, config, coeff_field=2) == {0: 1, 1: 2, 2: 1}


@pytest.mark.slow
def test_klein_bottle_over_z3_is_a_consistency_check_only(config):
    """At p = 3 the answer is (1, 1, 0) — and only path A can say so.

    This is **not** an independent confirmation of the Betti numbers: the reference is
    restricted to Z/2 by design (``docs/method.md`` §1.2), so at p = 3 there is nothing
    to cross-check against.  What the test does confirm is that the field parameter
    reaches the computation at all: were ``coeff_field`` read from the config and then
    dropped, both fields would return the Z/2 answer and the difference below would
    vanish.  Rational coefficients are unavailable in ripser, so Q is checked through
    p = 3.
    """
    dataset = build_static_cloud("klein_bottle", config)
    over_two = measure(dataset, config, coeff_field=2)
    over_three = measure(dataset, config, coeff_field=3)
    assert over_three == {0: 1, 1: 1, 2: 0}
    assert over_two != over_three, "the field parameter never reached the computation"


@pytest.mark.slow
def test_hopf_link_keeps_its_components_apart(config):
    """docs/method.md §1.2: two linked circles have the homology of two disjoint ones.

    The dataset is a geometric stress test — the pipeline must not glue two components
    that pass close to one another.  Expecting beta_0 = 1 here is an error.
    """
    dataset = build_static_cloud("hopf_link", config)
    assert measure(dataset, config)[0] == 2


@pytest.mark.slow
def test_the_old_criterion_still_fails_where_it_was_measured_to(config):
    """Guards the reason the criterion was replaced (docs/method.md §1.2a).

    The diameter-normalised rule loses the cavity of the torus.  If this ever starts
    passing, the measurement behind the replacement no longer holds and the decision
    must be revisited rather than quietly inherited.
    """
    topology = config.topology_check
    dataset = build_static_cloud("torus", config)
    old = significant_betti(
        pairwise_distances(dataset.points),
        max_homology_dim=topology.max_homology_dim,
        coeff_field=topology.coeff_field,
        criterion="diam_ratio",
        significance_ratio=topology.significance_ratio,
    )
    assert old[2] == 0, "the old rule found the cavity; method.md §1.2a needs revisiting"
    assert measure(dataset, config)[2] == 1


# --- Test 2: the two implementations against each other --------------------------


@pytest.mark.parametrize("name", STATIC_GENERATORS)
def test_paths_a_and_b_agree_on_the_same_complex(name, config):
    """Agreement of the implementations, not agreement with the table.

    Both paths measure one complex: ripser's barcode read at a scale, and the reference
    Betti numbers of the flag complex of the proximity graph at that same scale.  The
    scales are midpoints between adjacent distinct distances — at a distance itself the
    simplex enters the complex and the comparison would turn into float equality against
    ripser's float32 barcode.

    The scales are fixed by :func:`cross_check_scales` and never chosen by looking at the
    Betti numbers that come out; doing so would be fitting the truth.
    """
    topology = config.topology_check
    dataset = build_static_cloud(name, config)

    positions = np.linspace(
        0, len(dataset.points) - 1, topology.cross_check_n_points
    ).round().astype(int)
    distances = pairwise_distances(dataset.points[positions])

    diagrams = persistence_diagrams(
        distances,
        max_homology_dim=topology.max_homology_dim,
        coeff_field=topology.coeff_field,
    )
    scales = cross_check_scales(distances, topology.cross_check_n_scales)
    assert len(scales) == topology.cross_check_n_scales

    for scale in scales:
        path_a = betti_at_scale(diagrams, scale)
        # The proximity graph, through the project's one implementation of the edge
        # rule: at scale eps an edge is d <= eps, which is `distance_within`.  Writing
        # the comparison here by hand would be a third spelling of the same rule.
        adjacency = adjacency_from_snapshot(
            distances, edge_rule="distance_within", edge_threshold=scale
        )
        path_b = betti_numbers_of_graph(
            adjacency,
            max_homology_dim=topology.max_homology_dim,
            max_nodes=topology.reference_max_nodes,
        )
        assert path_a == path_b, f"paths disagree at scale {scale}"


def test_cross_check_scales_avoid_the_distances_themselves(config):
    dataset = build_static_cloud("circle", config)
    positions = np.linspace(
        0, config.static_clouds.n_points - 1, config.topology_check.cross_check_n_points
    ).round().astype(int)
    distances = pairwise_distances(dataset.points[positions])

    scales = cross_check_scales(distances, config.topology_check.cross_check_n_scales)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    assert np.isin(scales, upper).sum() == 0
    assert np.all(np.diff(scales) > 0)


def test_cross_check_subsample_fits_the_reference_limit(config):
    assert config.topology_check.cross_check_n_points <= config.topology_check.reference_max_nodes


# --- the criterion itself, on hand-made spectra ----------------------------------


def diagram_of(persistences):
    return np.array([[0.0, value] for value in persistences], dtype=float)


def count(persistences, *, diameter, floor=0.07, gap=1.55):
    from tdadyn.synthetic.persistence import _count_by_gap

    return _count_by_gap(diagram_of(persistences), diameter, floor, gap)


def test_two_nearly_equal_features_are_both_found():
    """The case that killed the rule without a step into the dust.

    Two disjoint circles leave two nearly equal bars above the floor; there is no gap
    *between* them, so a rule looking only at gaps among survivors reports nothing.
    """
    assert count([0.83, 0.79] + [0.02] * 50, diameter=5.0) == 2


def test_a_smooth_staircase_of_noise_yields_nothing():
    staircase = list(np.linspace(0.35, 0.001, 300))
    assert count(staircase, diameter=2.13) == 0


def test_a_single_dominant_feature_is_found():
    assert count([1.7, 0.09, 0.08, 0.07], diameter=5.6) == 1


def test_an_empty_diagram_yields_nothing():
    from tdadyn.synthetic.persistence import _count_by_gap

    assert _count_by_gap(np.empty((0, 2)), 1.0, 0.07, 1.55) == 0


def test_infinite_bars_always_count():
    from tdadyn.synthetic.persistence import _count_by_gap

    diagram = np.array([[0.0, np.inf], [0.0, 0.9]] + [[0.0, 0.01]] * 30)
    assert _count_by_gap(diagram, 5.0, 0.07, 1.55) == 2


def test_the_criterion_needs_its_parameters(config):
    distances = pairwise_distances(build_static_cloud("circle", config).points)
    with pytest.raises(ValueError, match="needs gap_noise_floor_ratio"):
        significant_betti(
            distances, max_homology_dim=1, coeff_field=2, criterion="gap"
        )
    with pytest.raises(ValueError, match="needs significance_ratio"):
        significant_betti(
            distances, max_homology_dim=1, coeff_field=2, criterion="diam_ratio"
        )
    with pytest.raises(ValueError, match="unknown criterion"):
        significant_betti(
            distances, max_homology_dim=1, coeff_field=2, criterion="elbow"
        )


def test_betti_at_scale_uses_a_half_open_interval():
    """At eps = death the class is already dead (docs/validation.md §1.1)."""
    diagrams = {0: np.array([[0.0, 1.0], [0.0, np.inf]]), 1: np.array([[0.5, 2.0]])}
    assert betti_at_scale(diagrams, 0.999) == {0: 2, 1: 1}
    assert betti_at_scale(diagrams, 1.0) == {0: 1, 1: 1}
    assert betti_at_scale(diagrams, 2.0) == {0: 1, 1: 0}
