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

from tdadyn.synthetic import (
    betti_at_scale,
    cross_check_scales,
    pairwise_distances,
    persistence_diagrams,
    significant_betti,
)
from tdadyn.synthetic.reference import betti_numbers_of_graph

from conftest import PLANAR_GENERATORS, build_planar_cloud

EXPECTED_BETTI = {
    "circle": {0: 1, 1: 1, 2: 0},
    "two_circles": {0: 2, 1: 2, 2: 0},
    "figure_eight": {0: 1, 1: 2, 2: 0},
}


# --- Test 1: the full cloud against the table, path A only -----------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", PLANAR_GENERATORS)
@pytest.mark.parametrize("clean", [True, False], ids=["sigma-zero", "sigma-config"])
def test_significant_bars_match_the_expectation_table(name, clean, config):
    topology = config.topology_check
    sigma = 0.0 if clean else config.static_clouds.noise_sigma
    dataset = build_planar_cloud(name, config, noise_sigma=sigma)

    betti = significant_betti(
        pairwise_distances(dataset.points),
        max_homology_dim=topology.max_homology_dim,
        coeff_field=topology.coeff_field,
        significance_ratio=topology.significance_ratio,
    )
    assert betti == EXPECTED_BETTI[name]


@pytest.mark.slow
@pytest.mark.parametrize("name", PLANAR_GENERATORS)
def test_the_dataset_declares_what_the_pipeline_finds(name, config):
    """The truth stored with the dataset and the truth the pipeline recovers agree."""
    topology = config.topology_check
    dataset = build_planar_cloud(name, config)
    betti = significant_betti(
        pairwise_distances(dataset.points),
        max_homology_dim=topology.max_homology_dim,
        coeff_field=topology.coeff_field,
        significance_ratio=topology.significance_ratio,
    )
    assert betti == dataset.ground_truth.expected_betti


# --- Test 2: the two implementations against each other --------------------------


@pytest.mark.parametrize("name", PLANAR_GENERATORS)
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
    dataset = build_planar_cloud(name, config)

    positions = np.linspace(
        0, config.static_clouds.n_points - 1, topology.cross_check_n_points
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
        # The proximity graph as a 0/1 matrix; the reference takes an edge where the
        # weight is strictly above the threshold, so 0.0 selects exactly the ones.
        adjacency = (distances <= scale).astype(np.float64)
        np.fill_diagonal(adjacency, 0.0)
        path_b = betti_numbers_of_graph(
            adjacency,
            threshold=0.0,
            max_homology_dim=topology.max_homology_dim,
            max_nodes=topology.reference_max_nodes,
        )
        assert path_a == path_b, f"paths disagree at scale {scale}"


def test_cross_check_scales_avoid_the_distances_themselves(config):
    dataset = build_planar_cloud("circle", config)
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


def test_betti_at_scale_uses_a_half_open_interval():
    """At eps = death the class is already dead (docs/validation.md §1.1)."""
    diagrams = {0: np.array([[0.0, 1.0], [0.0, np.inf]]), 1: np.array([[0.5, 2.0]])}
    assert betti_at_scale(diagrams, 0.999) == {0: 2, 1: 1}
    assert betti_at_scale(diagrams, 1.0) == {0: 1, 1: 1}
    assert betti_at_scale(diagrams, 2.0) == {0: 1, 1: 0}
