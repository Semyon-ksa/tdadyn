"""Family 1 planar generators: contracts, determinism, metadata.

Topology lives in ``test_static_betti.py``; everything here is fast and needs no ripser.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.synthetic import (
    dataset_equal,
    load_dataset,
    make_dataset_id,
    make_two_circles,
    rng_for,
    save_dataset,
)
from tdadyn.synthetic.schema import validate_ground_truth_dict

from conftest import STATIC_GENERATORS, build_static_cloud, build_static_cloud

# Transcribed from the table in docs/method.md §1.2, independently of what the
# generators declare — comparing the two is the point of one of the tests below.
# The Klein bottle row is the one for Z/2, which is the working field of the project.
EXPECTED_BETTI = {
    "circle": {0: 1, 1: 1, 2: 0},
    "two_circles": {0: 2, 1: 2, 2: 0},
    "figure_eight": {0: 1, 1: 2, 2: 0},
    "torus": {0: 1, 1: 2, 2: 1},
    "sphere": {0: 1, 1: 0, 2: 1},
    "klein_bottle": {0: 1, 1: 2, 2: 1},
    "hopf_link": {0: 2, 1: 2, 2: 0},
}

AMBIENT_DIMENSION = {
    "circle": 2,
    "two_circles": 2,
    "figure_eight": 2,
    "torus": 3,
    "sphere": 3,
    "klein_bottle": 4,
    "hopf_link": 3,
}


@pytest.fixture(params=STATIC_GENERATORS)
def static_name(request):
    return request.param


def test_same_seed_gives_bitwise_identical_points(static_name, config):
    first = build_static_cloud(static_name, config)
    second = build_static_cloud(static_name, config)
    assert dataset_equal(first, second)


def test_global_numpy_seed_does_not_change_the_cloud(static_name, config):
    expected = build_static_cloud(static_name, config)
    np.random.seed(4242)
    np.random.normal(size=1000)
    assert dataset_equal(build_static_cloud(static_name, config), expected)


def test_shape_and_dtype_follow_the_axis_table(static_name, config):
    clouds = config.static_clouds
    dataset = build_static_cloud(static_name, config)
    expected_count = clouds.n_points_by_generator.get(static_name, clouds.n_points)
    assert dataset.points.shape == (expected_count, AMBIENT_DIMENSION[static_name])
    assert dataset.points.dtype == np.float64


def test_declared_expectations_match_the_table_in_the_docs(static_name, config):
    """The generator's ``expected_betti`` must equal the table it was transcribed from.

    Catches the two halves of the truth drifting apart: what a dataset claims about
    itself and what the topological test checks against.
    """
    dataset = build_static_cloud(static_name, config)
    assert dataset.ground_truth.expected_betti == EXPECTED_BETTI[static_name]
    assert dataset.ground_truth.coeff_field == config.topology_check.coeff_field
    assert dataset.ground_truth.family == "static_cloud"


def test_seed_is_taken_from_the_stream(static_name, config):
    dataset = build_static_cloud(static_name, config)
    assert dataset.ground_truth.seed == config.seed


def test_ground_truth_passes_schema_validation(static_name, config):
    from tdadyn.synthetic.schema import ground_truth_to_json_dict

    dataset = build_static_cloud(static_name, config)
    dataset_id = make_dataset_id(
        dataset.ground_truth, float_precision=config.naming.float_precision
    )
    validate_ground_truth_dict(ground_truth_to_json_dict(dataset.ground_truth, dataset_id))


def test_round_trip_through_disk(static_name, config, tmp_path):
    dataset = build_static_cloud(static_name, config)
    path = save_dataset(
        dataset, tmp_path / "static_cloud", float_precision=config.naming.float_precision
    )
    restored = load_dataset(path, float_precision=config.naming.float_precision)
    assert dataset_equal(restored, dataset)


def test_every_parameter_reaches_the_directory_name(static_name, config):
    dataset = build_static_cloud(static_name, config)
    dataset_id = make_dataset_id(
        dataset.ground_truth, float_precision=config.naming.float_precision
    )
    assert dataset_id.startswith(static_name)
    assert dataset_id.endswith(f"seed{config.seed}")


def test_klein_bottle_expectations_depend_on_the_field(config):
    """The one object whose truth changes with the field (docs/method.md §1.2)."""
    from tdadyn.synthetic import klein_bottle_betti

    assert klein_bottle_betti(2) == {0: 1, 1: 2, 2: 1}
    assert klein_bottle_betti(3) == {0: 1, 1: 1, 2: 0}
    # Different answers, so the two must not be able to share a directory: the field is
    # part of the parameters and therefore of the name.
    dataset = build_static_cloud("klein_bottle", config)
    assert "coeff_field" in dataset.ground_truth.params


def test_hopf_link_expects_two_components(config):
    """docs/method.md §1.2: expecting beta_0 = 1 for a link is an error."""
    dataset = build_static_cloud("hopf_link", config)
    assert dataset.ground_truth.expected_betti[0] == 2
    assert "сцеп" in dataset.ground_truth.notes


def test_noise_level_changes_the_cloud_but_not_the_underlying_sample(config):
    """Sampling comes first, so the curve is identical across noise levels."""
    clean = build_static_cloud("circle", config, noise_sigma=0.0)
    noisy = build_static_cloud("circle", config, noise_sigma=config.static_clouds.noise_sigma)
    assert not np.array_equal(clean.points, noisy.points)
    displacement = np.linalg.norm(noisy.points - clean.points, axis=1)
    # Displacements are Rayleigh-distributed with the configured sigma; a loose bound is
    # enough to show the noise is the configured one and not, say, ten times it.
    assert displacement.mean() < 5 * config.static_clouds.noise_sigma


def test_circle_points_lie_on_the_circle_when_clean(config):
    dataset = build_static_cloud("circle", config, noise_sigma=0.0)
    radii = np.linalg.norm(dataset.points, axis=1)
    assert np.allclose(radii, config.static_clouds.circle.radius)


def test_two_circles_components_stay_apart(config):
    """A structural check on beta_0 = 2 that needs no persistent homology."""
    clouds = config.static_clouds
    dataset = build_static_cloud("two_circles", config, noise_sigma=0.0)
    n_first = clouds.n_points // 2
    first, second = dataset.points[:n_first], dataset.points[n_first:]
    gap = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=-1).min()
    assert gap >= clouds.two_circles.separation - 2 * clouds.two_circles.radius


def test_overlapping_circles_are_refused(config):
    clouds = config.static_clouds
    with pytest.raises(ValueError, match="beta_0 is not 2"):
        make_two_circles(
            10,
            clouds.two_circles.radius,
            2 * clouds.two_circles.radius,
            0.0,
            config.topology_check.coeff_field,
            rng_for(config.seed, "two_circles"),
        )


def test_figure_eight_records_its_non_uniform_density(config):
    """docs/method.md §1.1 allows the non-uniformity but requires it to be recorded."""
    dataset = build_static_cloud("figure_eight", config)
    assert "самопересеч" in dataset.ground_truth.notes


def test_figure_eight_passes_through_its_crossing(config):
    dataset = build_static_cloud("figure_eight", config, noise_sigma=0.0)
    assert np.linalg.norm(dataset.points, axis=1).min() < 0.1 * config.static_clouds.figure_eight.scale


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_points": 0}, "n_points must be >= 1"),
        ({"noise_sigma": -0.1}, "noise_sigma must be >= 0"),
        ({"coeff_field": 1}, "coeff_field must be an int >= 2"),
    ],
)
def test_invalid_arguments_are_refused(config, kwargs, message):
    from tdadyn.synthetic import make_circle

    arguments = {
        "n_points": 10,
        "radius": config.static_clouds.circle.radius,
        "noise_sigma": 0.0,
        "coeff_field": config.topology_check.coeff_field,
        **kwargs,
    }
    with pytest.raises((ValueError, TypeError), match=message):
        make_circle(rng=rng_for(config.seed, "circle"), **arguments)
