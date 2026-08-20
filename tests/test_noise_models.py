"""Family 4: controlled noise over the reference datasets.

The level is the truth: each dataset records which mechanism ran at which level over which
base, and the tests check that the data match that claim.

The stability bound asserted here is :math:`d_B \\le 2\\,d_H`, **not** :math:`d_B \\le d_H`.
The factor two comes from the Rips filtration being parametrised by the *diameter* of a
simplex, so a displacement of each point by :math:`\\delta` moves a filtration value by up
to :math:`2\\delta` (``docs/method.md`` §4.1).  Measured on the circle over the three
calibration seeds, five of twelve combinations violate the bound without the factor, with
a ratio up to 1.31 — and none violate it with the factor.  Note that :math:`H_0` never
violates either form, so a test written only in dimension zero would pass while leaving
the wrong theorem in place.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tdadyn.core import dataset_equal, load_dataset, rng_for, save_dataset
from tdadyn.synthetic import (
    edge_count,
    make_jittered_cloud,
    make_jittered_network,
    make_noisy_snapshots,
    pairwise_distances,
    persistence_diagrams,
)

from tdadyn.synthetic import GAUSSIAN_KERNEL_RANGE, noise_stream

from conftest import build_dynamic_graph, build_static_cloud

STABILITY_FACTOR = 2.0
"""Rips is parametrised by diameter, so a delta-displacement moves filtration values by
2*delta.  See the module docstring and ``docs/method.md`` §4.1 for the measurement that
rejects the factor-one form."""

NOISY_BASE = "circle"
"""Base object for the noise tests: planar, cheap, and its H1 is the dimension where the
stability bound actually bites."""


def weighted_base(config, n_points):
    """A clean cloud together with its Gaussian-kernel network.

    Weight jitter applies to the *network* built from a cloud, not to the cloud, and the
    family-1 generators do not attach one — so it is built here through the same
    calibrated path family 1 uses (``docs/method.md`` §1.3).
    """
    from tdadyn.synthetic.network import gaussian_kernel, kernel_width

    base = build_static_cloud(NOISY_BASE, config, n_points=n_points, noise_sigma=0.0)
    distances = pairwise_distances(base.points)
    sigma = kernel_width(
        distances, median_fraction=config.network_from_cloud.kernel_sigma_median_fraction
    )
    return base, gaussian_kernel(distances, sigma=sigma)


BASE_ID = "circle_N200_radius1_sig0_coeff_field2_seed20260201"


def noisy_cloud(config, *, kind="point_jitter", level=0.05, repeat=0, n_points=200):
    """Build one noisy dataset over a clean base of the repository configuration.

    Returns ``(base, weights, noisy)``.  Which array the noisy dataset carries depends on
    the mechanism: family 4 stores the perturbed array alone (``docs/data.md`` §1).
    """
    base, weights = weighted_base(config, n_points)
    stream = rng_for(config.seed, noise_stream(kind, "circle"), repeat)
    if kind == "point_jitter":
        noisy = make_jittered_cloud(base.points, BASE_ID, level, repeat, stream)
    else:
        noisy = make_jittered_network(
            weights, BASE_ID, level, GAUSSIAN_KERNEL_RANGE, repeat, stream
        )
    return base, weights, noisy


def finite_bars(diagram):
    """Bars with a finite death.

    ``persim`` drops non-finite deaths itself and warns; doing it here makes the choice
    explicit.  In :math:`H_0` both diagrams carry exactly one infinite bar, which would
    pair with its counterpart and contribute nothing.
    """
    return diagram[np.isfinite(diagram[:, 1])]


def diagrams_of(points, config):
    return persistence_diagrams(
        pairwise_distances(points),
        max_homology_dim=1,
        coeff_field=config.topology_check.coeff_field,
    )


def hausdorff(left, right):
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return max(distances.min(axis=1).max(), distances.min(axis=0).max())


# --- the truth record ---------------------------------------------------------------


def test_the_truth_is_the_noise_level(config):
    *_, noisy = noisy_cloud(config)
    ground_truth = noisy.ground_truth
    assert ground_truth.family == "noisy"
    assert ground_truth.truth_kind == "noise_level"
    assert ground_truth.noise_level["kind"] == "point_jitter"
    assert ground_truth.noise_level["level"] == pytest.approx(0.05)
    assert ground_truth.noise_level["base"]
    assert ground_truth.expected_betti is None


def test_the_noise_level_keys_are_strings(config):
    # jsonify refuses non-string keys, so a dict that round-trips must use them.
    *_, noisy = noisy_cloud(config)
    assert all(isinstance(key, str) for key in noisy.ground_truth.noise_level)


def test_the_dataset_survives_a_round_trip(config, tmp_path):
    *_, noisy = noisy_cloud(config)
    path = save_dataset(noisy, tmp_path, float_precision=config.naming.float_precision)
    assert dataset_equal(load_dataset(path), noisy)


def test_the_repeat_index_reaches_the_directory_name(config):
    from tdadyn.core import make_dataset_id

    precision = config.naming.float_precision
    first = make_dataset_id(noisy_cloud(config, repeat=0)[2].ground_truth, float_precision=precision)
    second = make_dataset_id(noisy_cloud(config, repeat=1)[2].ground_truth, float_precision=precision)
    # Without the repeat in the name, n_repeats realisations overwrite one another and
    # format_dataset_id cannot catch it: the collision is between calls, not within one.
    assert first != second


def test_only_the_perturbed_array_reaches_disk(config, tmp_path):
    """The single layout rule of family 4 (``docs/data.md`` §1).

    A directory must never hold a perturbed array beside the source it no longer matches.
    Measured before the rule was applied: after point jitter at level 0.10 the carried
    network differed from the network of the stored points by up to 0.36, and a consumer
    who rebuilt it would have got something other than what was on disk, silently.
    """
    precision = config.naming.float_precision
    _, _, jittered_points = noisy_cloud(config, kind="point_jitter")
    _, _, jittered_network = noisy_cloud(config, kind="weight_jitter")

    cloud_dir = save_dataset(jittered_points, tmp_path / "a", float_precision=precision)
    network_dir = save_dataset(jittered_network, tmp_path / "b", float_precision=precision)
    written = lambda path: {file.name for file in path.glob("*.npy")}

    assert written(cloud_dir) == {"points.npy"}, "a jittered cloud must not carry a network"
    assert written(network_dir) == {"weights.npy"}, "a jittered network must not carry points"


def test_a_weights_only_directory_loads_as_a_network(config, tmp_path):
    from tdadyn.core import NetworkDataset

    _, _, jittered = noisy_cloud(config, kind="weight_jitter")
    path = save_dataset(jittered, tmp_path, float_precision=config.naming.float_precision)
    loaded = load_dataset(path)
    assert isinstance(loaded, NetworkDataset)
    assert dataset_equal(loaded, jittered)


def test_a_cloud_with_a_network_still_loads_as_a_cloud(fake_point_cloud, config, tmp_path):
    # weights.npy is a fallback marker, not a primary one: a point cloud may legitimately
    # carry a network beside its points, and that must not become ambiguous.
    path = save_dataset(fake_point_cloud, tmp_path, float_precision=config.naming.float_precision)
    assert {file.name for file in path.glob("*.npy")} == {"points.npy", "weights.npy"}
    assert isinstance(load_dataset(path), type(fake_point_cloud))


def test_a_weight_range_the_transform_cannot_produce_is_refused(config):
    # The range is a property of the transform, not a setting: a Gaussian kernel lies in
    # (0, 1], so base weights outside the declared range mean the wrong transform.
    _, weights = weighted_base(config, 100)
    with pytest.raises(ValueError, match="property of the transform"):
        make_jittered_network(weights, BASE_ID, 0.05, (0.5, 0.9), 0, rng_for(1, "probe"))


# --- level zero is the clean object -------------------------------------------------


@pytest.mark.parametrize("kind", ["point_jitter", "weight_jitter"])
def test_zero_level_reproduces_the_base_bitwise(config, kind):
    base, weights, noisy = noisy_cloud(config, kind=kind, level=0.0)
    if kind == "point_jitter":
        assert np.array_equal(noisy.points, base.points)
        # No network beside moved points: docs/data.md 1, family 4.
        assert noisy.weights is None
    else:
        assert np.array_equal(noisy.weights, weights)


def test_the_draw_happens_even_at_zero(config):
    # The stream must advance identically at every level, or "level 0" is a different
    # sample of the underlying object rather than the same one.
    stream = rng_for(config.seed, "draw_probe")
    before = stream.normal(size=3).copy()
    stream = rng_for(config.seed, "draw_probe")
    from tdadyn.synthetic import jitter_points

    jitter_points(np.zeros((5, 2)), sigma=0.0, rng=stream)
    after = stream.normal(size=3)
    assert not np.array_equal(before, after)


# --- the p+/p- fractions ------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["cycle_lifecycle", "geometric_loop"])
@pytest.mark.parametrize("level", [0.0, 0.05, 0.10])
def test_the_edge_fractions_are_exact_on_the_upper_triangle(config, scenario, level):
    # docs/method.md 5: counting the full matrix understates the fraction by half.
    # Both scenarios are exercised because their edge rules point opposite ways.
    base = build_dynamic_graph(scenario, config)
    truth = base.ground_truth
    rule = {"edge_rule": truth.edge_rule, "edge_threshold": truth.edge_threshold}
    snapshot = base.snapshots[0]
    n_present = edge_count(snapshot, **rule)
    n_absent = snapshot.shape[0] * (snapshot.shape[0] - 1) // 2 - n_present

    added = make_noisy_snapshots(
        base.snapshots, "base", "spurious_edges", level, truth.edge_rule,
        truth.edge_threshold, 0, rng_for(config.seed, "p_plus"),
    )
    dropped = make_noisy_snapshots(
        base.snapshots, "base", "missing_edges", level, truth.edge_rule,
        truth.edge_threshold, 0, rng_for(config.seed, "p_minus"),
    )
    assert edge_count(added.snapshots[0], **rule) - n_present == round(level * n_absent)
    assert n_present - edge_count(dropped.snapshots[0], **rule) == round(level * n_present)


@pytest.mark.parametrize("scenario", ["cycle_lifecycle", "geometric_loop"])
def test_noisy_snapshots_stay_symmetric(config, scenario):
    base = build_dynamic_graph(scenario, config)
    truth = base.ground_truth
    noisy = make_noisy_snapshots(
        base.snapshots, "base", "spurious_edges", 0.1, truth.edge_rule,
        truth.edge_threshold, 0, rng_for(config.seed, "sym"),
    )
    assert np.array_equal(noisy.snapshots, np.swapaxes(noisy.snapshots, -1, -2))


# --- reproducibility ----------------------------------------------------------------


def test_the_generator_is_deterministic(config):
    assert dataset_equal(noisy_cloud(config)[2], noisy_cloud(config)[2])


def test_the_global_seed_does_not_leak(config):
    expected = noisy_cloud(config)[2]
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(noisy_cloud(config)[2], expected)


def test_different_repeats_give_different_data(config):
    assert not dataset_equal(noisy_cloud(config, repeat=0)[2], noisy_cloud(config, repeat=1)[2])


# --- stability (docs/method.md 4.1) --------------------------------------------------


@pytest.mark.slow
def test_the_bottleneck_distance_grows_with_the_noise_level(config):
    from persim import bottleneck

    levels = config.noise.point_jitter_sigma
    base = build_static_cloud(NOISY_BASE, config, n_points=200, noise_sigma=0.0)
    clean = diagrams_of(base.points, config)
    means = []
    for level in levels:
        values = []
        for repeat in range(config.noise.n_repeats):
            *_, noisy = noisy_cloud(config, level=level, repeat=repeat)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # the explicit filter must make it moot
                values.append(
                    bottleneck(finite_bars(clean[1]), finite_bars(diagrams_of(noisy.points, config)[1]))
                )
        means.append(float(np.mean(values)))
    assert means[0] == 0.0
    assert all(later > earlier for earlier, later in zip(means, means[1:])), means


@pytest.mark.slow
@pytest.mark.parametrize("seed", [20260201, 7, 99])
def test_the_stability_theorem_holds_with_the_factor_two(config, seed):
    from persim import bottleneck

    base = build_static_cloud(NOISY_BASE, config, n_points=400, noise_sigma=0.0)
    clean = diagrams_of(base.points, config)
    for level in config.noise.point_jitter_sigma[1:]:
        noisy_points = make_jittered_cloud(
            base.points, "base", level, 0, rng_for(seed, "stability")
        ).points
        noisy = diagrams_of(noisy_points, config)
        separation = hausdorff(base.points, noisy_points)
        for dim in (0, 1):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                distance = bottleneck(finite_bars(clean[dim]), finite_bars(noisy[dim]))
            assert distance <= STABILITY_FACTOR * separation, (
                f"dim {dim}, level {level}: d_B={distance:.4f} against "
                f"{STABILITY_FACTOR}*d_H={STABILITY_FACTOR * separation:.4f}"
            )
