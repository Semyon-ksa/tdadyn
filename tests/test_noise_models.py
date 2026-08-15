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

from tdadyn.synthetic import (
    dataset_equal,
    edge_count,
    load_dataset,
    make_noisy_cloud,
    make_noisy_snapshots,
    pairwise_distances,
    persistence_diagrams,
    rng_for,
    save_dataset,
)

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


def noisy_cloud(config, *, kind="point_jitter", level=0.05, repeat=0, n_points=200):
    """Build one noisy cloud over a clean base of the repository configuration."""
    base, weights = weighted_base(config, n_points)
    return base, weights, make_noisy_cloud(
        base.points,
        weights,
        "circle_N200_radius1_sig0_coeff_field2_seed20260201",
        kind,
        level,
        (0.0, 1.0),
        repeat,
        rng_for(config.seed, f"noise_{kind}", repeat),
    )


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
    from tdadyn.synthetic import make_dataset_id

    precision = config.naming.float_precision
    first = make_dataset_id(noisy_cloud(config, repeat=0)[2].ground_truth, float_precision=precision)
    second = make_dataset_id(noisy_cloud(config, repeat=1)[2].ground_truth, float_precision=precision)
    # Without the repeat in the name, n_repeats realisations overwrite one another and
    # format_dataset_id cannot catch it: the collision is between calls, not within one.
    assert first != second


# --- level zero is the clean object -------------------------------------------------


@pytest.mark.parametrize("kind", ["point_jitter", "weight_jitter"])
def test_zero_level_reproduces_the_base_bitwise(config, kind):
    base, weights, noisy = noisy_cloud(config, kind=kind, level=0.0)
    assert np.array_equal(noisy.points, base.points)
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
        noisy_points = make_noisy_cloud(
            base.points, None, "base", "point_jitter", level, (0.0, 1.0), 0,
            rng_for(seed, "stability"),
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
