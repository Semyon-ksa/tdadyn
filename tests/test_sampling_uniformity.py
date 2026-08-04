"""Sampling must be uniform by measure on the manifold, not by parameter.

``docs/validation.md`` §3 asks for these as tests of their own, because a naive
implementation looks entirely plausible and silently eats the :math:`\\beta_2` bar.  Each
test therefore comes with its negative control: the deliberately wrong sampling mode must
**fail** the same check, otherwise the test does not demonstrate what it claims to.

No persistent homology here — these are fast and run on every ``pytest -q``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from conftest import build_static_cloud

# A p-value this small means the sample contradicts the claimed density; the same
# threshold is used for the positive and the negative direction of every test.
ALPHA = 0.01


def torus_angles(points, R, r):
    """Recover the tube angle phi from points on the torus."""
    radial = np.hypot(points[:, 0], points[:, 1])
    return np.arctan2(points[:, 2] / r, (radial - R) / r) % (2 * np.pi)


def torus_cdf(R, r):
    """Exact CDF of the area-uniform tube angle, from the density (R + r cos phi)."""

    def cdf(phi):
        return (R * phi + r * np.sin(phi)) / (2 * np.pi * R)

    return cdf


def klein_folded_angles(points, a, b):
    """Recover |v| in [0, pi] from points on the Klein bottle.

    Only ``cos v`` is directly recoverable: the sign of ``sin v`` is entangled with the
    branch of ``u/2``.  The density is symmetric under ``v -> -v``, so folding to
    ``[0, pi]`` loses nothing the test needs.
    """
    radial = np.hypot(points[:, 0], points[:, 1])
    return np.arccos(np.clip((radial - a) / b, -1.0, 1.0))


def klein_folded_cdf(a, b, n_grid=200_001):
    """Numeric CDF of the folded angle, from the area element of ``docs/method.md`` §1.1."""
    grid = np.linspace(0.0, np.pi, n_grid)
    density = np.sqrt((a + b * np.cos(grid)) ** 2 + (b * np.sin(grid)) ** 2 / 4)
    cumulative = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2 * np.diff(grid))])
    cumulative /= cumulative[-1]

    def cdf(value):
        return np.interp(value, grid, cumulative)

    return cdf


# --- torus -----------------------------------------------------------------------


def test_torus_rejection_sampling_is_uniform_by_area(config):
    """The tube angle must follow the density proportional to (R + r cos phi)."""
    torus = config.static_clouds.torus
    dataset = build_static_cloud("torus", config, noise_sigma=0.0, sampling="rejection")
    phi = torus_angles(dataset.points, torus.R, torus.r)
    result = stats.kstest(phi, torus_cdf(torus.R, torus.r))
    assert result.pvalue > ALPHA, f"rejection sampling failed the KS test: {result}"


def test_naive_torus_sampling_fails_the_same_check(config):
    """The negative control: without it the test above proves nothing.

    Uniform angles oversample the inner rim, and the check must notice.
    """
    torus = config.static_clouds.torus
    dataset = build_static_cloud("torus", config, noise_sigma=0.0, sampling="naive")
    phi = torus_angles(dataset.points, torus.R, torus.r)
    result = stats.kstest(phi, torus_cdf(torus.R, torus.r))
    assert result.pvalue < ALPHA, f"naive sampling passed the KS test: {result}"


def test_naive_torus_records_that_it_is_unfit(config):
    dataset = build_static_cloud("torus", config, sampling="naive")
    assert "naive" in dataset.ground_truth.notes
    assert dataset.ground_truth.params["sampling"] == "naive"


# --- Klein bottle ----------------------------------------------------------------


def test_klein_rejection_sampling_is_uniform_by_area(config):
    """Same mistake as on the torus, and just as invisible (docs/validation.md §3)."""
    klein = config.static_clouds.klein_bottle
    dataset = build_static_cloud("klein_bottle", config, noise_sigma=0.0, sampling="rejection")
    folded = klein_folded_angles(dataset.points, klein.a, klein.b)
    result = stats.kstest(folded, klein_folded_cdf(klein.a, klein.b))
    assert result.pvalue > ALPHA, f"rejection sampling failed the KS test: {result}"


def test_naive_klein_sampling_fails_the_same_check(config):
    klein = config.static_clouds.klein_bottle
    dataset = build_static_cloud("klein_bottle", config, noise_sigma=0.0, sampling="naive")
    folded = klein_folded_angles(dataset.points, klein.a, klein.b)
    result = stats.kstest(folded, klein_folded_cdf(klein.a, klein.b))
    assert result.pvalue < ALPHA, f"naive sampling passed the KS test: {result}"


# --- sphere ----------------------------------------------------------------------


def test_sphere_muller_sampling_has_no_pole_clustering(config):
    """Archimedes: for a uniform sphere the height is uniform on [-radius, radius]."""
    radius = config.static_clouds.sphere.radius
    dataset = build_static_cloud("sphere", config, noise_sigma=0.0, sampling="muller")
    height = dataset.points[:, 2]
    result = stats.kstest(height, stats.uniform(loc=-radius, scale=2 * radius).cdf)
    assert result.pvalue > ALPHA, f"Muller sampling failed the KS test: {result}"


def test_naive_angles_cluster_at_the_poles(config):
    radius = config.static_clouds.sphere.radius
    dataset = build_static_cloud("sphere", config, noise_sigma=0.0, sampling="naive_angles")
    height = dataset.points[:, 2]
    result = stats.kstest(height, stats.uniform(loc=-radius, scale=2 * radius).cdf)
    assert result.pvalue < ALPHA, f"naive angle sampling passed the KS test: {result}"


@pytest.mark.parametrize("sampling", ["muller", "naive_angles"])
def test_sphere_points_lie_on_the_sphere(config, sampling):
    radius = config.static_clouds.sphere.radius
    dataset = build_static_cloud("sphere", config, noise_sigma=0.0, sampling=sampling)
    assert np.allclose(np.linalg.norm(dataset.points, axis=1), radius)


# --- the embedding itself --------------------------------------------------------


def test_klein_embedding_factors_through_the_identification(config):
    """(u, v) ~ (u + 2*pi, -v) is what makes this a Klein bottle and not a torus."""
    klein = config.static_clouds.klein_bottle
    a, b = klein.a, klein.b

    def embed(u, v):
        tube = a + b * np.cos(v)
        return np.column_stack(
            [
                tube * np.cos(u),
                tube * np.sin(u),
                b * np.sin(v) * np.cos(u / 2),
                b * np.sin(v) * np.sin(u / 2),
            ]
        )

    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 2 * np.pi, 2000)
    v = rng.uniform(0.0, 2 * np.pi, 2000)
    assert np.abs(embed(u, v) - embed(u + 2 * np.pi, -v)).max() < 1e-12


def test_torus_points_lie_on_the_torus(config):
    torus = config.static_clouds.torus
    dataset = build_static_cloud("torus", config, noise_sigma=0.0)
    radial = np.hypot(dataset.points[:, 0], dataset.points[:, 1])
    residual = (radial - torus.R) ** 2 + dataset.points[:, 2] ** 2
    assert np.allclose(residual, torus.r**2)
