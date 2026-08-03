"""Family 1: point clouds sampled from manifolds whose topology is known analytically.

The parametrisations follow ``docs/method.md`` §1.1 and the expected Betti numbers are
transcribed from the table in §1.2.  Those numbers are an **input**: if a test does not
converge, the fault is in the generator, the sampling or the generation parameters — never
in the table (``docs/validation.md`` §1).

This module covers the planar curves of session S3; the surfaces follow in S4.
"""

from __future__ import annotations

import numpy as np

from .contracts import FORMAT_VERSION, GroundTruth, PointCloudDataset
from .naming import register_name_order
from .rng import base_seed_of

CIRCLE_BETTI = {0: 1, 1: 1, 2: 0}
TWO_CIRCLES_BETTI = {0: 2, 1: 2, 2: 0}
FIGURE_EIGHT_BETTI = {0: 1, 1: 2, 2: 0}
"""Expected Betti numbers, from the table in ``docs/method.md`` §1.2.

All three dimensions are stated because ``docs/task.md`` fixes the maximum homology
dimension of the project at 2; a dimension whose answer is zero is written out, since a
missing key cannot be told from a forgotten one.
"""

register_name_order("circle", ("n_points", "radius", "noise_sigma", "coeff_field"))
register_name_order(
    "two_circles", ("n_points", "radius", "separation", "noise_sigma", "coeff_field")
)
register_name_order("figure_eight", ("n_points", "scale", "noise_sigma", "coeff_field"))


def make_circle(
    n_points: int,
    radius: float,
    noise_sigma: float,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample a circle :math:`S^1` in the plane.

    Parameters
    ----------
    n_points : int
        Number of points.
    radius : float
        Radius of the circle.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from; the only source of randomness.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 2)`` with the ground truth :math:`(1, 1, 0)`.

    Notes
    -----
    ``docs/method.md`` §1.1: :math:`\\theta \\sim U(0, 2\\pi)` and
    :math:`(r\\cos\\theta, r\\sin\\theta)`.  Uniformity in :math:`\\theta` *is* uniformity
    by arc length here, so no rejection step is needed — unlike the torus in S4.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("radius", radius)

    points = _circle_points(n_points, radius, rng)
    points = _add_ambient_noise(points, noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "radius": float(radius),
        "noise_sigma": float(noise_sigma),
        "coeff_field": int(coeff_field),
    }
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("circle", params, CIRCLE_BETTI, rng, notes=""),
    )


def make_two_circles(
    n_points: int,
    radius: float,
    separation: float,
    noise_sigma: float,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample two disjoint circles of equal radius.

    Parameters
    ----------
    n_points : int
        Total number of points, split ``n_points // 2`` to the first circle and the
        remainder to the second, so that an odd count stays deterministic.
    radius : float
        Radius of both circles.
    separation : float
        Distance between the centres; must exceed ``2 * radius`` or the components meet
        and :math:`\\beta_0` is not 2.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 2)`` with the ground truth :math:`(2, 2, 0)`.

    Notes
    -----
    ``docs/method.md`` §1.1: the separation sets how long the second component lives in
    :math:`H_0`, which is what makes this the test for :math:`\\beta_0`.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("radius", radius)
    _check_positive("separation", separation)
    if separation <= 2 * radius:
        raise ValueError(
            f"separation ({separation}) must exceed 2 * radius ({2 * radius}), "
            "otherwise the two circles touch and beta_0 is not 2"
        )

    n_first = n_points // 2
    first = _circle_points(n_first, radius, rng)
    second = _circle_points(n_points - n_first, radius, rng) + np.array([separation, 0.0])
    points = _add_ambient_noise(np.vstack([first, second]), noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "radius": float(radius),
        "separation": float(separation),
        "noise_sigma": float(noise_sigma),
        "coeff_field": int(coeff_field),
    }
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("two_circles", params, TWO_CIRCLES_BETTI, rng, notes=""),
    )


def make_figure_eight(
    n_points: int,
    scale: float,
    noise_sigma: float,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample a figure eight — the wedge of two circles — as a lemniscate.

    Parameters
    ----------
    n_points : int
        Number of points.
    scale : float
        Overall scale factor of the curve.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 2)`` with the ground truth :math:`(1, 2, 0)`.

    Notes
    -----
    ``docs/method.md`` §1.1: :math:`x = \\cos t`, :math:`y = \\sin t \\cos t` with
    :math:`t \\sim U(0, 2\\pi)`.  Uniformity in :math:`t` is **not** uniformity by arc
    length — the curve moves fastest near the self-intersection, so the density drops
    there.  The docs allow this but require it to be recorded, so it goes into
    ``GroundTruth.notes``.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("scale", scale)

    parameter = rng.uniform(0.0, 2 * np.pi, n_points)
    points = scale * np.column_stack(
        [np.cos(parameter), np.sin(parameter) * np.cos(parameter)]
    )
    points = _add_ambient_noise(points, noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "scale": float(scale),
        "noise_sigma": float(noise_sigma),
        "coeff_field": int(coeff_field),
    }
    notes = (
        "Равномерность по параметру t не даёт равномерности по длине дуги: "
        "плотность падает у точки самопересечения (docs/method.md, раздел 1.1)."
    )
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("figure_eight", params, FIGURE_EIGHT_BETTI, rng, notes=notes),
    )


def _circle_points(n_points: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a circle centred at the origin, uniformly by arc length."""
    angle = rng.uniform(0.0, 2 * np.pi, n_points)
    return radius * np.column_stack([np.cos(angle), np.sin(angle)])


def _add_ambient_noise(
    points: np.ndarray, noise_sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add isotropic Gaussian noise in the ambient space.

    The draw happens even at ``noise_sigma == 0``, where it contributes zeros.  Sampling
    the curve therefore comes first and is bitwise identical across noise levels, which
    is what lets a noisy dataset be compared with its clean counterpart point by point.
    """
    return points + rng.normal(0.0, noise_sigma, points.shape)


def _ground_truth(
    generator: str,
    params: dict,
    expected_betti: dict[int, int],
    rng: np.random.Generator,
    *,
    notes: str,
) -> GroundTruth:
    return GroundTruth(
        format_version=FORMAT_VERSION,
        family="static_cloud",
        generator=generator,
        params=params,
        seed=base_seed_of(rng),
        coeff_field=params["coeff_field"],
        expected_betti=dict(expected_betti),
        notes=notes,
    )


def _check_common(n_points: int, noise_sigma: float, coeff_field: int) -> None:
    if not isinstance(n_points, (int, np.integer)) or isinstance(n_points, bool):
        raise TypeError(f"n_points must be an int, got {type(n_points).__name__}")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if noise_sigma < 0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")
    if not isinstance(coeff_field, (int, np.integer)) or coeff_field < 2:
        raise ValueError(f"coeff_field must be an int >= 2, got {coeff_field!r}")


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
