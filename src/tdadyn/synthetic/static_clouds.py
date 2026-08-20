"""Family 1: point clouds sampled from manifolds whose topology is known analytically.

The parametrisations follow ``docs/method.md`` §1.1 and the expected Betti numbers are
transcribed from the table in §1.2.  Those numbers are an **input**: if a test does not
converge, the fault is in the generator, the sampling or the generation parameters — never
in the table (``docs/validation.md`` §1).

This module covers the planar curves of session S3; the surfaces follow in S4.
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.contracts import FORMAT_VERSION, GroundTruth, PointCloudDataset
from tdadyn.core.naming import register_name_order
from tdadyn.core.rng import base_seed_of

CIRCLE_BETTI = {0: 1, 1: 1, 2: 0}
TWO_CIRCLES_BETTI = {0: 2, 1: 2, 2: 0}
FIGURE_EIGHT_BETTI = {0: 1, 1: 2, 2: 0}
TORUS_BETTI = {0: 1, 1: 2, 2: 1}
SPHERE_BETTI = {0: 1, 1: 0, 2: 1}
HOPF_LINK_BETTI = {0: 2, 1: 2, 2: 0}
KLEIN_BOTTLE_BETTI_MOD_2 = {0: 1, 1: 2, 2: 1}
KLEIN_BOTTLE_BETTI_ODD = {0: 1, 1: 1, 2: 0}
"""Expected Betti numbers, from the table in ``docs/method.md`` §1.2.

All three dimensions are stated because ``docs/task.md`` fixes the maximum homology
dimension of the project at 2; a dimension whose answer is zero is written out, since a
missing key cannot be told from a forgotten one.

The Klein bottle is the one object whose answer depends on the field: over Z/2 the
torsion loop of :math:`H_1` is visible and the answer is (1, 2, 1); in any other
characteristic it is not, and the answer is (1, 1, 0).  Neither is "more correct" —
they answer different questions.

The Hopf link is (2, 2, 0), the same as two disjoint circles: linking is a property of
the embedding, not of the homotopy type.  Expecting :math:`\\beta_0 = 1` here is an error.
"""

register_name_order("circle", ("n_points", "radius", "noise_sigma", "coeff_field"))
register_name_order(
    "two_circles", ("n_points", "radius", "separation", "noise_sigma", "coeff_field")
)
register_name_order("figure_eight", ("n_points", "scale", "noise_sigma", "coeff_field"))
register_name_order(
    "torus", ("n_points", "R", "r", "noise_sigma", "sampling", "coeff_field")
)
register_name_order("sphere", ("n_points", "radius", "noise_sigma", "sampling", "coeff_field"))
register_name_order(
    "klein_bottle", ("n_points", "a", "b", "noise_sigma", "sampling", "coeff_field")
)
register_name_order(
    "hopf_link", ("n_points", "radius", "offset", "noise_sigma", "coeff_field")
)

_NAIVE_SAMPLING_WARNING = (
    "Сэмплирование naive неравномерно по площади и портит ground-truth: набор пригоден "
    "только как отрицательный контроль теста равномерности, но не для табличной проверки "
    "чисел Бетти (docs/method.md, раздел 1.1)."
)


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


def make_torus(
    n_points: int,
    R: float,
    r: float,
    noise_sigma: float,
    sampling: str,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample the torus :math:`T^2` in three dimensions.

    Parameters
    ----------
    n_points : int
        Number of points.
    R, r : float
        Major and minor radii, with ``R > r``.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    sampling : {'rejection', 'naive'}
        ``rejection`` samples uniformly by area; ``naive`` samples the angles uniformly
        and oversamples the inner rim.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 3)`` with the ground truth :math:`(1, 2, 1)`.

    Notes
    -----
    ``docs/method.md`` §1.1.  The area element is proportional to
    :math:`(R + r\\cos\\varphi)`, so uniform angles are **not** uniform by area and the
    inner rim ends up oversampled.  Rejection sampling accepts :math:`\\varphi` with
    probability :math:`(R + r\\cos\\varphi)/(R + r)`.  The non-uniformity is able to eat
    the :math:`\\beta_2` bar, which is why it has a test of its own.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("R", R)
    _check_positive("r", r)
    if R <= r:
        raise ValueError(f"the torus requires R > r, got R={R}, r={r}")
    _check_sampling(sampling, ("rejection", "naive"))

    theta = rng.uniform(0.0, 2 * np.pi, n_points)
    if sampling == "rejection":
        phi = _rejection_sample(
            n_points, lambda angle: R + r * np.cos(angle), R + r, rng
        )
    else:
        phi = rng.uniform(0.0, 2 * np.pi, n_points)

    tube = R + r * np.cos(phi)
    points = np.column_stack([tube * np.cos(theta), tube * np.sin(theta), r * np.sin(phi)])
    points = _add_ambient_noise(points, noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "R": float(R),
        "r": float(r),
        "noise_sigma": float(noise_sigma),
        "sampling": sampling,
        "coeff_field": int(coeff_field),
    }
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth(
            "torus", params, TORUS_BETTI, rng, notes=_sampling_notes(sampling)
        ),
    )


def make_sphere(
    n_points: int,
    radius: float,
    noise_sigma: float,
    sampling: str,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample the sphere :math:`S^2` in three dimensions.

    Parameters
    ----------
    n_points : int
        Number of points.
    radius : float
        Radius of the sphere.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    sampling : {'muller', 'naive_angles'}
        ``muller`` is uniform by area; ``naive_angles`` clusters points at the poles.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 3)`` with the ground truth :math:`(1, 0, 1)`.

    Notes
    -----
    ``docs/method.md`` §1.1: Müller's method draws :math:`g \\sim \\mathcal{N}(0, I_3)`
    and normalises it.  This is the test for :math:`H_2` without :math:`H_1`, so it is
    also the one that shows up false positives in dimension 1.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("radius", radius)
    _check_sampling(sampling, ("muller", "naive_angles"))

    if sampling == "muller":
        gaussian = rng.normal(size=(n_points, 3))
        directions = gaussian / np.linalg.norm(gaussian, axis=1, keepdims=True)
    else:
        theta = rng.uniform(0.0, 2 * np.pi, n_points)
        phi = rng.uniform(0.0, np.pi, n_points)
        directions = np.column_stack(
            [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        )
    points = _add_ambient_noise(radius * directions, noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "radius": float(radius),
        "noise_sigma": float(noise_sigma),
        "sampling": sampling,
        "coeff_field": int(coeff_field),
    }
    notes = "" if sampling == "muller" else _NAIVE_SAMPLING_WARNING
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("sphere", params, SPHERE_BETTI, rng, notes=notes),
    )


def make_klein_bottle(
    n_points: int,
    a: float,
    b: float,
    noise_sigma: float,
    sampling: str,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample the Klein bottle through its tubular embedding in four dimensions.

    Parameters
    ----------
    n_points : int
        Number of points.
    a, b : float
        Major and minor radii of the tube, with ``a > b``.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    sampling : {'rejection', 'naive'}
        ``rejection`` samples uniformly by area.
    coeff_field : int
        Field of coefficients.  **The expected Betti numbers depend on it**: over Z/2 the
        answer is (1, 2, 1), in any other characteristic (1, 1, 0).
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 4)``.

    Notes
    -----
    ``docs/method.md`` §1.1 fixes the parametrisation

    .. math::
        x = (a + b\\cos v)\\cos u,\\quad y = (a + b\\cos v)\\sin u,\\quad
        z = b\\sin v\\cos\\tfrac{u}{2},\\quad w = b\\sin v\\sin\\tfrac{u}{2}.

    It factors through the identification :math:`(u, v) \\sim (u + 2\\pi, -v)`, which is
    what makes it a Klein bottle rather than a torus, and it is an embedding — in
    :math:`\\mathbb{R}^3` only an immersion with a self-intersection exists, and that
    would destroy the ground truth.

    Sampling is by rejection, as for the torus.  Here :math:`|\\partial_v| = b` is
    constant and :math:`\\partial_u \\perp \\partial_v`, so the area element is
    :math:`b\\sqrt{(a + b\\cos v)^2 + b^2\\sin^2 v/4}`, whose maximum over :math:`v` is
    :math:`b(a + b)` at :math:`v = 0`.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("a", a)
    _check_positive("b", b)
    if a <= b:
        raise ValueError(
            f"the Klein bottle requires a > b, got a={a}, b={b}; otherwise the factor "
            "(a + b cos v) reaches zero and the surface degenerates"
        )
    _check_sampling(sampling, ("rejection", "naive"))

    u = rng.uniform(0.0, 2 * np.pi, n_points)
    if sampling == "rejection":
        v = _rejection_sample(n_points, _klein_area_density(a, b), a + b, rng)
    else:
        v = rng.uniform(0.0, 2 * np.pi, n_points)

    tube = a + b * np.cos(v)
    points = np.column_stack(
        [
            tube * np.cos(u),
            tube * np.sin(u),
            b * np.sin(v) * np.cos(u / 2),
            b * np.sin(v) * np.sin(u / 2),
        ]
    )
    points = _add_ambient_noise(points, noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "a": float(a),
        "b": float(b),
        "noise_sigma": float(noise_sigma),
        "sampling": sampling,
        "coeff_field": int(coeff_field),
    }
    notes = (
        "Числа Бетти зависят от поля коэффициентов: над Z/2 — (1,2,1), при иной "
        "характеристике — (1,1,0), см. docs/method.md, раздел 1.2. "
    ) + _sampling_notes(sampling)
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth(
            "klein_bottle", params, klein_bottle_betti(coeff_field), rng, notes=notes.strip()
        ),
    )


def make_hopf_link(
    n_points: int,
    radius: float,
    offset: float,
    noise_sigma: float,
    coeff_field: int,
    rng: np.random.Generator,
) -> PointCloudDataset:
    """Sample two linked circles in three dimensions.

    Parameters
    ----------
    n_points : int
        Total number of points, split ``n_points // 2`` to the first circle.
    radius : float
        Radius of both circles.
    offset : float
        Centre offset of the second circle along the x axis; the circles are linked
        exactly when ``0 < offset < 2 * radius``.
    noise_sigma : float
        Standard deviation of the ambient Gaussian noise.
    coeff_field : int
        Field of coefficients the expected Betti numbers are stated for.
    rng : numpy.random.Generator
        Stream to draw from.

    Returns
    -------
    PointCloudDataset
        Points of shape ``(n_points, 3)`` with the ground truth :math:`(2, 2, 0)`.

    Notes
    -----
    ``docs/method.md`` §1.1: circle one lies in the *xy* plane about the origin, circle
    two in the *xz* plane about ``(offset, 0, 0)``.

    The warning in §1.2 matters here: two linked circles have exactly the homology of two
    disjoint ones, :math:`(2, 2, 0)`.  Linking is a property of the embedding, not of the
    homotopy type, and persistent homology of the space does not see it.  The dataset is
    a **geometric stress test** — the pipeline must not glue two components that pass
    close to one another — and expecting :math:`\\beta_0 = 1` here is an error.
    """
    _check_common(n_points, noise_sigma, coeff_field)
    _check_positive("radius", radius)
    _check_positive("offset", offset)
    if offset >= 2 * radius:
        raise ValueError(
            f"offset ({offset}) must lie strictly between 0 and 2 * radius "
            f"({2 * radius}), otherwise the circles are not linked"
        )

    n_first = n_points // 2
    first_angle = rng.uniform(0.0, 2 * np.pi, n_first)
    second_angle = rng.uniform(0.0, 2 * np.pi, n_points - n_first)
    first = np.column_stack(
        [radius * np.cos(first_angle), radius * np.sin(first_angle), np.zeros(n_first)]
    )
    second = np.column_stack(
        [
            offset + radius * np.cos(second_angle),
            np.zeros(n_points - n_first),
            radius * np.sin(second_angle),
        ]
    )
    points = _add_ambient_noise(np.vstack([first, second]), noise_sigma, rng)
    params = {
        "n_points": int(n_points),
        "radius": float(radius),
        "offset": float(offset),
        "noise_sigma": float(noise_sigma),
        "coeff_field": int(coeff_field),
    }
    notes = (
        "Геометрический стресс-тест: сцепленность не различается персистентными "
        "гомологиями, ожидание (2,2,0) — то же, что у двух разъединённых окружностей."
    )
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("hopf_link", params, HOPF_LINK_BETTI, rng, notes=notes),
    )


def klein_bottle_betti(coeff_field: int) -> dict[int, int]:
    """Expected Betti numbers of the Klein bottle for one field of coefficients.

    Parameters
    ----------
    coeff_field : int
        Prime characteristic.

    Returns
    -------
    dict of int to int
        ``(1, 2, 1)`` over Z/2 and ``(1, 1, 0)`` otherwise.

    Notes
    -----
    ``docs/method.md`` §1.2: the Klein bottle has a loop that is not a boundary but
    becomes one when traversed twice.  Over Z/2 the coefficient 2 vanishes and the loop
    shows up as a class of its own; in any other characteristic it does not.
    """
    return dict(KLEIN_BOTTLE_BETTI_MOD_2 if coeff_field == 2 else KLEIN_BOTTLE_BETTI_ODD)


def _klein_area_density(a: float, b: float):
    """Area element of the tubular Klein bottle, up to the constant factor ``b``."""

    def density(angle: np.ndarray) -> np.ndarray:
        return np.sqrt((a + b * np.cos(angle)) ** 2 + (b * np.sin(angle)) ** 2 / 4)

    return density


def _rejection_sample(
    n_points: int, density, density_max: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw angles in ``[0, 2*pi)`` with probability proportional to ``density``.

    Candidates are drawn in batches of ``n_points`` and accepted with probability
    ``density(candidate) / density_max``; the surplus of the last batch is discarded so
    that exactly ``n_points`` angles come back.
    """
    accepted = np.empty(n_points)
    filled = 0
    while filled < n_points:
        candidate = rng.uniform(0.0, 2 * np.pi, n_points)
        keep = rng.uniform(0.0, 1.0, n_points) < density(candidate) / density_max
        taken = candidate[keep][: n_points - filled]
        accepted[filled : filled + taken.size] = taken
        filled += taken.size
    return accepted


def _sampling_notes(sampling: str) -> str:
    return "" if sampling in ("rejection", "muller") else _NAIVE_SAMPLING_WARNING


def _check_sampling(sampling: str, allowed: tuple[str, ...]) -> None:
    if sampling not in allowed:
        raise ValueError(f"sampling must be one of {allowed}, got {sampling!r}")


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
        truth_kind="expected_betti",
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
