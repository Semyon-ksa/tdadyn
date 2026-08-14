"""Family 3: coupled oscillators driven by a known coupling schedule.

The point of these datasets is that :math:`A(t)` is an **input**: the sliding window in
:mod:`tdadyn.synthetic.sliding_window` has to recover it, and the truth it is checked
against is the schedule written here, never the recovered matrix.

Two properties of the construction are load-bearing and are checked by the tests rather
than assumed:

* **Each regime is integrated by its own call to** ``solve_ivp``.  :math:`A(t)` is
  piecewise constant, so the right-hand side is discontinuous at a regime boundary; an
  adaptive step that crosses one averages two different vector fields and its error
  estimate is computed on neither.  Both accuracy and reproducibility suffer
  (``docs/method.md`` §3.3).
* **The transient is integrated and discarded.**  The Rössler system starts off its
  attractor, and a window landing on the descent measures the descent, not the coupling.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .contracts import FORMAT_VERSION, GroundTruth, SeriesDataset
from .naming import register_name_order
from .rng import base_seed_of

register_name_order(
    "kuramoto", ("n_channels", "n_timepoints", "K", "omega_scale", "regime_length")
)
register_name_order(
    "roessler", ("n_channels", "n_timepoints", "epsilon", "c", "regime_length")
)

ROESSLER_NOTES = (
    "Рёсслер при c=5.7 хаотичен: те же начальные условия и t_eval дают тот же ряд, но "
    "набор НЕ устойчив к смене версии scipy, метода интегрирования или допусков — при "
    "них ряд поедет целиком (docs/method.md, 3.3). Переходный процесс отброшен."
)

KURAMOTO_NOTES = (
    "Наблюдаемая x_i = sin(theta_i). A(t) кусочно-постоянна, каждый режим интегрируется "
    "отдельным вызовом solve_ivp (docs/method.md, 3.3). Переходный процесс отброшен."
)


def coupling_matrix(partition: tuple[int, ...], n_channels: int) -> np.ndarray:
    """Block-diagonal coupling of one regime.

    Parameters
    ----------
    partition : tuple of int
        Sizes of the contiguous blocks; they must sum to ``n_channels``.
    n_channels : int
        Number of channels.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_channels, n_channels)``, 1.0 between distinct channels of the same
        block and 0.0 elsewhere, symmetric with a zero diagonal.
    """
    if sum(partition) != n_channels:
        raise ValueError(
            f"partition {partition} sums to {sum(partition)}, not n_channels={n_channels}"
        )
    matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    start = 0
    for size in partition:
        matrix[start : start + size, start : start + size] = 1.0
        start += size
    np.fill_diagonal(matrix, 0.0)
    return matrix


def coupling_schedule(
    regimes: tuple[tuple[int, ...], ...], n_channels: int
) -> np.ndarray:
    """Stack the coupling matrices of every regime, shape ``(len(regimes), n, n)``."""
    return np.stack([coupling_matrix(partition, n_channels) for partition in regimes])


def community_labels(
    regimes: tuple[tuple[int, ...], ...], n_channels: int
) -> np.ndarray:
    """Block label of every channel in every regime, shape ``(len(regimes), n)``, int64."""
    labels = np.zeros((len(regimes), n_channels), dtype=np.int64)
    for index, partition in enumerate(regimes):
        start = 0
        for block, size in enumerate(partition):
            labels[index, start : start + size] = block
            start += size
    return labels


def make_kuramoto(
    n_channels: int,
    n_timepoints: int,
    dt: float,
    coupling_strength: float,
    omega_scale: float,
    regimes: tuple[tuple[int, ...], ...],
    regime_length: int,
    transient_length: int,
    integrator_method: str,
    rtol: float,
    atol: float,
    rng: np.random.Generator,
) -> SeriesDataset:
    """Integrate the Kuramoto model under a scheduled coupling.

    Parameters
    ----------
    n_channels : int
        Number of oscillators.
    n_timepoints : int
        Length of the stored series; must equal ``regime_length * len(regimes)``.
    dt : float
        Output step.
    coupling_strength : float
        The constant written ``K`` in ``docs/method.md`` §3.2.  Spelled out because the
        letter K means three different things in this project (``docs/method.md`` §0).
    omega_scale : float
        Standard deviation of the natural frequencies, drawn from a normal distribution.
    regimes : tuple of tuple of int
        Block sizes per regime; consecutive partitions must be related by refinement.
    regime_length : int
        Stored samples per regime.
    transient_length : int
        Samples integrated under the first regime and discarded.
    integrator_method : str
        ``solve_ivp`` method.
    rtol, atol : float
        Integrator tolerances.
    rng : numpy.random.Generator
        Source of the natural frequencies and the initial phases.

    Returns
    -------
    SeriesDataset
        ``series`` of shape ``(n_channels, n_timepoints)`` holding
        :math:`x_i = \\sin\\theta_i`, and ``coupling`` of shape
        ``(len(regimes), n_channels, n_channels)``.
    """
    _check_common(n_channels, n_timepoints, dt, regimes, regime_length, transient_length)
    if coupling_strength < 0.0:
        raise ValueError(f"coupling_strength must be >= 0, got {coupling_strength}")
    if omega_scale < 0.0:
        raise ValueError(f"omega_scale must be >= 0, got {omega_scale}")

    omega = rng.normal(0.0, omega_scale, size=n_channels)
    initial = rng.uniform(0.0, 2 * np.pi, size=n_channels)
    matrices = coupling_schedule(regimes, n_channels)

    def derivative(_t, theta, adjacency):
        differences = theta[None, :] - theta[:, None]
        return omega + (coupling_strength / n_channels) * (adjacency * np.sin(differences)).sum(1)

    phases = _integrate_by_regime(
        derivative,
        initial,
        matrices,
        dt=dt,
        regime_length=regime_length,
        transient_length=transient_length,
        method=integrator_method,
        rtol=rtol,
        atol=atol,
    )
    series = np.sin(phases)
    params = {
        "n_channels": int(n_channels),
        "n_timepoints": int(n_timepoints),
        "K": float(coupling_strength),
        "omega_scale": float(omega_scale),
        "regime_length": int(regime_length),
    }
    return SeriesDataset(
        series=series,
        coupling=matrices,
        ground_truth=_ground_truth("kuramoto", params, regimes, n_channels, rng, KURAMOTO_NOTES),
    )


def make_roessler(
    n_channels: int,
    n_timepoints: int,
    dt: float,
    a: float,
    b: float,
    c: float,
    epsilon: float,
    regimes: tuple[tuple[int, ...], ...],
    regime_length: int,
    transient_length: int,
    integrator_method: str,
    rtol: float,
    atol: float,
    rng: np.random.Generator,
) -> SeriesDataset:
    """Integrate diffusively coupled Rössler systems under a scheduled coupling.

    Parameters
    ----------
    n_channels : int
        Number of oscillators; the state vector has ``3 * n_channels`` entries.
    n_timepoints : int
        Length of the stored series; must equal ``regime_length * len(regimes)``.
    dt : float
        Output step.
    a, b, c : float
        Rössler parameters; the standard values are 0.2, 0.2 and 5.7.
    epsilon : float
        Diffusive coupling strength on the :math:`x` component.
    regimes, regime_length, transient_length : see :func:`make_kuramoto`
    integrator_method : str
        ``solve_ivp`` method.
    rtol, atol : float
        Integrator tolerances.
    rng : numpy.random.Generator
        Source of the initial conditions.

    Returns
    -------
    SeriesDataset
        ``series`` holds the :math:`x` component of every oscillator.

    Notes
    -----
    At :math:`c = 5.7` the system is chaotic, so the dataset is reproducible given the
    same inputs but **not** stable against a change of solver, method or tolerances.
    That limitation is recorded in the dataset's notes rather than treated as a defect.
    """
    _check_common(n_channels, n_timepoints, dt, regimes, regime_length, transient_length)
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    initial = rng.uniform(-1.0, 1.0, size=3 * n_channels)
    matrices = coupling_schedule(regimes, n_channels)
    degree = matrices.sum(axis=2)

    def derivative(_t, state, adjacency, row_sums):
        x, y, z = state[:n_channels], state[n_channels : 2 * n_channels], state[2 * n_channels :]
        # Diffusive coupling: sum_j A_ij (x_j - x_i) = (A @ x) - deg_i * x_i.
        coupling = adjacency @ x - row_sums * x
        return np.concatenate((-y - z + epsilon * coupling, x + a * y, b + z * (x - c)))

    states = _integrate_by_regime(
        derivative,
        initial,
        matrices,
        dt=dt,
        regime_length=regime_length,
        transient_length=transient_length,
        method=integrator_method,
        rtol=rtol,
        atol=atol,
        extra=degree,
    )
    series = np.ascontiguousarray(states[:n_channels])
    params = {
        "n_channels": int(n_channels),
        "n_timepoints": int(n_timepoints),
        "epsilon": float(epsilon),
        "c": float(c),
        "regime_length": int(regime_length),
    }
    return SeriesDataset(
        series=series,
        coupling=matrices,
        ground_truth=_ground_truth("roessler", params, regimes, n_channels, rng, ROESSLER_NOTES),
    )


def _integrate_by_regime(
    derivative,
    initial: np.ndarray,
    matrices: np.ndarray,
    *,
    dt: float,
    regime_length: int,
    transient_length: int,
    method: str,
    rtol: float,
    atol: float,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate one regime per call, chaining the state, and drop the transient.

    A single call spanning the whole schedule would let an adaptive step straddle a
    regime boundary, where the right-hand side jumps; the step would then average two
    vector fields and control its error against neither (``docs/method.md`` §3.3).
    """
    state = initial
    start_time = 0.0

    if transient_length > 0:
        state, start_time = _run_segment(
            derivative,
            state,
            matrices[0],
            None if extra is None else extra[0],
            start_time=start_time,
            n_samples=transient_length,
            dt=dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )[1:]

    segments = []
    for index in range(len(matrices)):
        samples, state, start_time = _run_segment(
            derivative,
            state,
            matrices[index],
            None if extra is None else extra[index],
            start_time=start_time,
            n_samples=regime_length,
            dt=dt,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        segments.append(samples)
    return np.concatenate(segments, axis=1)


def _run_segment(
    derivative,
    state: np.ndarray,
    adjacency: np.ndarray,
    extra: np.ndarray | None,
    *,
    start_time: float,
    n_samples: int,
    dt: float,
    method: str,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrate one regime; return its samples, the state after it and the next time.

    ``t_eval`` carries one point past the stored samples so that the continuation state
    is a value the solver actually produced, not an extrapolation.
    """
    times = start_time + np.arange(n_samples + 1) * dt
    args = (adjacency,) if extra is None else (adjacency, extra)
    solution = solve_ivp(
        derivative,
        (times[0], times[-1]),
        state,
        t_eval=times,
        args=args,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"solve_ivp failed on a regime: {solution.message}")
    return solution.y[:, :n_samples], solution.y[:, n_samples], times[-1]


def _ground_truth(
    generator: str,
    params: dict,
    regimes: tuple[tuple[int, ...], ...],
    n_channels: int,
    rng: np.random.Generator,
    notes: str,
) -> GroundTruth:
    """Truth of a series: the coupling schedule, as community labels per regime.

    ``truth_kind`` is ``partition`` and not ``betti_curve``: the truth of these datasets
    is which channels are coupled to which, and there is no barcode among it
    (``docs/data.md`` §4.1).
    """
    return GroundTruth(
        format_version=FORMAT_VERSION,
        family="series",
        truth_kind="partition",
        generator=generator,
        params=params,
        seed=base_seed_of(rng),
        community_labels=community_labels(regimes, n_channels),
        notes=notes,
    )


def _check_common(
    n_channels: int,
    n_timepoints: int,
    dt: float,
    regimes: tuple[tuple[int, ...], ...],
    regime_length: int,
    transient_length: int,
) -> None:
    """Validate what neither model can repair on its own."""
    if n_channels < 2:
        raise ValueError(f"n_channels must be >= 2, got {n_channels}")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be positive and finite, got {dt!r}")
    if regime_length < 1:
        raise ValueError(f"regime_length must be >= 1, got {regime_length}")
    if transient_length < 0:
        raise ValueError(f"transient_length must be >= 0, got {transient_length}")
    if not regimes:
        raise ValueError("at least one coupling regime is required")
    expected = regime_length * len(regimes)
    if n_timepoints != expected:
        raise ValueError(
            f"n_timepoints ({n_timepoints}) must equal regime_length * len(regimes) "
            f"({expected}); otherwise the series and coupling.npy drift apart and a "
            "window is matched against the wrong regime"
        )
