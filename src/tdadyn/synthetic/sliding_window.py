"""Sliding window: multivariate series to a sequence of weighted graphs.

This is a **transform**, not a generator, and it follows the shape of
:mod:`tdadyn.synthetic.network`.  The reason is the layout of ``docs/data.md`` §1: a
``series/`` dataset stores ``series.npy`` and ``coupling.npy``, and there is no directory
for windowed graphs.  Family 1 does the same — the cloud is stored, the network is derived.

Correlation is used here in **two different ways** and they must not be mixed
(``docs/method.md`` §3.4):

* as an edge weight, via :math:`|r|`, which then becomes a distance :math:`1 - |r|`;
* as a statistic, via Fisher's :math:`z = \\operatorname{arctanh} r`, whose variance is
  roughly constant and which is therefore what to average or compare.

:math:`z` is unbounded and :math:`1 - z` is neither in :math:`[0, 1]` nor a metric, so it
never enters the distance path.
"""

from __future__ import annotations

import numpy as np

from .network import distances_from_similarity

FISHER_CLIP = 1.0 - 1e-12
"""Correlations are clipped to this before ``arctanh``: a window in which two channels
are exactly proportional gives :math:`r = 1` and :math:`z = \\infty`."""


def window_starts(n_timepoints: int, *, length: int, step: int) -> np.ndarray:
    """First index of every complete window.

    Parameters
    ----------
    n_timepoints : int
        Length of the series.
    length : int
        Window length.
    step : int
        Stride between consecutive windows.

    Returns
    -------
    numpy.ndarray
        Start indices, int64.  Only complete windows are returned; a trailing partial
        window would be computed from fewer samples and its correlations would not be
        comparable with the rest.
    """
    if length < 2:
        raise ValueError(f"length must be >= 2 for a correlation, got {length}")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if n_timepoints < length:
        raise ValueError(f"series of {n_timepoints} points is shorter than the window ({length})")
    return np.arange(0, n_timepoints - length + 1, step, dtype=np.int64)


def correlation_windows(series: np.ndarray, *, length: int, step: int) -> np.ndarray:
    """Pearson correlation matrix of every window.

    Parameters
    ----------
    series : numpy.ndarray
        Shape ``(n_channels, n_timepoints)`` — channel first, per ``docs/data.md`` §2.
    length, step : int
        Window geometry.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_windows, n_channels, n_channels)``, float64, symmetric with ones on
        the diagonal.
    """
    _check_series(series)
    starts = window_starts(series.shape[1], length=length, step=step)
    windows = np.empty((len(starts), series.shape[0], series.shape[0]), dtype=np.float64)
    for index, start in enumerate(starts):
        block = series[:, start : start + length]
        spread = block.std(axis=1)
        if not (spread > 0).all():
            raise ValueError(
                f"window starting at {start} has a constant channel; its correlation with "
                "anything is undefined"
            )
        correlation = np.corrcoef(block)
        # corrcoef is symmetric up to rounding; make it exactly so, as the contracts
        # compare bitwise.
        windows[index] = (correlation + correlation.T) / 2
    return windows


def similarity_from_correlation(correlations: np.ndarray) -> np.ndarray:
    """Edge weights from correlations: :math:`|r|` with a zero diagonal.

    The modulus is not cosmetic.  ``1 - r`` exceeds one for anticorrelated channels, and
    a distance matrix must be non-negative; more to the point, a strong anticorrelation
    *is* a strong relation, and dropping it by sign would declare those channels
    unrelated (``docs/method.md`` §3.4).
    """
    weights = np.abs(np.asarray(correlations, dtype=np.float64))
    if weights.ndim == 2:
        np.fill_diagonal(weights, 0.0)
        return weights
    for index in range(weights.shape[0]):
        np.fill_diagonal(weights[index], 0.0)
    return weights


def dynamic_graph_from_series(series: np.ndarray, *, length: int, step: int) -> np.ndarray:
    """Turn a series into a sequence of weighted graphs, shape ``(n_windows, n, n)``."""
    return similarity_from_correlation(correlation_windows(series, length=length, step=step))


def distances_from_windows(similarities: np.ndarray) -> np.ndarray:
    """Distance matrices of every window, via :math:`d = 1 - |r|`.

    Reuses :func:`tdadyn.synthetic.network.distances_from_similarity` with the
    ``one_minus`` transform — the same function whose docstring names this as the task-3
    path :math:`d = 1 - |r|`.
    """
    return np.stack(
        [distances_from_similarity(window, transform="one_minus") for window in similarities]
    )


def fisher_z(correlations: np.ndarray) -> np.ndarray:
    """Fisher transform, for statistics only.

    Never on the path to distances: :math:`z` is unbounded, so :math:`1 - z` is neither
    in :math:`[0, 1]` nor a metric (``docs/method.md`` §3.4).
    """
    clipped = np.clip(np.asarray(correlations, dtype=np.float64), -FISHER_CLIP, FISHER_CLIP)
    return np.arctanh(clipped)


def regime_of_window(
    start: int, *, length: int, regime_length: int, n_regimes: int
) -> int | None:
    """Which regime a window lies inside, or ``None`` if it straddles a boundary.

    A straddling window mixes two coupling matrices, so there is no single truth to
    compare it against; such windows are excluded from the agreement statistic rather
    than counted as failures.
    """
    first, last = start // regime_length, (start + length - 1) // regime_length
    if first != last or first >= n_regimes:
        return None
    return int(first)


def coupling_agreement(recovered: np.ndarray, truth: np.ndarray, *, fisher: bool) -> float:
    """How well one recovered matrix agrees with the coupling that produced it.

    Parameters
    ----------
    recovered : numpy.ndarray
        Correlation matrix of a window, shape ``(n, n)``.
    truth : numpy.ndarray
        The regime's coupling matrix, same shape, 0/1.
    fisher : bool
        Compare on Fisher's :math:`z` rather than on raw :math:`r`.  This is the use the
        transform exists for: comparing correlations as statistics.

    Returns
    -------
    float
        Pearson correlation between the off-diagonal entries of the two matrices.  It is
        a correlation of correlations, which is why the threshold it is compared against
        is calibrated rather than reasoned about (``config.yaml``,
        ``sliding_window.min_coupling_agreement``).
    """
    if recovered.shape != truth.shape or recovered.ndim != 2:
        raise ValueError(f"shapes disagree: {recovered.shape} against {truth.shape}")
    off_diagonal = ~np.eye(recovered.shape[0], dtype=bool)
    left = fisher_z(recovered)[off_diagonal] if fisher else recovered[off_diagonal]
    right = truth[off_diagonal]
    if left.std() == 0 or right.std() == 0:
        raise ValueError("agreement is undefined when either matrix is constant")
    return float(np.corrcoef(left, right)[0, 1])


def _check_series(series: np.ndarray) -> None:
    if not isinstance(series, np.ndarray):
        raise TypeError(f"series must be a numpy array, got {type(series).__name__}")
    if series.ndim != 2:
        raise ValueError(f"series must be 2-D (n_channels, n_timepoints), got {series.shape}")
    if not np.isfinite(series).all():
        raise ValueError("series holds non-finite values")
