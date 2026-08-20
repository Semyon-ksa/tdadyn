"""Family 3: coupled oscillators and the sliding window.

The truth here is the coupling schedule, which is an **input**.  The window has to recover
it; nothing in this file takes the recovered matrix as the answer.

Numbers quoted in the assertions come from the calibration recorded in ``config.yaml``
(``sliding_window.min_coupling_agreement``) and from ``docs/method.md`` §3.1a and §3.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import dataset_equal, load_dataset, rng_for, save_dataset
from tdadyn.synthetic import coupling_matrix
from tdadyn.synthetic import sliding_window as sw

from conftest import build_series

OFF_DIAGONAL_TOLERANCE = 0.10
"""Largest within-minus-between gap admissible when the coupling is switched off.

Measured on the three calibration seeds: 0.008, 0.029, 0.011 with ``K = 0``, against
0.238 when the coupling is on.  The bound sits between those, so the test fails on a
construction that invents connectivity instead of merely passing on a finite-window
wobble — a looser value would pass on data with real spurious structure in it."""


@pytest.fixture
def kuramoto(config):
    return build_series("kuramoto", config)


# --- the coupling schedule (docs/method.md 3.1a) ------------------------------------


def test_a_regime_is_a_block_diagonal_partition():
    matrix = coupling_matrix((2, 2), 4)
    expected = np.array(
        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    )
    assert np.array_equal(matrix, expected)


def test_the_coupling_is_symmetric_with_a_zero_diagonal(kuramoto):
    coupling = kuramoto.coupling
    assert np.array_equal(coupling, np.swapaxes(coupling, -1, -2))
    assert np.array_equal(
        np.diagonal(coupling, axis1=1, axis2=2), np.zeros((coupling.shape[0], coupling.shape[1]))
    )


def test_the_schedule_merges_and_then_splits(kuramoto, config):
    # docs/method.md 3.1a: consecutive partitions are related by refinement, so the
    # transitions really are a merge and a split rather than a re-cut.
    labels = kuramoto.ground_truth.community_labels
    regimes = config.oscillators.coupling_schedule.regimes
    assert labels.shape == (len(regimes), config.oscillators.n_channels)
    assert [len(set(row.tolist())) for row in labels] == [len(p) for p in regimes]
    # The merge: two blocks of the first regime share a label in the second.
    assert len(set(labels[1].tolist())) < len(set(labels[0].tolist()))
    assert np.array_equal(labels[2], labels[0])


# --- shapes, dtypes, reproducibility ------------------------------------------------


def test_the_series_has_the_declared_shape_and_dtype(kuramoto, config):
    oscillators = config.oscillators
    assert kuramoto.series.shape == (oscillators.n_channels, oscillators.n_timepoints)
    assert kuramoto.series.dtype == np.float64
    assert kuramoto.coupling.shape[0] == len(oscillators.coupling_schedule.regimes)


def test_the_integrator_is_reproducible(config):
    assert dataset_equal(build_series("kuramoto", config), build_series("kuramoto", config))


def test_the_global_seed_does_not_leak(config, kuramoto):
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(build_series("kuramoto", config), kuramoto)


def test_a_different_seed_gives_a_different_series(config):
    other = build_series("kuramoto", config, seed=7)
    assert not dataset_equal(other, build_series("kuramoto", config))


def test_the_dataset_survives_a_round_trip(kuramoto, config, tmp_path):
    path = save_dataset(kuramoto, tmp_path, float_precision=config.naming.float_precision)
    assert dataset_equal(load_dataset(path), kuramoto)


def test_the_truth_is_the_partition_not_a_barcode(kuramoto):
    # docs/data.md 4.1: a series has no topological truth; its truth is which channels
    # are coupled to which.
    ground_truth = kuramoto.ground_truth
    assert ground_truth.truth_kind == "partition"
    assert ground_truth.betti_curve is None
    assert ground_truth.events is None
    assert ground_truth.community_labels.dtype == np.int64


# --- the window (docs/method.md 3.4) ------------------------------------------------


def test_switching_the_coupling_off_leaves_no_connectivity(config):
    # The control on the construction itself: with K = 0 the oscillators are independent,
    # so any structure the window reports would be an artefact of the window.
    silent = build_series("kuramoto", config, coupling_strength=0.0)
    window = config.sliding_window
    similarities = sw.dynamic_graph_from_series(
        silent.series, length=window.length, step=window.step
    )
    off_diagonal = ~np.eye(config.oscillators.n_channels, dtype=bool)
    within = similarities[:, off_diagonal][:, silent.coupling[0][off_diagonal] == 1]
    between = similarities[:, off_diagonal][:, silent.coupling[0][off_diagonal] == 0]
    assert abs(within.mean() - between.mean()) < OFF_DIAGONAL_TOLERANCE


def test_the_window_covers_several_oscillation_periods(config):
    # docs/method.md 3.4, the lower bound that used to be missing: below one period the
    # correlation measures the shared trend, not the coupling.
    oscillators = config.oscillators
    span = config.sliding_window.length * oscillators.dt
    period = 2 * np.pi / oscillators.kuramoto.omega_scale
    assert span > period, f"window spans {span / period:.2f} periods"


def test_the_coupling_is_above_the_synchronisation_threshold(config):
    # Effective within-block coupling (K/N)*(block-1) against K_c ~ 1.6 * omega_scale.
    oscillators = config.oscillators
    smallest_block = min(min(p) for p in oscillators.coupling_schedule.regimes)
    effective = oscillators.kuramoto.K / oscillators.n_channels * (smallest_block - 1)
    assert effective > 1.6 * oscillators.kuramoto.omega_scale


def test_only_complete_windows_are_returned(config):
    starts = sw.window_starts(1000, length=100, step=10)
    assert starts[0] == 0 and starts[-1] == 900
    assert (starts[-1] + 100) <= 1000


def test_a_window_straddling_a_regime_boundary_has_no_single_truth(config):
    schedule = config.oscillators.coupling_schedule
    length, regime = config.sliding_window.length, schedule.regime_length
    assert sw.regime_of_window(0, length=length, regime_length=regime, n_regimes=3) == 0
    straddling = regime - length // 2
    assert (
        sw.regime_of_window(straddling, length=length, regime_length=regime, n_regimes=3) is None
    )


def test_fisher_stays_off_the_distance_path():
    # docs/method.md 3.4: z is unbounded, so 1 - z is neither in [0, 1] nor a metric.
    correlations = np.array([[1.0, -0.8], [-0.8, 1.0]])
    similarity = sw.similarity_from_correlation(correlations)
    assert similarity[0, 1] == pytest.approx(0.8)
    assert (similarity >= 0).all() and (similarity <= 1).all()
    assert sw.fisher_z(np.array([0.99])) > 2.0  # far outside [0, 1]


def test_distances_from_windows_are_valid_distance_matrices(kuramoto, config):
    window = config.sliding_window
    similarities = sw.dynamic_graph_from_series(
        kuramoto.series, length=window.length, step=window.step
    )
    distances = sw.distances_from_windows(similarities[:3])
    assert (distances >= 0).all()
    assert np.array_equal(distances, np.swapaxes(distances, -1, -2))
    assert np.array_equal(
        np.diagonal(distances, axis1=1, axis2=2), np.zeros((3, config.oscillators.n_channels))
    )


# --- agreement with the true coupling ------------------------------------------------


def pooled_agreement(dataset, config):
    """Mean recovered matrix per regime, correlated with that regime's coupling.

    The calibrated statistic (``config.yaml``): per-window agreement is too noisy to
    threshold, so the windows lying entirely inside one regime are averaged first.
    """
    oscillators, window = config.oscillators, config.sliding_window
    schedule = oscillators.coupling_schedule
    similarities = sw.dynamic_graph_from_series(
        dataset.series, length=window.length, step=window.step
    )
    starts = sw.window_starts(oscillators.n_timepoints, length=window.length, step=window.step)
    grouped: dict[int, list[np.ndarray]] = {}
    for index, start in enumerate(starts):
        regime = sw.regime_of_window(
            int(start),
            length=window.length,
            regime_length=schedule.regime_length,
            n_regimes=len(schedule.regimes),
        )
        if regime is not None:
            grouped.setdefault(regime, []).append(similarities[index])
    return {
        regime: sw.coupling_agreement(
            np.mean(windows, axis=0), dataset.coupling[regime], fisher=False
        )
        for regime, windows in grouped.items()
    }


@pytest.mark.slow
@pytest.mark.parametrize("seed", [20260201, 7, 99])
def test_the_window_recovers_the_coupling_on_every_calibration_seed(config, seed):
    dataset = build_series("kuramoto", config, seed=seed)
    threshold = config.sliding_window.min_coupling_agreement
    for regime, value in sorted(pooled_agreement(dataset, config).items()):
        assert value > threshold, f"regime {regime}: {value:.3f} <= {threshold}"


@pytest.mark.slow
def test_the_roessler_blocks_are_not_completely_synchronised(config):
    # docs/method.md 3.3: above the window the blocks synchronise completely, channels
    # inside a block become bitwise identical and |r| is exactly 1. The dataset then has
    # no within-block variability at all and there is nothing to recover. Measured at
    # epsilon 0.04 the spread is 2.7e-9, at 0.15 it is exactly 0.
    dataset = build_series("roessler", config)
    schedule = config.oscillators.coupling_schedule
    largest_block = max(max(partition) for partition in schedule.regimes)
    for regime, partition in enumerate(schedule.regimes):
        start = regime * schedule.regime_length
        block = dataset.series[: partition[0], start : start + schedule.regime_length]
        spread = np.abs(block - block[0]).max()
        assert spread > 1.0, f"regime {regime}: block collapsed to spread {spread:.2e}"
    assert largest_block <= config.oscillators.n_channels


@pytest.mark.slow
def test_shuffling_the_channels_destroys_the_agreement(config):
    # The honest null: the same recovered matrices against a relabelled truth. Measured
    # -0.039..0.068 across the calibration seeds, against 0.383..0.878 for the truth.
    dataset = build_series("kuramoto", config)
    permutation = rng_for(config.seed, "series_null").permutation(config.oscillators.n_channels)
    shuffled = type(dataset)(
        series=dataset.series,
        coupling=dataset.coupling[:, permutation][:, :, permutation],
        ground_truth=dataset.ground_truth,
    )
    threshold = config.sliding_window.min_coupling_agreement
    for regime, value in sorted(pooled_agreement(shuffled, config).items()):
        assert value < threshold, f"regime {regime}: shuffled labels scored {value:.3f}"
