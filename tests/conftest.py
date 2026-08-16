"""Fixtures for the S1 scaffolding tests.

The datasets built here are deliberately fake: S1 has no generators yet, and the
package must not ship code that pretends to be one, so the fakes live in the test
tree.  They are still built through ``rng_for``, so they are reproducible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdadyn.synthetic import (
    DYNAMIC_GENERATORS,
    FORMAT_VERSION,
    SERIES_GENERATORS,
    STATIC_GENERATORS,
    DynamicGraphDataset,
    Event,
    GroundTruth,
    PointCloudDataset,
    SeriesDataset,
    build_dynamic_graph,
    build_dynamic_sbm,
    build_network,
    build_null_cloud,
    build_null_graph,
    build_series,
    build_static_cloud,
    load_config,
    rng_for,
)
from tdadyn.synthetic import naming

SEED = 20260201

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

PLANAR_GENERATORS = ("circle", "two_circles", "figure_eight")

FAKE_NAME_ORDERS = {
    "fake_cloud": ("n_points", "dim", "noise_sigma", "sampling"),
    "fake_dynamic": ("n_nodes", "n_snapshots", "threshold"),
    "fake_series": ("n_channels", "n_timepoints", "regimes"),
}


@pytest.fixture(autouse=True)
def register_fake_name_orders(monkeypatch):
    """Register name orders for the fake generators, undone after each test."""
    for generator, order in FAKE_NAME_ORDERS.items():
        monkeypatch.setitem(naming._NAME_ORDER, generator, order)


def make_fake_point_cloud(seed: int = SEED, n_points: int = 8) -> PointCloudDataset:
    """Build a reproducible fake point cloud with a similarity matrix."""
    rng = rng_for(seed, "fake_cloud")
    points = rng.normal(size=(n_points, 2))
    raw = np.exp(-((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    weights = (raw + raw.T) / 2  # exactly symmetric: IEEE addition is commutative
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="static_cloud",
        truth_kind="expected_betti",
        generator="fake_cloud",
        params={"n_points": n_points, "dim": 2, "noise_sigma": 0.03, "sampling": "uniform"},
        seed=seed,
        coeff_field=2,
        expected_betti={0: 1, 1: 1, 2: 0},
        notes="Фиктивный набор для проверки каркаса, не для топологии.",
    )
    return PointCloudDataset(points=points, ground_truth=ground_truth, weights=weights)


def make_fake_dynamic_graph(seed: int = SEED) -> DynamicGraphDataset:
    """Build a reproducible fake sequence of graph snapshots with an event log."""
    n_nodes, n_snapshots = 5, 4
    rng = rng_for(seed, "fake_dynamic")
    raw = rng.random((n_snapshots, n_nodes, n_nodes))
    snapshots = (raw + raw.transpose(0, 2, 1)) / 2
    for index in range(n_snapshots):
        np.fill_diagonal(snapshots[index], 0.0)
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="dynamic_graph",
        truth_kind="betti_curve",
        generator="fake_dynamic",
        params={"n_nodes": n_nodes, "n_snapshots": n_snapshots, "threshold": 0.5},
        seed=seed,
        coeff_field=2,
        complex="flag",
        edge_rule="weight_above",
        edge_threshold=0.5,
        betti_curve={0: [1, 1, 2, 1], 1: [0, 1, 1, 1], 2: [0, 0, 0, 0]},
        events=[
            Event(1, 1, "birth", "h1_a", "путь замыкается в цикл"),
            Event(2, 0, "birth", "h0_b", "граф распадается на две компоненты"),
            Event(3, 0, "death", "h0_b", "компоненты сливаются"),
        ],
        notes="Фиктивная кривая Бетти при случайных снимках: набор проверяет каркас "
        "(схему, round-trip, детерминизм), а не топологию. Согласованность событий "
        "с кривой проверяется на cycle_lifecycle, см. tests/test_dynamic_events.py.",
    )
    return DynamicGraphDataset(snapshots=snapshots, ground_truth=ground_truth)


def make_fake_series(seed: int = SEED) -> SeriesDataset:
    """Build a reproducible fake multivariate series with a coupling schedule."""
    n_channels, n_timepoints, n_regimes = 3, 10, 2
    rng = rng_for(seed, "fake_series")
    series = rng.normal(size=(n_channels, n_timepoints))
    raw = rng.random((n_regimes, n_channels, n_channels))
    coupling = (raw + raw.transpose(0, 2, 1)) / 2
    ground_truth = GroundTruth(
        format_version=FORMAT_VERSION,
        family="series",
        truth_kind="partition",
        generator="fake_series",
        params={"n_channels": n_channels, "n_timepoints": n_timepoints, "regimes": n_regimes},
        seed=seed,
        community_labels=np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64),
        notes="Фиктивный ряд для проверки каркаса.",
    )
    return SeriesDataset(series=series, coupling=coupling, ground_truth=ground_truth)


@pytest.fixture
def fake_point_cloud() -> PointCloudDataset:
    return make_fake_point_cloud()


@pytest.fixture
def fake_dynamic_graph() -> DynamicGraphDataset:
    return make_fake_dynamic_graph()


@pytest.fixture
def fake_series() -> SeriesDataset:
    return make_fake_series()


@pytest.fixture(scope="session")
def config():
    """The repository configuration, loaded once."""
    return load_config(CONFIG_PATH)


@pytest.fixture(params=["point_cloud", "dynamic_graph", "series"])
def any_fake_dataset(request):
    """Every fake dataset in turn, so that generic checks cover all three shapes."""
    return {
        "point_cloud": make_fake_point_cloud,
        "dynamic_graph": make_fake_dynamic_graph,
        "series": make_fake_series,
    }[request.param]()
