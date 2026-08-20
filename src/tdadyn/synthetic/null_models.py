"""Family 4: null models — data with no schedule and no planned events.

**A null set does not assert zero; it lets a false-positive rate be measured.**  That
distinction is the whole design (``docs/method.md`` §4.2).  A random graph honestly has
cycles at every workable density — measured on :math:`n = 24`, a clean :math:`(1,0,0)`
appears only above :math:`p \\approx 0.8`, a nearly complete graph — so requiring "no
significant bars" would be requiring the impossible.  The criterion likewise has a
non-zero false-positive rate on random clouds — 1.5% at the configured size, measured over
200 trials on this generator's own stream (``docs/method.md`` §1.2a).

So the truth recorded here is a statement about the **construction**: there is no
schedule and the event log is empty.  It is checkable at generator level and does not
depend on what persistence happens to report.  The observed Betti curve is computed by
the reference but **not written into** ``ground_truth.json``: a measured curve under the
name of truth is truth as pipeline output, the same fork resolved for the SBM in §3.1.

The two models are split across the two pipelines of the project, because a Rips
filtration over a cloud has no density parameter at all — it sweeps every threshold —
while a thresholded graph does.
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.contracts import (
    FORMAT_VERSION,
    DynamicGraphDataset,
    GroundTruth,
    PointCloudDataset,
)
from tdadyn.core.naming import register_name_order
from tdadyn.core.rng import base_seed_of

register_name_order("uniform_cloud", ("n_points", "dim", "repeat"))
register_name_order("erdos_renyi", ("n_nodes", "n_snapshots", "edge_probability", "repeat"))

UNIFORM_CLOUD_NOTES = (
    "Нулевая модель семейства 1: равномерное облако в кубе. Истина — расписания нет, "
    "событий не запланировано; список событий пуст. Критерий значимости имеет на таких "
    "облаках ненулевую частоту ложных срабатываний (1.5% при 400 точках в R^3, измерено "
    "на 200 испытаниях, docs/method.md 1.2a) — набор позволяет её измерить, а не "
    "утверждает ноль."
)

ERDOS_RENYI_NOTES = (
    "Нулевая модель семейства 2: независимые снимки G(n, p) при плотности "
    "cycle_lifecycle. Истина — расписания нет, список событий пуст. Числа Бетти при "
    "этой плотности НЕ нулевые (типично (3,3,0), docs/method.md 4.2): случайный граф "
    "честно имеет циклы, и это нулевое распределение, а не артефакт. Наблюдённая кривая "
    "в ground_truth не пишется — она измерение, а не предсказание."
)


def make_uniform_cloud(
    n_points: int, dim: int, repeat: int, rng: np.random.Generator
) -> PointCloudDataset:
    """Sample a uniform cloud in the unit cube.

    Parameters
    ----------
    n_points : int
        Number of points.
    dim : int
        Ambient dimension.  It matters: for a given point count the criterion fires more
        often in the plane than in space, so the two are not interchangeable
        (``docs/method.md`` §1.2a).
    repeat : int
        Realisation index, part of the name so realisations do not overwrite each other.
    rng : numpy.random.Generator
        The only source of randomness.

    Returns
    -------
    PointCloudDataset
        With ``family="null"``, ``truth_kind="none"`` and an empty event log.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    points = rng.uniform(0.0, 1.0, size=(n_points, dim))
    params = {"n_points": int(n_points), "dim": int(dim), "repeat": _checked_repeat(repeat)}
    return PointCloudDataset(
        points=points,
        ground_truth=_ground_truth("uniform_cloud", params, rng, UNIFORM_CLOUD_NOTES),
    )


def make_erdos_renyi(
    n_nodes: int,
    n_snapshots: int,
    edge_probability: float,
    edge_weight: float,
    repeat: int,
    rng: np.random.Generator,
) -> DynamicGraphDataset:
    """Sample independent :math:`G(n, p)` snapshots.

    Parameters
    ----------
    n_nodes : int
        Nodes per snapshot.
    n_snapshots : int
        Length of the sequence.
    edge_probability : float
        Density; the caller derives it from ``cycle_lifecycle`` rather than reading a
        config key, because a thresholded graph has exactly one density and inventing a
        key would enshrine a choice instead of removing it.
    edge_weight : float
        Weight written on a present edge, matching the similarity convention.
    repeat : int
        Realisation index.
    rng : numpy.random.Generator
        The only source of randomness.

    Returns
    -------
    DynamicGraphDataset
        With ``family="null"``, ``truth_kind="none"`` and an empty event log.

    Notes
    -----
    Snapshots are drawn **independently**, so the Betti curve fluctuates from frame to
    frame.  That fluctuation is the null distribution the localisation algorithm of task 2
    is to be measured against — it is not a defect and must not be smoothed away.
    """
    if n_nodes < 3:
        raise ValueError(f"n_nodes must be >= 3, got {n_nodes}")
    if n_snapshots < 1:
        raise ValueError(f"n_snapshots must be >= 1, got {n_snapshots}")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError(f"edge_probability must lie in [0, 1], got {edge_probability}")
    if not np.isfinite(edge_weight) or edge_weight <= 0.0:
        raise ValueError(f"edge_weight must be positive and finite, got {edge_weight!r}")

    upper = np.triu_indices(n_nodes, 1)
    snapshots = np.zeros((n_snapshots, n_nodes, n_nodes), dtype=np.float64)
    for index in range(n_snapshots):
        present = rng.random(len(upper[0])) < edge_probability
        frame = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        frame[upper] = present * edge_weight
        snapshots[index] = frame + frame.T

    params = {
        "n_nodes": int(n_nodes),
        "n_snapshots": int(n_snapshots),
        "edge_probability": float(edge_probability),
        "repeat": _checked_repeat(repeat),
    }
    return DynamicGraphDataset(
        snapshots=snapshots,
        ground_truth=_ground_truth(
            "erdos_renyi",
            params,
            rng,
            ERDOS_RENYI_NOTES,
            edge_rule="weight_above",
            edge_threshold=edge_weight / 2.0,
        ),
    )


def cycle_lifecycle_density(n_nodes: int, n_edges: int) -> float:
    """Edge density of a graph, counted on the upper triangle.

    The ER null matches the density of ``cycle_lifecycle``; the count is over pairs, not
    over the full matrix, for the reason in ``docs/method.md`` §5.
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
    return n_edges / (n_nodes * (n_nodes - 1) / 2)


def _ground_truth(
    generator: str,
    params: dict,
    rng: np.random.Generator,
    notes: str,
    *,
    edge_rule: str | None = None,
    edge_threshold: float | None = None,
) -> GroundTruth:
    """Truth of a null set: no schedule, and an empty event log to say so."""
    return GroundTruth(
        format_version=FORMAT_VERSION,
        family="null",
        truth_kind="none",
        generator=generator,
        params=params,
        seed=base_seed_of(rng),
        edge_rule=edge_rule,
        edge_threshold=edge_threshold,
        # The empty list is the claim, and it is not the same as the field being absent:
        # absent would mean "not stated", empty means "stated to be nothing".
        events=[],
        notes=notes,
    )


def _checked_repeat(repeat: int) -> int:
    if not isinstance(repeat, (int, np.integer)) or isinstance(repeat, bool) or repeat < 0:
        raise ValueError(f"repeat must be a non-negative int, got {repeat!r}")
    return int(repeat)
