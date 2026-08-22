"""Datasets the zigzag tests run on.

Built in memory rather than read from disk: the construction under test does not touch
the filesystem, and a test that needed ``data/synthetic/`` to exist would fail for a
reason that has nothing to do with what it checks.  The reading path itself is exercised
where it belongs — in the tests of :mod:`tdadyn.core`.
"""

from __future__ import annotations

import pytest

from tdadyn.synthetic import build_dynamic_graph

FAMILY_TWO = ("cycle_lifecycle", "geometric_loop")
"""The two scenarios of family 2 (``docs/method.md`` §2).  The dynamic SBM lays out in the
same directory but is **not** here: it has 60 nodes against ``reference_max_nodes`` of 30,
so its complexes cannot be built at all — see the measurement in ``docs/plan_task2.md`` §Z1.
"""


# Session-scoped: the generators are deterministic and the datasets are read-only here, so
# rebuilding one per test bought nothing and cost more than every assertion put together —
# measured at 0.25 s of setup against milliseconds of checking.


@pytest.fixture(scope="session")
def cycle_lifecycle(config):
    return build_dynamic_graph("cycle_lifecycle", config)


@pytest.fixture(scope="session")
def geometric_loop(config):
    return build_dynamic_graph("geometric_loop", config)


@pytest.fixture(scope="session", params=FAMILY_TWO)
def family_two_dataset(request, config):
    """Each family-2 scenario in turn, so the invariants are checked on both."""
    return build_dynamic_graph(request.param, config)


@pytest.fixture(scope="session")
def zigzag_of(config):
    """Complexes, indices, oracle modules and barcodes, computed once each.

    The oracle is quadratic in the number of indices and costs seconds per dataset and
    dimension; the datasets are read-only and everything here is deterministic, so
    recomputing per test cost far more than every assertion together — 109 s against 17 s
    measured across the backend tests alone.  Same lesson as the fixtures above, one layer
    up again.
    """
    from tdadyn.reference import barcode, module_from_complexes
    from tdadyn.zigzag import index_set, zigzag_complexes

    cache: dict = {}
    args = {
        "max_homology_dim": config.zigzag.max_homology_dim,
        "max_nodes": config.topology_check.reference_max_nodes,
    }

    def complexes(dataset):
        key = ("complexes", dataset.ground_truth.generator, dataset.snapshots.tobytes()[:64])
        if key not in cache:
            diagram = zigzag_complexes(dataset, **args)
            indices = index_set(len(dataset.snapshots))
            cache[key] = ([diagram[index] for index in indices], indices)
        return cache[key]

    def module(dataset, dim):
        key = ("module", dataset.ground_truth.generator, dataset.snapshots.tobytes()[:64], dim)
        if key not in cache:
            cache[key] = module_from_complexes(complexes(dataset)[0], dim)
        return cache[key]

    def bars(dataset, dim):
        key = ("bars", dataset.ground_truth.generator, dataset.snapshots.tobytes()[:64], dim)
        if key not in cache:
            cache[key] = barcode(module(dataset, dim))
        return cache[key]

    return type("ZigzagOf", (), {"complexes": staticmethod(complexes),
                                 "module": staticmethod(module),
                                 "bars": staticmethod(bars)})
