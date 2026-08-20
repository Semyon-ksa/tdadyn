"""Determinism of the scaffolding: names, datasets and comparison semantics."""

from __future__ import annotations

import numpy as np
import pytest

from tdadyn.core import dataset_equal, format_dataset_id, make_dataset_id
from tdadyn.core.contracts import values_equal
from tdadyn.core.naming import register_name_order

from conftest import make_fake_dynamic_graph, make_fake_point_cloud, make_fake_series

FLOAT_PRECISION = 6
BUILDERS = (make_fake_point_cloud, make_fake_dynamic_graph, make_fake_series)


@pytest.mark.parametrize("build", BUILDERS, ids=lambda fn: fn.__name__)
def test_same_seed_gives_bitwise_identical_arrays(build):
    first, second = build(), build()
    assert dataset_equal(first, second)


@pytest.mark.parametrize("build", BUILDERS, ids=lambda fn: fn.__name__)
def test_global_numpy_seed_does_not_change_the_dataset(build):
    expected = build()
    np.random.seed(999)
    np.random.normal(size=1000)
    assert dataset_equal(build(), expected)


def test_different_seeds_give_different_data():
    assert not dataset_equal(make_fake_point_cloud(seed=1), make_fake_point_cloud(seed=2))


def test_equality_operator_does_not_silently_succeed(fake_point_cloud):
    # eq=False: two distinct objects compare unequal, so a test that reaches for ==
    # fails loudly instead of comparing arrays with an ambiguous truth value.
    assert fake_point_cloud != make_fake_point_cloud()
    assert dataset_equal(fake_point_cloud, make_fake_point_cloud())


def test_dataset_equal_notices_a_single_changed_value(fake_point_cloud):
    modified = make_fake_point_cloud()
    modified.points[0, 0] += 1e-12
    assert not dataset_equal(fake_point_cloud, modified)


def test_comparison_notices_a_changed_dtype():
    # Equal values in a different dtype are not the same data: a round-trip that
    # narrowed float64 to float32 must not pass as identical.
    values = np.array([1.0, 2.0, 3.0])
    assert not values_equal(values, values.astype(np.float32))
    assert values_equal(values, values.copy())


# --- naming ----------------------------------------------------------------------


def test_dataset_id_depends_only_on_generator_params_and_seed(fake_point_cloud):
    first = make_dataset_id(fake_point_cloud.ground_truth, float_precision=FLOAT_PRECISION)
    second = make_dataset_id(make_fake_point_cloud().ground_truth, float_precision=FLOAT_PRECISION)
    assert first == second
    assert first == "fake_cloud_N8_dim2_sig0.03_sampling-uniform_seed20260201"


def test_every_parameter_appears_in_the_name():
    # The check that stops a parameter from dropping out of the name: two tori with
    # different R must not end up in the same directory.
    with pytest.raises(ValueError, match="Every parameter must appear"):
        format_dataset_id(
            "torus",
            {"n_points": 400, "R": 2.0, "r": 0.8},
            20260201,
            order=("n_points", "R"),
            float_precision=FLOAT_PRECISION,
        )


def test_a_name_order_key_absent_from_params_is_also_an_error():
    with pytest.raises(ValueError, match="absent from params"):
        format_dataset_id(
            "torus",
            {"n_points": 400},
            1,
            order=("n_points", "R"),
            float_precision=FLOAT_PRECISION,
        )


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"R": 2.0}, "g_R2_seed1"),
        ({"R": 0.03}, "g_R0.03_seed1"),
        ({"R": 1e-7}, "g_R1e-07_seed1"),
        ({"sampling": "rejection"}, "g_sampling-rejection_seed1"),
        ({"fisher_z": True}, "g_fisher_z-true_seed1"),
        ({"schedule": [1.0, 1.4, 1.0]}, "g_schedule1-1.4-1_seed1"),
        ({"n_points": 400}, "g_N400_seed1"),
    ],
)
def test_value_formatting_is_stable(params, expected):
    key = next(iter(params))
    assert (
        format_dataset_id("g", params, 1, order=(key,), float_precision=FLOAT_PRECISION)
        == expected
    )


SCHEDULE_PARAMS = {
    "n_nodes": 24,
    "n_snapshots": 60,
    "edge_weight": 1.0,
    "threshold": 0.5,
    "path_closes_at": 12,
    "chord_added_at": 27,
    "edge_removed_at": 41,
    "component_splits_at": 48,
    "component_merges_at": 55,
}
SCHEDULE_ORDER = [
    "n_nodes",
    "n_snapshots",
    "edge_weight",
    "threshold",
    (
        "sched",
        [
            "path_closes_at",
            "chord_added_at",
            "edge_removed_at",
            "component_splits_at",
            "component_merges_at",
        ],
    ),
]


def test_a_group_of_keys_collapses_into_one_token():
    # docs/data.md 1.1: listing five schedule keys separately makes the name unreadable,
    # dropping them makes two scenarios collide. The group is the way out of both.
    assert (
        format_dataset_id(
            "cycle_lifecycle",
            SCHEDULE_PARAMS,
            20260201,
            order=SCHEDULE_ORDER,
            float_precision=FLOAT_PRECISION,
        )
        == "cycle_lifecycle_N24_T60_w1_thr0.5_sched12-27-41-48-55_seed20260201"
    )


def test_a_key_outside_every_group_is_still_refused():
    # Grouping must not become a hole in the coverage check.
    with pytest.raises(ValueError, match="Every parameter must appear"):
        format_dataset_id(
            "g", {"a": 1, "b": 2, "c": 3}, 1, order=[("grp", ["a", "b"])], float_precision=6
        )


def test_a_group_key_absent_from_params_is_refused():
    with pytest.raises(ValueError, match="absent from params"):
        format_dataset_id("g", {"a": 1}, 1, order=[("grp", ["a", "b"])], float_precision=6)


def test_a_key_listed_in_two_groups_is_refused():
    with pytest.raises(ValueError, match="duplicate keys"):
        format_dataset_id(
            "g", {"a": 1}, 1, order=[("x", ["a"]), ("y", ["a"])], float_precision=6
        )


def test_a_group_prefix_colliding_with_a_label_is_refused():
    # Two tokens under one label read as one parameter and hide the other.
    with pytest.raises(ValueError, match="duplicate labels"):
        format_dataset_id(
            "g", {"a": 1, "b": 2}, 1, order=["a", ("a", ["b"])], float_precision=6
        )


def test_a_malformed_group_is_refused():
    with pytest.raises(ValueError, match="empty"):
        format_dataset_id("g", {"a": 1}, 1, order=[("grp", [])], float_precision=6)
    with pytest.raises(TypeError, match="must list its keys"):
        format_dataset_id("g", {"a": 1}, 1, order=[("grp", "a")], float_precision=6)


def test_registering_a_group_order_is_idempotent(monkeypatch):
    from tdadyn.core import naming

    monkeypatch.setattr(naming, "_NAME_ORDER", dict(naming._NAME_ORDER))
    register_name_order("grouped", ["a", ("s", ["b", "c"])])
    register_name_order("grouped", ["a", ("s", ["b", "c"])])
    with pytest.raises(ValueError, match="already registered"):
        register_name_order("grouped", ["a", ("s", ["c", "b"])])


def test_unusable_values_are_rejected():
    with pytest.raises(ValueError, match="not usable in a path"):
        format_dataset_id("g", {"label": "a/b"}, 1, order=("label",), float_precision=6)
    with pytest.raises(ValueError, match="non-finite"):
        format_dataset_id("g", {"x": float("nan")}, 1, order=("x",), float_precision=6)
    with pytest.raises(TypeError, match="cannot appear in a dataset name"):
        format_dataset_id("g", {"x": {"y": 1}}, 1, order=("x",), float_precision=6)


def test_registering_a_conflicting_order_is_refused(monkeypatch):
    from tdadyn.core import naming

    monkeypatch.setitem(naming._NAME_ORDER, "duplicate", ("a", "b"))
    register_name_order("duplicate", ("a", "b"))  # idempotent
    with pytest.raises(ValueError, match="already registered"):
        register_name_order("duplicate", ("b", "a"))


def test_unregistered_generator_is_reported(fake_point_cloud, monkeypatch):
    from tdadyn.core import naming

    monkeypatch.delitem(naming._NAME_ORDER, "fake_cloud")
    with pytest.raises(KeyError, match="no registered name order"):
        make_dataset_id(fake_point_cloud.ground_truth, float_precision=FLOAT_PRECISION)
