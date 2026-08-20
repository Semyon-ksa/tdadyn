"""The ground-truth schema: field set, dimension keys, event types, families."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tdadyn.core import FORMAT_VERSION, Event, GroundTruth, ground_truth_equal
from tdadyn.core.schema import (
    SchemaError,
    ground_truth_from_json_dict,
    ground_truth_to_json_dict,
    validate_ground_truth_dict,
)

DATASET_ID = "fake_cloud_N8_dim2_sig0.03_sampling-uniform_seed20260201"


def as_json(dataset) -> dict:
    return ground_truth_to_json_dict(dataset.ground_truth, DATASET_ID)


def test_round_trip_through_json_preserves_the_record(any_fake_dataset):
    payload = as_json(any_fake_dataset)
    restored = ground_truth_from_json_dict(payload)
    assert ground_truth_equal(restored, any_fake_dataset.ground_truth)


def test_dimension_keys_are_strings_in_json_and_ints_in_code(fake_dynamic_graph):
    payload = as_json(fake_dynamic_graph)
    assert set(payload["betti_curve"]) == {"0", "1", "2"}
    restored = ground_truth_from_json_dict(payload)
    assert set(restored.betti_curve) == {0, 1, 2}


def test_community_labels_survive_as_int64(fake_series):
    restored = ground_truth_from_json_dict(as_json(fake_series))
    assert restored.community_labels.dtype == np.int64
    assert np.array_equal(restored.community_labels, fake_series.ground_truth.community_labels)


def test_absent_optional_fields_are_omitted_not_written_as_null(fake_series):
    payload = as_json(fake_series)
    assert "expected_betti" not in payload
    assert "events" not in payload
    assert None not in payload.values()


def test_split_is_not_an_event_type():
    # A barcode has births and deaths only; "split" belongs in the narrative.
    with pytest.raises(ValueError, match="narrative"):
        Event(3, 1, "split", "h1_a", "хорда делит цикл на два")


def test_split_is_rejected_by_the_json_validator(fake_dynamic_graph):
    payload = as_json(fake_dynamic_graph)
    payload["events"][0]["type"] = "split"
    with pytest.raises(SchemaError, match="no split or merge"):
        validate_ground_truth_dict(payload)


def test_unknown_field_is_rejected(fake_dynamic_graph):
    payload = {**as_json(fake_dynamic_graph), "betti_curve_v2": {}}
    with pytest.raises(SchemaError, match="betti_curve_v2"):
        validate_ground_truth_dict(payload)


def test_missing_required_field_is_rejected(fake_dynamic_graph):
    payload = as_json(fake_dynamic_graph)
    del payload["seed"]
    with pytest.raises(SchemaError, match="seed"):
        validate_ground_truth_dict(payload)


def test_stale_format_version_is_rejected(fake_dynamic_graph):
    payload = {**as_json(fake_dynamic_graph), "format_version": FORMAT_VERSION + 1}
    with pytest.raises(SchemaError, match="format_version"):
        validate_ground_truth_dict(payload)


@pytest.mark.parametrize(
    ("truth_kind", "field"),
    [
        ("expected_betti", "expected_betti"),
        ("betti_curve", "betti_curve"),
        ("betti_curve", "events"),
        ("betti_curve", "edge_rule"),
        ("betti_curve", "edge_threshold"),
        ("partition", "community_labels"),
        ("noise_level", "noise_level"),
    ],
)
def test_the_truth_kind_requires_its_fields(fake_dynamic_graph, truth_kind, field):
    # docs/data.md 4.1: the required fields hang off truth_kind, not off family — the
    # dynamic SBM lays out like a dynamic graph and carries no barcode at all.
    payload = as_json(fake_dynamic_graph)
    payload["truth_kind"] = truth_kind
    payload.pop(field, None)
    with pytest.raises(SchemaError, match=field):
        validate_ground_truth_dict(payload)


def test_a_dynamic_graph_may_carry_a_partition_instead_of_a_barcode(fake_dynamic_graph):
    # The case the split exists for: the SBM. Same layout on disk, different truth.
    payload = as_json(fake_dynamic_graph)
    payload["truth_kind"] = "partition"
    for field in ("betti_curve", "events", "complex", "edge_rule", "edge_threshold"):
        payload.pop(field, None)
    payload["community_labels"] = [[0, 0, 1], [0, 1, 1]]
    validate_ground_truth_dict(payload)


def test_dimensions_must_be_stated_from_zero_upwards(fake_dynamic_graph):
    payload = as_json(fake_dynamic_graph)
    del payload["betti_curve"]["2"]
    payload["betti_curve"]["3"] = [0, 0, 0, 0]
    with pytest.raises(SchemaError, match="every dimension from 0"):
        validate_ground_truth_dict(payload)


def test_a_skipped_dimension_is_rejected():
    # beta_1 = 0 must be written out; a missing key is indistinguishable from a
    # forgotten one.
    with pytest.raises(ValueError, match="contiguous range"):
        GroundTruth(
            format_version=FORMAT_VERSION,
            family="static_cloud",
            truth_kind="expected_betti",
            generator="fake_cloud",
            params={},
            seed=1,
            coeff_field=2,
            expected_betti={0: 1, 2: 1},
        )


def test_betti_curve_lengths_must_agree_across_dimensions():
    with pytest.raises(ValueError, match="different lengths"):
        GroundTruth(
            format_version=FORMAT_VERSION,
            family="dynamic_graph",
            truth_kind="betti_curve",
            generator="fake_dynamic",
            params={},
            seed=1,
            coeff_field=2,
            complex="flag",
            edge_rule="weight_above",
            edge_threshold=0.5,
            betti_curve={0: [1, 1, 1], 1: [0, 1]},
            events=[],
        )


def test_event_outside_the_betti_curve_is_rejected():
    with pytest.raises(ValueError, match="outside the betti_curve"):
        GroundTruth(
            format_version=FORMAT_VERSION,
            family="dynamic_graph",
            truth_kind="betti_curve",
            generator="fake_dynamic",
            params={},
            seed=1,
            coeff_field=2,
            complex="flag",
            edge_rule="weight_above",
            edge_threshold=0.5,
            betti_curve={0: [1, 1], 1: [0, 0]},
            events=[Event(5, 1, "birth", "h1_a", "цикл замкнулся")],
        )


def test_non_serialisable_parameter_is_refused(fake_point_cloud):
    ground_truth = copy.copy(fake_point_cloud.ground_truth)
    payload_params = dict(ground_truth.params)
    payload_params["callback"] = print
    object.__setattr__(ground_truth, "params", payload_params)
    with pytest.raises(SchemaError, match="cannot be stored in JSON"):
        ground_truth_to_json_dict(ground_truth, DATASET_ID)
