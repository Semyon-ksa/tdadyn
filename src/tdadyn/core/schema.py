"""Serialisation and validation of ``ground_truth.json``.

This module is the single place where homology-dimension keys are converted between
the ``int`` used in code and the ``str`` that JSON forces upon us, as required by
``docs/data.md``.  Nothing else in the package may perform that conversion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    EDGE_RULES,
    EVENT_TYPES,
    FAMILIES,
    FORMAT_VERSION,
    REQUIRED_BY_TRUTH_KIND,
    TRUTH_KINDS,
    Event,
    GroundTruth,
)

_REQUIRED_KEYS = (
    "format_version",
    "dataset_id",
    "family",
    "truth_kind",
    "generator",
    "params",
    "seed",
)

_OPTIONAL_KEYS = (
    "coeff_field",
    "complex",
    "edge_rule",
    "edge_threshold",
    "expected_betti",
    "betti_curve",
    "events",
    "community_labels",
    "noise_level",
    "notes",
)

_EVENT_KEYS = ("snapshot", "dim", "type", "feature_id", "narrative")


class SchemaError(ValueError):
    """Raised when a ground-truth record does not satisfy the schema."""


def ground_truth_to_json_dict(ground_truth: GroundTruth, dataset_id: str) -> dict[str, Any]:
    """Convert a ground-truth record to its JSON representation.

    Parameters
    ----------
    ground_truth : GroundTruth
        Record to serialise.
    dataset_id : str
        Derived identifier, stored so that a directory can be checked against its
        own parameters on load.

    Returns
    -------
    dict
        JSON-ready mapping.  Fields that are ``None`` are omitted rather than written
        as ``null``, which keeps the file byte-stable as optional fields are added.
    """
    payload: dict[str, Any] = {
        "format_version": ground_truth.format_version,
        "dataset_id": dataset_id,
        "family": ground_truth.family,
        "truth_kind": ground_truth.truth_kind,
        "generator": ground_truth.generator,
        "params": jsonify("params", ground_truth.params),
        "seed": int(ground_truth.seed),
        "notes": ground_truth.notes,
    }
    if ground_truth.coeff_field is not None:
        payload["coeff_field"] = int(ground_truth.coeff_field)
    if ground_truth.complex is not None:
        payload["complex"] = ground_truth.complex
    if ground_truth.edge_rule is not None:
        payload["edge_rule"] = ground_truth.edge_rule
    if ground_truth.edge_threshold is not None:
        payload["edge_threshold"] = float(ground_truth.edge_threshold)
    if ground_truth.expected_betti is not None:
        payload["expected_betti"] = _dims_to_json(ground_truth.expected_betti)
    if ground_truth.betti_curve is not None:
        payload["betti_curve"] = _dims_to_json(
            {dim: [int(v) for v in curve] for dim, curve in ground_truth.betti_curve.items()}
        )
    if ground_truth.events is not None:
        payload["events"] = [
            {
                "snapshot": int(event.snapshot),
                "dim": int(event.dim),
                "type": event.type,
                "feature_id": event.feature_id,
                "narrative": event.narrative,
            }
            for event in ground_truth.events
        ]
    if ground_truth.community_labels is not None:
        payload["community_labels"] = ground_truth.community_labels.tolist()
    if ground_truth.noise_level is not None:
        payload["noise_level"] = jsonify("noise_level", ground_truth.noise_level)
    return payload


def ground_truth_from_json_dict(payload: dict[str, Any]) -> GroundTruth:
    """Rebuild a ground-truth record from its JSON representation.

    Parameters
    ----------
    payload : dict
        Parsed ``ground_truth.json``.

    Returns
    -------
    GroundTruth
        The record; the ``dataset_id`` field of the file is not part of the record
        and is verified by :mod:`tdadyn.core.io` instead.

    Raises
    ------
    SchemaError
        If the payload does not satisfy the schema.
    """
    validate_ground_truth_dict(payload)
    expected_betti = payload.get("expected_betti")
    betti_curve = payload.get("betti_curve")
    events = payload.get("events")
    community_labels = payload.get("community_labels")
    return GroundTruth(
        format_version=payload["format_version"],
        family=payload["family"],
        truth_kind=payload["truth_kind"],
        generator=payload["generator"],
        params=payload["params"],
        seed=payload["seed"],
        coeff_field=payload.get("coeff_field"),
        complex=payload.get("complex"),
        edge_rule=payload.get("edge_rule"),
        edge_threshold=payload.get("edge_threshold"),
        expected_betti=None if expected_betti is None else _dims_from_json(expected_betti),
        betti_curve=None if betti_curve is None else _dims_from_json(betti_curve),
        events=None if events is None else [Event(**event) for event in events],
        community_labels=(
            None if community_labels is None else np.asarray(community_labels, dtype=np.int64)
        ),
        noise_level=payload.get("noise_level"),
        notes=payload.get("notes", ""),
    )


def validate_ground_truth_dict(payload: Any) -> None:
    """Validate the JSON representation of a ground-truth record.

    Parameters
    ----------
    payload : Any
        Parsed ``ground_truth.json``.

    Raises
    ------
    SchemaError
        With a message naming the offending field.
    """
    if not isinstance(payload, dict):
        raise SchemaError(f"ground truth must be a JSON object, got {type(payload).__name__}")

    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise SchemaError(f"missing required fields: {missing}")
    unknown = set(payload) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS)
    if unknown:
        raise SchemaError(f"unknown fields: {sorted(unknown)}")

    if payload["format_version"] != FORMAT_VERSION:
        raise SchemaError(
            f"format_version is {payload['format_version']!r}, this package reads {FORMAT_VERSION}"
        )
    if payload["family"] not in FAMILIES:
        raise SchemaError(f"family must be one of {list(FAMILIES)}, got {payload['family']!r}")
    if payload["truth_kind"] not in TRUTH_KINDS:
        raise SchemaError(
            f"truth_kind must be one of {list(TRUTH_KINDS)}, got {payload['truth_kind']!r}"
        )
    for key in ("dataset_id", "generator"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise SchemaError(f"{key} must be a non-empty string")
    if not isinstance(payload["params"], dict):
        raise SchemaError("params must be an object")
    if not isinstance(payload["seed"], int) or isinstance(payload["seed"], bool):
        raise SchemaError(f"seed must be an int, got {payload['seed']!r}")

    # The kind of truth decides the required fields, not the family: the dynamic SBM lays
    # out like a dynamic graph and has no barcode at all (docs/method.md, 3.1).
    for key in REQUIRED_BY_TRUTH_KIND[payload["truth_kind"]]:
        if key not in payload:
            raise SchemaError(f"truth_kind {payload['truth_kind']!r} requires the field {key!r}")

    if "edge_rule" in payload and payload["edge_rule"] not in EDGE_RULES:
        raise SchemaError(
            f"edge_rule must be one of {list(EDGE_RULES)}, got {payload['edge_rule']!r}"
        )
    if "edge_threshold" in payload and (
        not isinstance(payload["edge_threshold"], (int, float))
        or isinstance(payload["edge_threshold"], bool)
    ):
        raise SchemaError(f"edge_threshold must be a number, got {payload['edge_threshold']!r}")

    if "expected_betti" in payload:
        _validate_dim_mapping("expected_betti", payload["expected_betti"], of_lists=False)
    if "betti_curve" in payload:
        _validate_dim_mapping("betti_curve", payload["betti_curve"], of_lists=True)
    if "events" in payload:
        _validate_events(payload["events"])
    if "coeff_field" in payload and (
        not isinstance(payload["coeff_field"], int) or payload["coeff_field"] < 2
    ):
        raise SchemaError(f"coeff_field must be an int >= 2, got {payload['coeff_field']!r}")
    if "complex" in payload and not isinstance(payload["complex"], str):
        raise SchemaError("complex must be a string")
    if "noise_level" in payload and not isinstance(payload["noise_level"], dict):
        raise SchemaError("noise_level must be an object")
    if "notes" in payload and not isinstance(payload["notes"], str):
        raise SchemaError("notes must be a string")
    if "community_labels" in payload and not isinstance(payload["community_labels"], list):
        raise SchemaError("community_labels must be an array")


def _validate_dim_mapping(name: str, mapping: Any, *, of_lists: bool) -> None:
    if not isinstance(mapping, dict):
        raise SchemaError(f"{name} must be an object keyed by homology dimension")
    dims = []
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.isdigit():
            raise SchemaError(f"{name} keys must be decimal strings, got {key!r}")
        dims.append(int(key))
        if of_lists:
            if not isinstance(value, list) or not all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                raise SchemaError(f"{name}[{key}] must be an array of ints")
        elif not isinstance(value, int) or isinstance(value, bool):
            raise SchemaError(f"{name}[{key}] must be an int")
    if sorted(dims) != list(range(len(dims))):
        raise SchemaError(
            f"{name} must state every dimension from 0 upwards, got {sorted(dims)}"
        )


def _validate_events(events: Any) -> None:
    if not isinstance(events, list):
        raise SchemaError("events must be an array")
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise SchemaError(f"events[{index}] must be an object")
        missing = [key for key in _EVENT_KEYS if key not in event]
        unknown = set(event) - set(_EVENT_KEYS)
        if missing or unknown:
            raise SchemaError(f"events[{index}]: missing {missing}, unknown {sorted(unknown)}")
        if event["type"] not in EVENT_TYPES:
            raise SchemaError(
                f"events[{index}].type must be one of {list(EVENT_TYPES)}, got {event['type']!r}. "
                "A barcode has no split or merge; that wording belongs in narrative."
            )
        for key in ("snapshot", "dim"):
            if not isinstance(event[key], int) or isinstance(event[key], bool):
                raise SchemaError(f"events[{index}].{key} must be an int")
        for key in ("feature_id", "narrative"):
            if not isinstance(event[key], str):
                raise SchemaError(f"events[{index}].{key} must be a string")


def _dims_to_json(mapping: dict[int, Any]) -> dict[str, Any]:
    """Convert integer homology-dimension keys to the strings JSON requires."""
    return {str(int(dim)): value for dim, value in mapping.items()}


def _dims_from_json(mapping: dict[str, Any]) -> dict[int, Any]:
    """Convert homology-dimension keys back from strings to ints."""
    return {int(dim): value for dim, value in mapping.items()}


def jsonify(path: str, value: Any) -> Any:
    """Convert numpy scalars and arrays to plain JSON types, refusing anything else."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise SchemaError(f"{path}: {value!r} has no JSON representation")
        return float(value)
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise SchemaError(f"{path}: object keys must be strings, got {key!r}")
        return {key: jsonify(f"{path}.{key}", item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(f"{path}[{i}]", item) for i, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return jsonify(path, value.tolist())
    raise SchemaError(f"{path}: values of type {type(value).__name__} cannot be stored in JSON")
