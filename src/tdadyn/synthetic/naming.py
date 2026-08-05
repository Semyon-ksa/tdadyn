"""Deterministic directory names for generated datasets.

The name carries *every* generator parameter.  The examples in ``docs/data.md`` used
to list only some of them, which meant that two tori differing in ``R`` produced the
same name and silently overwrote each other.  Completeness is enforced by
:func:`format_dataset_id`, which refuses to build a name unless the declared
parameter order covers the parameter dict exactly.

A parameter may appear in the name either on its own or inside a **group**: several
keys collapsed into one token under a shared prefix, written
``("sched", ["path_closes_at", ...])`` in the declared order.  Groups exist for the
schedule-driven generators of family 2, where listing five keys separately makes the
name unreadable and dropping them makes two scenarios collide (``docs/data.md`` §1.1).
Coverage is not relaxed by grouping: a key inside a group counts as covered, a key in
no entry at all still refuses to build a name.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contracts import GroundTruth

ABBREVIATIONS = {
    "n_points": "N",
    "n_nodes": "N",
    "n_channels": "N",
    "n_snapshots": "T",
    "n_timepoints": "T",
    "noise_sigma": "sig",
    "edge_weight": "w",
    "threshold": "thr",
}
"""Short forms for the most frequent parameters, so that names stay readable.
Every other parameter appears under its own key."""

NameOrderEntry = str | tuple[str, tuple[str, ...]]
"""One position in a name order: a parameter key, or a ``(prefix, keys)`` group."""

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")

_NAME_ORDER: dict[str, tuple[NameOrderEntry, ...]] = {}


def register_name_order(
    generator: str, order: Sequence[str | Sequence[Any]]
) -> None:
    """Declare the parameter order a generator's directory name uses.

    Parameters
    ----------
    generator : str
        Generator name, matching :attr:`GroundTruth.generator`.
    order : sequence
        Positions in the order they appear in the name.  Each is either a parameter
        key, or a ``(prefix, keys)`` pair whose values are joined into a single token.

    Raises
    ------
    ValueError
        If the generator is already registered with a different order.

    Notes
    -----
    The order is declared explicitly rather than taken from the insertion order of
    ``params``: ``ground_truth.json`` is written with sorted keys, so insertion order
    does not survive a round-trip and a name derived from a loaded record would drift
    away from the directory it was loaded from.
    """
    order = _normalise_order(generator, order)
    registered = _NAME_ORDER.get(generator)
    if registered is not None and registered != order:
        raise ValueError(
            f"generator {generator!r} is already registered with order {registered}, "
            f"refusing to re-register with {order}"
        )
    _NAME_ORDER[generator] = order


def name_order(generator: str) -> tuple[NameOrderEntry, ...]:
    """Return the registered parameter order for a generator.

    Parameters
    ----------
    generator : str
        Generator name.

    Returns
    -------
    tuple of NameOrderEntry
        Positions in name order: parameter keys and ``(prefix, keys)`` groups.

    Raises
    ------
    KeyError
        If the generator has not called :func:`register_name_order`.
    """
    try:
        return _NAME_ORDER[generator]
    except KeyError:
        raise KeyError(
            f"generator {generator!r} has no registered name order; "
            "call register_name_order() next to the generator definition"
        ) from None


def make_dataset_id(ground_truth: GroundTruth, *, float_precision: int) -> str:
    """Build the dataset identifier for a ground-truth record.

    Parameters
    ----------
    ground_truth : GroundTruth
        Record whose generator, params and seed determine the name.
    float_precision : int
        Significant digits for float parameters, from ``config.naming.float_precision``.

    Returns
    -------
    str
        The directory name, e.g. ``torus_N400_R2_r0.8_sig0.03_sampling-rejection_seed20260201``.
    """
    return format_dataset_id(
        ground_truth.generator,
        ground_truth.params,
        ground_truth.seed,
        order=name_order(ground_truth.generator),
        float_precision=float_precision,
    )


def format_dataset_id(
    generator: str,
    params: Mapping[str, Any],
    seed: int,
    *,
    order: Sequence[str],
    float_precision: int,
) -> str:
    """Format a dataset identifier from raw parts.

    Parameters
    ----------
    generator : str
        Generator name.
    params : mapping
        Generator arguments; every key must appear in ``order``, on its own or in a group.
    seed : int
        Seed of the dataset's stream.
    order : sequence
        Positions in the order they appear in the name: parameter keys and
        ``(prefix, keys)`` groups.
    float_precision : int
        Significant digits for float parameters.

    Returns
    -------
    str
        The directory name.

    Raises
    ------
    ValueError
        If ``order`` and ``params`` do not cover exactly the same keys — that is the
        check which keeps a parameter from dropping out of the name unnoticed.
    """
    if not _SAFE_TOKEN.match(generator or ""):
        raise ValueError(f"generator name {generator!r} is not usable in a path")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not isinstance(float_precision, int) or float_precision < 1:
        raise ValueError(f"float_precision must be a positive int, got {float_precision!r}")

    order = _normalise_order(generator, order)
    keys = tuple(key for entry in order for key in _entry_keys(entry))
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys in name order for {generator!r}: {keys}")
    labels = tuple(_entry_label(entry) for entry in order)
    if len(set(labels)) != len(labels):
        # Two tokens under one label read as a single parameter and hide the other.
        raise ValueError(f"duplicate labels in name order for {generator!r}: {labels}")
    missing = set(params) - set(keys)
    extra = set(keys) - set(params)
    if missing or extra:
        raise ValueError(
            f"name order for {generator!r} does not match its params: "
            f"missing from the name {sorted(missing)}, absent from params {sorted(extra)}. "
            "Every parameter must appear in the directory name, otherwise two datasets "
            "can collide and overwrite each other."
        )

    parts = [generator]
    for entry, label in zip(order, labels):
        if isinstance(entry, str):
            value = _format_value(f"{generator}.{entry}", params[entry], float_precision)
        else:
            # A group formats as the list of its values, which _format_value already
            # joins with dashes: ("sched", [12, 27]) becomes "sched12-27".
            prefix, group_keys = entry
            value = _format_value(
                f"{generator}.{prefix}", [params[key] for key in group_keys], float_precision
            )
        # Numbers read fine glued to the label ("N400"); words need a separator.
        separator = "" if value[0].isdigit() or value[0] in "-." else "-"
        parts.append(f"{label}{separator}{value}")
    parts.append(f"seed{int(seed)}")
    return "_".join(parts)


def _normalise_order(generator: str, order: Sequence[Any]) -> tuple[NameOrderEntry, ...]:
    """Convert a declared name order into hashable entries.

    Groups arrive as ``(prefix, [keys])`` with a list inside, which is convenient to
    write but unhashable, and the order is compared on re-registration.
    """
    entries: list[NameOrderEntry] = []
    for entry in order:
        if isinstance(entry, str):
            entries.append(entry)
            continue
        if not isinstance(entry, Sequence) or len(entry) != 2:
            raise TypeError(
                f"name order for {generator!r}: an entry is either a parameter key or a "
                f"(prefix, keys) group, got {entry!r}"
            )
        prefix, group_keys = entry
        if not isinstance(prefix, str) or not _SAFE_TOKEN.match(prefix):
            raise ValueError(
                f"name order for {generator!r}: group prefix {prefix!r} is not usable in a path"
            )
        if isinstance(group_keys, str) or not isinstance(group_keys, Sequence):
            raise TypeError(
                f"name order for {generator!r}: group {prefix!r} must list its keys, "
                f"got {group_keys!r}"
            )
        group_keys = tuple(group_keys)
        if not group_keys:
            raise ValueError(f"name order for {generator!r}: group {prefix!r} is empty")
        for key in group_keys:
            if not isinstance(key, str):
                raise TypeError(
                    f"name order for {generator!r}: group {prefix!r} holds a non-string "
                    f"key {key!r}"
                )
        entries.append((prefix, group_keys))
    return tuple(entries)


def _entry_keys(entry: NameOrderEntry) -> tuple[str, ...]:
    """Parameter keys covered by one position of a name order."""
    return (entry,) if isinstance(entry, str) else entry[1]


def _entry_label(entry: NameOrderEntry) -> str:
    """Label one position of a name order appears under."""
    return ABBREVIATIONS.get(entry, entry) if isinstance(entry, str) else entry[0]


def _format_value(path: str, value: Any, float_precision: int) -> str:
    """Render one parameter value as a filesystem-safe, locale-independent token."""
    if value is None:
        return "none"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"{path}: non-finite parameter {value!r} cannot appear in a name")
        return format(float(value), f".{float_precision}g")
    if isinstance(value, str):
        if not _SAFE_TOKEN.match(value):
            raise ValueError(
                f"{path}: string parameter {value!r} is not usable in a path; "
                "allowed characters are letters, digits, '_', '.' and '-'"
            )
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if not items:
            return "empty"
        return "-".join(_format_value(f"{path}[{i}]", item, float_precision) for i, item in enumerate(items))
    raise TypeError(
        f"{path}: parameter of type {type(value).__name__} cannot appear in a dataset name; "
        "flatten nested structures into scalar parameters"
    )
