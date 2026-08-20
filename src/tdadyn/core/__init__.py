"""Shared foundation: what every layer above needs and no layer owns.

Three consumers, not one.  ``synthetic`` generates the truth, ``reference`` computes
independently against it, ``zigzag`` consumes it — and all three need the same
dataclasses, the same reading of a snapshot into a graph, the same derived random
streams and the same configuration.  Duplicating any of those would not be
independence but a second reading of the data; ``edge_rule`` in particular has
**exactly one** implementation, :func:`adjacency_from_snapshot`, because the second
one silently turns a metric dataset into the complement of its graph
(``docs/method.md`` §2.3).

What deliberately did **not** move here is the significance criterion
(``synthetic.persistence.significant_betti``).  Its parameters are calibrated on
family-1 point clouds and ``docs/zigzag.md`` §7 forbids inheriting them; leaving it
one layer up makes that ban structural rather than textual.
"""

from .config import Config, ConfigError, load_config, parse_config
from .contracts import (
    EDGE_RULES,
    EVENT_TYPES,
    FAMILIES,
    FORMAT_VERSION,
    REQUIRED_BY_TRUTH_KIND,
    TRUTH_KINDS,
    Dataset,
    DynamicGraphDataset,
    Event,
    GroundTruth,
    NetworkDataset,
    PointCloudDataset,
    SeriesDataset,
    adjacency_from_snapshot,
    dataset_equal,
    ground_truth_equal,
)
from .io import load_dataset, load_ground_truth, read_json, save_dataset, write_json
from .manifest import (
    ManifestEntry,
    entry_payload,
    make_entry,
    read_manifest,
    truth_digest,
    update_manifest,
    write_manifest,
)
from .naming import format_dataset_id, make_dataset_id, register_name_order
from .rng import base_seed_of, rng_for
from .schema import SchemaError, validate_ground_truth_dict

__all__ = [
    "Config",
    "ConfigError",
    "Dataset",
    "DynamicGraphDataset",
    "EDGE_RULES",
    "EVENT_TYPES",
    "Event",
    "FAMILIES",
    "FORMAT_VERSION",
    "GroundTruth",
    "ManifestEntry",
    "NetworkDataset",
    "PointCloudDataset",
    "REQUIRED_BY_TRUTH_KIND",
    "SchemaError",
    "SeriesDataset",
    "TRUTH_KINDS",
    "adjacency_from_snapshot",
    "base_seed_of",
    "dataset_equal",
    "entry_payload",
    "format_dataset_id",
    "ground_truth_equal",
    "load_config",
    "load_dataset",
    "load_ground_truth",
    "make_dataset_id",
    "make_entry",
    "parse_config",
    "read_json",
    "read_manifest",
    "register_name_order",
    "rng_for",
    "save_dataset",
    "truth_digest",
    "update_manifest",
    "validate_ground_truth_dict",
    "write_json",
    "write_manifest",
]
