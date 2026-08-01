"""Generators of synthetic datasets with analytically known topology.

The value of these datasets is that the correct answer is known in advance, so the
ground truth is an input to the tests, never an output of the code under test.
See ``docs/`` for the task, the method, the data contracts and the validation rules.
"""

from .config import Config, ConfigError, load_config, parse_config
from .contracts import (
    EVENT_TYPES,
    FAMILIES,
    FORMAT_VERSION,
    Dataset,
    DynamicGraphDataset,
    Event,
    GroundTruth,
    PointCloudDataset,
    SeriesDataset,
    dataset_equal,
    ground_truth_equal,
)
from .io import load_dataset, load_ground_truth, save_dataset
from .manifest import ManifestEntry, entry_payload, make_entry, read_manifest, update_manifest
from .naming import format_dataset_id, make_dataset_id, register_name_order
from .rng import rng_for
from .schema import SchemaError, validate_ground_truth_dict

__all__ = [
    "EVENT_TYPES",
    "FAMILIES",
    "FORMAT_VERSION",
    "Config",
    "ConfigError",
    "Dataset",
    "DynamicGraphDataset",
    "Event",
    "GroundTruth",
    "ManifestEntry",
    "PointCloudDataset",
    "SchemaError",
    "SeriesDataset",
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
    "read_manifest",
    "register_name_order",
    "rng_for",
    "save_dataset",
    "update_manifest",
    "validate_ground_truth_dict",
]
