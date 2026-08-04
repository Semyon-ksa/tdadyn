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
from .persistence import (
    betti_at_scale,
    cross_check_scales,
    pairwise_distances,
    persistence_diagrams,
    significant_betti,
)
from .rng import base_seed_of, rng_for
from .schema import SchemaError, validate_ground_truth_dict
from .static_clouds import (
    klein_bottle_betti,
    make_circle,
    make_figure_eight,
    make_hopf_link,
    make_klein_bottle,
    make_sphere,
    make_torus,
    make_two_circles,
)

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
    "base_seed_of",
    "betti_at_scale",
    "cross_check_scales",
    "dataset_equal",
    "entry_payload",
    "format_dataset_id",
    "ground_truth_equal",
    "klein_bottle_betti",
    "load_config",
    "load_dataset",
    "load_ground_truth",
    "make_circle",
    "make_dataset_id",
    "make_entry",
    "make_figure_eight",
    "make_hopf_link",
    "make_klein_bottle",
    "make_sphere",
    "make_torus",
    "make_two_circles",
    "pairwise_distances",
    "parse_config",
    "persistence_diagrams",
    "read_manifest",
    "register_name_order",
    "rng_for",
    "save_dataset",
    "significant_betti",
    "update_manifest",
    "validate_ground_truth_dict",
]
