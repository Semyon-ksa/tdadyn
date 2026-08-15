"""Generators of synthetic datasets with analytically known topology.

The value of these datasets is that the correct answer is known in advance, so the
ground truth is an input to the tests, never an output of the code under test.
See ``docs/`` for the task, the method, the data contracts and the validation rules.
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
    PointCloudDataset,
    SeriesDataset,
    adjacency_from_snapshot,
    dataset_equal,
    ground_truth_equal,
)
from .dynamic_events import (
    CYCLE_LIFECYCLE_NOTES,
    GEOMETRIC_LOOP_NOTES,
    make_cycle_lifecycle,
    make_geometric_loop,
)
from .dynamic_sbm import DYNAMIC_SBM_NOTES, make_dynamic_sbm
from .io import load_dataset, load_ground_truth, save_dataset
from .manifest import ManifestEntry, entry_payload, make_entry, read_manifest, update_manifest
from .naming import format_dataset_id, make_dataset_id, register_name_order
from .noise import (
    NOISE_KINDS,
    NOISY_NOTES,
    add_spurious_edges,
    drop_edges,
    edge_count,
    jitter_points,
    jitter_weights,
    make_noisy_cloud,
    make_noisy_snapshots,
)
from .persistence import (
    betti_at_scale,
    cross_check_scales,
    pairwise_distances,
    persistence_diagrams,
    significant_betti,
)
from .rng import base_seed_of, rng_for
from .series import (
    KURAMOTO_NOTES,
    ROESSLER_NOTES,
    community_labels,
    coupling_matrix,
    coupling_schedule,
    make_kuramoto,
    make_roessler,
)
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
    "CYCLE_LIFECYCLE_NOTES",
    "Config",
    "ConfigError",
    "DYNAMIC_SBM_NOTES",
    "Dataset",
    "DynamicGraphDataset",
    "EDGE_RULES",
    "EVENT_TYPES",
    "Event",
    "FAMILIES",
    "FORMAT_VERSION",
    "GEOMETRIC_LOOP_NOTES",
    "GroundTruth",
    "KURAMOTO_NOTES",
    "ManifestEntry",
    "NOISE_KINDS",
    "NOISY_NOTES",
    "PointCloudDataset",
    "REQUIRED_BY_TRUTH_KIND",
    "ROESSLER_NOTES",
    "SchemaError",
    "SeriesDataset",
    "TRUTH_KINDS",
    "add_spurious_edges",
    "adjacency_from_snapshot",
    "base_seed_of",
    "betti_at_scale",
    "community_labels",
    "coupling_matrix",
    "coupling_schedule",
    "cross_check_scales",
    "dataset_equal",
    "drop_edges",
    "edge_count",
    "entry_payload",
    "format_dataset_id",
    "ground_truth_equal",
    "jitter_points",
    "jitter_weights",
    "klein_bottle_betti",
    "load_config",
    "load_dataset",
    "load_ground_truth",
    "make_circle",
    "make_cycle_lifecycle",
    "make_dataset_id",
    "make_dynamic_sbm",
    "make_entry",
    "make_figure_eight",
    "make_geometric_loop",
    "make_hopf_link",
    "make_klein_bottle",
    "make_kuramoto",
    "make_noisy_cloud",
    "make_noisy_snapshots",
    "make_roessler",
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
