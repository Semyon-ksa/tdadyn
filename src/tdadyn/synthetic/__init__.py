"""Generators of synthetic datasets with analytically known topology.

The value of these datasets is that the correct answer is known in advance, so the
ground truth is an input to the tests, never an output of the code under test.
See ``docs/`` for the task, the method, the data contracts and the validation rules.

Dataclasses, configuration, derived random streams and dataset I/O are **not**
re-exported here: they live in :mod:`tdadyn.core` and are imported from there.  A
re-export would let a consumer of the truth import it from its producer, which is the
one direction the layering forbids (``tests/test_layering.py``), and it would do so invisibly.
"""

from .builders import (
    DYNAMIC_GENERATORS,
    GAUSSIAN_KERNEL_RANGE,
    SERIES_GENERATORS,
    STATIC_GENERATORS,
    build_dynamic_graph,
    build_dynamic_sbm,
    build_network,
    build_null_cloud,
    build_null_graph,
    build_series,
    build_static_cloud,
    noise_stream,
    null_edge_probability,
)
from .dynamic_events import (
    CYCLE_LIFECYCLE_NOTES,
    GEOMETRIC_LOOP_NOTES,
    make_cycle_lifecycle,
    make_geometric_loop,
)
from .dynamic_sbm import DYNAMIC_SBM_NOTES, make_dynamic_sbm
from .noise import (
    NOISE_KINDS,
    NOISY_NOTES,
    add_spurious_edges,
    drop_edges,
    edge_count,
    jitter_points,
    jitter_weights,
    make_jittered_cloud,
    make_jittered_network,
    make_noisy_snapshots,
)
from .null_models import (
    ERDOS_RENYI_NOTES,
    UNIFORM_CLOUD_NOTES,
    cycle_lifecycle_density,
    make_erdos_renyi,
    make_uniform_cloud,
)
from .persistence import (
    betti_at_scale,
    cross_check_scales,
    pairwise_distances,
    persistence_diagrams,
    significant_betti,
)
from .series import (
    KURAMOTO_NOTES,
    ROESSLER_NOTES,
    community_labels,
    coupling_matrix,
    coupling_schedule,
    make_kuramoto,
    make_roessler,
)
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
    "DYNAMIC_GENERATORS",
    "DYNAMIC_SBM_NOTES",
    "ERDOS_RENYI_NOTES",
    "GAUSSIAN_KERNEL_RANGE",
    "GEOMETRIC_LOOP_NOTES",
    "KURAMOTO_NOTES",
    "NOISE_KINDS",
    "NOISY_NOTES",
    "ROESSLER_NOTES",
    "SERIES_GENERATORS",
    "STATIC_GENERATORS",
    "UNIFORM_CLOUD_NOTES",
    "add_spurious_edges",
    "betti_at_scale",
    "build_dynamic_graph",
    "build_dynamic_sbm",
    "build_network",
    "build_null_cloud",
    "build_null_graph",
    "build_series",
    "build_static_cloud",
    "community_labels",
    "coupling_matrix",
    "coupling_schedule",
    "cross_check_scales",
    "cycle_lifecycle_density",
    "drop_edges",
    "edge_count",
    "jitter_points",
    "jitter_weights",
    "klein_bottle_betti",
    "make_circle",
    "make_cycle_lifecycle",
    "make_dynamic_sbm",
    "make_erdos_renyi",
    "make_figure_eight",
    "make_geometric_loop",
    "make_hopf_link",
    "make_jittered_cloud",
    "make_jittered_network",
    "make_klein_bottle",
    "make_kuramoto",
    "make_noisy_snapshots",
    "make_roessler",
    "make_sphere",
    "make_torus",
    "make_two_circles",
    "make_uniform_cloud",
    "noise_stream",
    "null_edge_probability",
    "pairwise_distances",
    "persistence_diagrams",
    "significant_betti",
]
