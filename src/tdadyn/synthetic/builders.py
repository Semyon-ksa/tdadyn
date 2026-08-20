"""The layer between ``config.yaml`` and the generators.

Every generator takes plain scalars and an ``rng``; this module is where the configuration
is turned into those arguments.  It lived in the test tree until S9, which meant nothing in
the package could build a dataset — the CLI needs exactly this mapping, and duplicating it
would let the built data and the tested data drift apart.

Streams are named after the **generator**, never after the dataset identifier
(``rng.py``): an identifier carries the parameters, so keying on it would renumber every
stream whenever a parameter changed, and noisy datasets at different levels would stop
being comparable.
"""

from __future__ import annotations

import numpy as np

from tdadyn.core.config import Config
from tdadyn.core.contracts import DynamicGraphDataset, PointCloudDataset, SeriesDataset
from tdadyn.core.rng import rng_for

from .dynamic_events import make_cycle_lifecycle, make_geometric_loop
from .dynamic_sbm import make_dynamic_sbm
from .network import gaussian_kernel, kernel_width
from .noise import edge_count
from .null_models import cycle_lifecycle_density, make_erdos_renyi, make_uniform_cloud
from .persistence import pairwise_distances
from .series import make_kuramoto, make_roessler
from .static_clouds import (
    make_circle,
    make_figure_eight,
    make_hopf_link,
    make_klein_bottle,
    make_sphere,
    make_torus,
    make_two_circles,
)

STATIC_GENERATORS = (
    "circle",
    "two_circles",
    "figure_eight",
    "torus",
    "sphere",
    "klein_bottle",
    "hopf_link",
)
DYNAMIC_GENERATORS = ("cycle_lifecycle", "geometric_loop")
SERIES_GENERATORS = ("kuramoto", "roessler")

GAUSSIAN_KERNEL_RANGE = (0.0, 1.0)
"""Range a Gaussian kernel can produce, and therefore the clipping range for weight noise.

Not a configuration key: :math:`\\exp(-d^2/2\\sigma^2)` lies in :math:`(0, 1]` by
definition, so this is a property of the transform rather than a setting.  A key would
invite a value the kernel can never produce, and the noise would then clip into a range the
clean data never occupied."""


def build_static_cloud(
    name: str,
    config: Config,
    *,
    noise_sigma: float | None = None,
    n_points: int | None = None,
    sampling: str | None = None,
    seed: int | None = None,
) -> PointCloudDataset:
    """Build one family-1 cloud from the configuration.

    Parameters
    ----------
    name : str
        One of :data:`STATIC_GENERATORS`.
    config : Config
        Loaded configuration; every geometric parameter comes from it.
    noise_sigma, n_points, sampling, seed : optional
        Override the configured values, for tests that need a clean cloud, a smaller one,
        the deliberately wrong sampling mode, or another seed.

    Returns
    -------
    PointCloudDataset
        The dataset, from the stream named after the generator.
    """
    clouds = config.static_clouds
    sigma = clouds.noise_sigma if noise_sigma is None else noise_sigma
    count = clouds.n_points_by_generator.get(name, clouds.n_points)
    if n_points is not None:
        count = n_points
    field = config.topology_check.coeff_field
    rng = rng_for(config.seed if seed is None else seed, name)

    if name == "circle":
        return make_circle(count, clouds.circle.radius, sigma, field, rng)
    if name == "two_circles":
        return make_two_circles(
            count, clouds.two_circles.radius, clouds.two_circles.separation, sigma, field, rng
        )
    if name == "figure_eight":
        return make_figure_eight(count, clouds.figure_eight.scale, sigma, field, rng)
    if name == "torus":
        return make_torus(
            count,
            clouds.torus.R,
            clouds.torus.r,
            sigma,
            clouds.torus.sampling if sampling is None else sampling,
            field,
            rng,
        )
    if name == "sphere":
        return make_sphere(
            count,
            clouds.sphere.radius,
            sigma,
            clouds.sphere.sampling if sampling is None else sampling,
            field,
            rng,
        )
    if name == "klein_bottle":
        return make_klein_bottle(
            count,
            clouds.klein_bottle.a,
            clouds.klein_bottle.b,
            sigma,
            clouds.klein_bottle.sampling if sampling is None else sampling,
            field,
            rng,
        )
    if name == "hopf_link":
        return make_hopf_link(
            count, clouds.hopf_link.radius, clouds.hopf_link.offset, sigma, field, rng
        )
    raise ValueError(f"unknown static-cloud generator {name!r}")


def build_network(points: np.ndarray, config: Config) -> np.ndarray:
    """Build the weighted network of a cloud, through the calibrated family-1 path.

    Parameters
    ----------
    points : numpy.ndarray
        Cloud to build the network over.
    config : Config
        Supplies the kernel width as a fraction of the median pairwise distance.

    Returns
    -------
    numpy.ndarray
        Symmetric similarity matrix.

    Notes
    -----
    No family-1 generator attaches a network — the cloud is the dataset and the network is
    derived — so this is where weight noise gets its input, and where a consumer rebuilds
    a network that family 4 deliberately did not store beside its points.
    """
    distances = pairwise_distances(points)
    sigma = kernel_width(
        distances, median_fraction=config.network_from_cloud.kernel_sigma_median_fraction
    )
    return gaussian_kernel(distances, sigma=sigma)


def build_dynamic_graph(
    name: str,
    config: Config,
    *,
    seed: int | None = None,
    coeff_field: int | None = None,
    n_nodes: int | None = None,
) -> DynamicGraphDataset:
    """Build one family-2 scenario from the configuration."""
    events = config.dynamic_events
    topology = config.topology_check
    field = topology.coeff_field if coeff_field is None else coeff_field
    rng = rng_for(config.seed if seed is None else seed, name)

    if name == "cycle_lifecycle":
        lifecycle = events.scenarios.cycle_lifecycle
        return make_cycle_lifecycle(
            events.n_nodes if n_nodes is None else n_nodes,
            events.n_snapshots,
            events.edge_weight,
            events.threshold,
            lifecycle.path_closes_at,
            lifecycle.chord_added_at,
            lifecycle.edge_removed_at,
            lifecycle.component_splits_at,
            lifecycle.component_merges_at,
            field,
            topology.max_homology_dim,
            topology.reference_max_nodes,
            rng,
        )
    if name == "geometric_loop":
        loop = events.scenarios.geometric_loop
        return make_geometric_loop(
            loop.n_points if n_nodes is None else n_nodes,
            events.n_snapshots,
            loop.rips_epsilon,
            loop.gap_angle,
            loop.radius_schedule,
            loop.gap_open_at,
            loop.gap_close_at,
            field,
            topology.max_homology_dim,
            topology.reference_max_nodes,
            rng,
        )
    raise ValueError(f"unknown dynamic-graph generator {name!r}")


def build_dynamic_sbm(config: Config, *, seed: int | None = None) -> DynamicGraphDataset:
    """Build the family-3 SBM from the configuration."""
    settings = config.dynamic_sbm
    return make_dynamic_sbm(
        settings.n_nodes,
        settings.n_snapshots,
        settings.block_sizes,
        settings.p_in,
        settings.p_out,
        settings.merge_at,
        settings.split_at,
        rng_for(config.seed if seed is None else seed, "dynamic_sbm"),
    )


def build_series(
    name: str,
    config: Config,
    *,
    seed: int | None = None,
    coupling_strength: float | None = None,
) -> SeriesDataset:
    """Build one family-3 series from the configuration."""
    oscillators = config.oscillators
    schedule = oscillators.coupling_schedule
    integrator = oscillators.integrator
    rng = rng_for(config.seed if seed is None else seed, name)
    shared = (
        schedule.regimes,
        schedule.regime_length,
        schedule.transient_length,
        integrator.method,
        integrator.rtol,
        integrator.atol,
    )
    if name == "kuramoto":
        strength = oscillators.kuramoto.K if coupling_strength is None else coupling_strength
        return make_kuramoto(
            oscillators.n_channels,
            oscillators.n_timepoints,
            oscillators.dt,
            strength,
            oscillators.kuramoto.omega_scale,
            *shared,
            rng,
        )
    if name == "roessler":
        strength = (
            oscillators.roessler.epsilon if coupling_strength is None else coupling_strength
        )
        return make_roessler(
            oscillators.n_channels,
            oscillators.n_timepoints,
            oscillators.dt,
            oscillators.roessler.a,
            oscillators.roessler.b,
            oscillators.roessler.c,
            strength,
            *shared,
            rng,
        )
    raise ValueError(f"unknown series generator {name!r}")


def build_null_cloud(
    config: Config, *, repeat: int = 0, n_points: int | None = None, dim: int | None = None
) -> PointCloudDataset:
    """Build the family-1 null model from the configuration."""
    settings = config.null_models.uniform_cloud
    return make_uniform_cloud(
        settings.n_points if n_points is None else n_points,
        settings.dim if dim is None else dim,
        repeat,
        rng_for(config.seed, "uniform_cloud", repeat),
    )


def null_edge_probability(config: Config, cycle_lifecycle: DynamicGraphDataset) -> float:
    """Density the ER null matches, taken from a ``cycle_lifecycle`` already built.

    The base is passed in rather than rebuilt: deriving the density used to reconstruct
    the whole scenario — reference implementation included — on every call.
    """
    truth = cycle_lifecycle.ground_truth
    n_edges = edge_count(
        cycle_lifecycle.snapshots[0],
        edge_rule=truth.edge_rule,
        edge_threshold=truth.edge_threshold,
    )
    return cycle_lifecycle_density(config.dynamic_events.n_nodes, n_edges)


def build_null_graph(
    config: Config, cycle_lifecycle: DynamicGraphDataset, *, repeat: int = 0
) -> DynamicGraphDataset:
    """Build the family-2 null model at the density of ``cycle_lifecycle``."""
    settings = config.null_models.erdos_renyi
    return make_erdos_renyi(
        settings.n_nodes,
        settings.n_snapshots,
        null_edge_probability(config, cycle_lifecycle),
        config.dynamic_events.edge_weight,
        repeat,
        rng_for(config.seed, "erdos_renyi", repeat),
    )


def noise_stream(kind: str, base_generator: str) -> str:
    """Stream name for one noise mechanism over one base.

    Keyed on the base's *generator*, not its identifier: the identifier carries the
    parameters, so keying on it would renumber the streams whenever a parameter changed
    and break comparability across noise levels (``rng.py``).
    """
    return f"{kind}_{base_generator}"
