"""Loading and validating ``config.yaml``.

Unknown keys are an error.  In a project where a typo means a parameter silently
kept its old value, "ignore what we do not recognise" is the worst possible default:
the run succeeds, the data look plausible, and the truth no longer describes them.

Every check below either restates a constraint from ``docs/`` or a comment already
present in ``config.yaml``; nothing here is a preference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import FORMAT_VERSION

DEFAULT_CONFIG_NAME = "config.yaml"

_STATIC_CLOUD_GENERATORS = (
    "circle",
    "two_circles",
    "figure_eight",
    "torus",
    "sphere",
    "klein_bottle",
    "hopf_link",
)


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is missing a key, has a wrong type or an impossible value."""


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class TopologyCheck:
    """Global topology-checking parameters; never tuned per dataset."""

    max_homology_dim: int
    coeff_field: int
    coeff_field_test_values: tuple[int, ...]
    criterion: str
    significance_ratio: float
    gap_noise_floor_ratio: float
    gap_min_ratio: float
    calibration_seeds: tuple[int, ...]
    reference_max_nodes: int
    cross_check_n_points: int
    cross_check_n_scales: int


@dataclass(frozen=True)
class Circle:
    radius: float


@dataclass(frozen=True)
class TwoCircles:
    radius: float
    separation: float


@dataclass(frozen=True)
class FigureEight:
    scale: float


@dataclass(frozen=True)
class Torus:
    R: float
    r: float
    sampling: str


@dataclass(frozen=True)
class Sphere:
    radius: float
    sampling: str


@dataclass(frozen=True)
class KleinBottle:
    embedding: str
    a: float
    b: float
    sampling: str


@dataclass(frozen=True)
class HopfLink:
    radius: float
    offset: float


@dataclass(frozen=True)
class StaticClouds:
    """Family 1: point clouds with analytically known Betti numbers."""

    n_points: int
    n_points_by_generator: dict[str, int]
    noise_sigma: float
    ambient_noise: str
    circle: Circle
    two_circles: TwoCircles
    figure_eight: FigureEight
    torus: Torus
    sphere: Sphere
    klein_bottle: KleinBottle
    hopf_link: HopfLink


@dataclass(frozen=True)
class NetworkFromCloud:
    method: str
    kernel_sigma: float
    knn_k: int
    distance_from_similarity: str


@dataclass(frozen=True)
class CycleLifecycle:
    """Snapshot indices of the scheduled homological events."""

    path_closes_at: int
    chord_added_at: int
    edge_removed_at: int
    component_splits_at: int
    component_merges_at: int


@dataclass(frozen=True)
class Scenarios:
    cycle_lifecycle: CycleLifecycle


@dataclass(frozen=True)
class GeometricLoop:
    n_points: int
    radius_schedule: tuple[float, ...]
    gap_open_at: int
    gap_close_at: int


@dataclass(frozen=True)
class DynamicEvents:
    """Family 2: graph snapshots with a scheduled event log."""

    n_nodes: int
    n_snapshots: int
    edge_weight: float
    threshold: float
    scenarios: Scenarios
    geometric_loop: GeometricLoop


@dataclass(frozen=True)
class DynamicSbm:
    n_nodes: int
    n_snapshots: int
    block_sizes: tuple[int, ...]
    p_in: float
    p_out: float
    merge_at: int
    split_at: int


@dataclass(frozen=True)
class Kuramoto:
    K: float
    omega_dist: str
    omega_scale: float


@dataclass(frozen=True)
class Roessler:
    a: float
    b: float
    c: float
    epsilon: float


@dataclass(frozen=True)
class Integrator:
    method: str
    rtol: float
    atol: float


@dataclass(frozen=True)
class CouplingSchedule:
    regime_length: int
    regimes: tuple[str, ...]


@dataclass(frozen=True)
class Oscillators:
    """Family 3: coupled oscillators with a known coupling schedule."""

    model: str
    n_channels: int
    n_timepoints: int
    dt: float
    kuramoto: Kuramoto
    roessler: Roessler
    integrator: Integrator
    coupling_schedule: CouplingSchedule


@dataclass(frozen=True)
class SlidingWindow:
    length: int
    step: int
    fisher_z: bool
    max_length_fraction_of_regime: float


@dataclass(frozen=True)
class Noise:
    """Family 4: controlled noise levels, shared by all noisy datasets."""

    weight_jitter_sigma: tuple[float, ...]
    spurious_edge_fraction: tuple[float, ...]
    missing_edge_fraction: tuple[float, ...]
    point_jitter_sigma: tuple[float, ...]
    n_repeats: int


@dataclass(frozen=True)
class ErdosRenyi:
    match_density_to: str


@dataclass(frozen=True)
class RandomGeometric:
    n_nodes: int
    match_density_to: str


@dataclass(frozen=True)
class NullModels:
    erdos_renyi: ErdosRenyi
    random_geometric: RandomGeometric
    n_realizations: int


@dataclass(frozen=True)
class Naming:
    float_precision: int


@dataclass(frozen=True)
class Config:
    """The validated contents of ``config.yaml``.

    Attributes
    ----------
    config_path : pathlib.Path
        Absolute path of the file this was loaded from.
    output_root : pathlib.Path
        Absolute output root, resolved relative to ``config_path``'s directory rather
        than the working directory, so that a run from a notebook writes to the same
        place as a run from the repository root.
    """

    format_version: int
    seed: int
    output_root: Path
    config_path: Path
    topology_check: TopologyCheck
    static_clouds: StaticClouds
    network_from_cloud: NetworkFromCloud
    dynamic_events: DynamicEvents
    dynamic_sbm: DynamicSbm
    oscillators: Oscillators
    sliding_window: SlidingWindow
    noise: Noise
    null_models: NullModels
    naming: Naming


# --------------------------------------------------------------------------- loading


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the configuration.

    Parameters
    ----------
    path : pathlib.Path or str, optional
        Path to ``config.yaml``; defaults to ``config.yaml`` in the working directory.

    Returns
    -------
    Config
        The validated configuration.

    Raises
    ------
    ConfigError
        With a message naming the offending key, e.g.
        ``config.yaml: oscillators.dt: expected a float > 0, got 0``.
    """
    path = Path(path) if path is not None else Path.cwd() / DEFAULT_CONFIG_NAME
    if not path.exists():
        raise ConfigError(f"{path} does not exist")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return parse_config(raw, config_path=path)


def parse_config(raw: Mapping[str, Any], *, config_path: Path | str) -> Config:
    """Validate an already-parsed configuration mapping.

    Parameters
    ----------
    raw : mapping
        Contents of ``config.yaml``.
    config_path : pathlib.Path or str
        Path the mapping came from; used for error messages and to resolve
        ``output_root``.

    Returns
    -------
    Config
        The validated configuration.
    """
    config_path = Path(config_path)
    root = _Reader(raw, origin=config_path.name, path="")

    format_version = root.integer("format_version", minimum=1)
    if format_version != FORMAT_VERSION:
        raise ConfigError(
            f"{config_path.name}: format_version is {format_version}, "
            f"this package implements {FORMAT_VERSION}"
        )
    seed = root.integer("seed", minimum=0)
    output_root = (config_path.parent / root.string("output_root")).resolve()

    config = Config(
        format_version=format_version,
        seed=seed,
        output_root=output_root,
        config_path=config_path.resolve(),
        topology_check=_topology_check(root.section("topology_check")),
        static_clouds=_static_clouds(root.section("static_clouds")),
        network_from_cloud=_network_from_cloud(root.section("network_from_cloud")),
        dynamic_events=_dynamic_events(root.section("dynamic_events")),
        dynamic_sbm=_dynamic_sbm(root.section("dynamic_sbm")),
        oscillators=_oscillators(root.section("oscillators")),
        sliding_window=_sliding_window(root.section("sliding_window")),
        noise=_noise(root.section("noise")),
        null_models=_null_models(root.section("null_models")),
        naming=_naming(root.section("naming")),
    )
    root.done()
    _check_cross_section(config, origin=config_path.name)
    return config


def _topology_check(reader: _Reader) -> TopologyCheck:
    section = TopologyCheck(
        # docs/task.md fixes the maximum homology dimension at 2.
        max_homology_dim=reader.integer("max_homology_dim", minimum=0, maximum=2),
        coeff_field=reader.integer("coeff_field", minimum=2),
        coeff_field_test_values=reader.integer_list(
            "coeff_field_test_values", minimum=2, min_length=1
        ),
        # The significance criterion is under revision (docs/validation.md §2); the old
        # diam_ratio rule is kept only so that the two can be compared.
        criterion=reader.string("criterion", choices=("gap", "diam_ratio")),
        # Strictly inside (0, 1): at 0 every noise bar counts as significant, at 1 only
        # a bar as long as the diameter would.
        significance_ratio=reader.number(
            "significance_ratio", minimum=0.0, maximum=1.0, exclusive=True
        ),
        gap_noise_floor_ratio=reader.number(
            "gap_noise_floor_ratio", minimum=0.0, maximum=1.0, exclusive=True
        ),
        # A gap of 1.0 would mean "no gap at all", so the minimum useful ratio is above 1.
        gap_min_ratio=reader.number("gap_min_ratio", minimum=1.0, exclusive=True),
        calibration_seeds=reader.integer_list("calibration_seeds", minimum=0, min_length=1),
        reference_max_nodes=reader.integer("reference_max_nodes", minimum=1),
        # Cross-check of paths A and B on family 1 (docs/validation.md §1.1): both
        # implementations measure the same complex, and agreement rather than the
        # expectation table is what the test asserts.
        cross_check_n_points=reader.integer("cross_check_n_points", minimum=2),
        cross_check_n_scales=reader.integer("cross_check_n_scales", minimum=1),
    )
    reader.done()
    for field in (section.coeff_field, *section.coeff_field_test_values):
        if not _is_prime(field):
            raise ConfigError(
                f"{reader.origin}: topology_check coefficient fields must be prime "
                f"(ripser works over Z/p), got {field}"
            )
    values = section.coeff_field_test_values
    if any(later <= earlier for earlier, later in zip(values, values[1:])):
        raise ConfigError(
            f"{reader.origin}: topology_check.coeff_field_test_values must be strictly "
            f"increasing, got {list(values)}"
        )
    if section.coeff_field not in values:
        raise ConfigError(
            f"{reader.origin}: topology_check.coeff_field ({section.coeff_field}) is not among "
            f"coeff_field_test_values ({list(values)}); the field the project actually works "
            "over must be one of the fields the tests exercise"
        )
    return section


def _static_clouds(reader: _Reader) -> StaticClouds:
    n_points = reader.integer("n_points", minimum=1)
    # Surfaces need more points than curves: at 400 the gap in H1 of the torus lands on
    # the wrong position (docs/validation.md §3).
    n_points_by_generator = reader.integer_mapping(
        "n_points_by_generator", minimum=1, allowed_keys=_STATIC_CLOUD_GENERATORS
    )
    noise_sigma = reader.number("noise_sigma", minimum=0.0)
    ambient_noise = reader.string("ambient_noise", choices=("gaussian",))

    circle_reader = reader.section("circle")
    circle = Circle(radius=circle_reader.number("radius", minimum=0.0, exclusive=True))
    circle_reader.done()

    two_reader = reader.section("two_circles")
    two_circles = TwoCircles(
        radius=two_reader.number("radius", minimum=0.0, exclusive=True),
        separation=two_reader.number("separation", minimum=0.0, exclusive=True),
    )
    two_reader.done()

    eight_reader = reader.section("figure_eight")
    figure_eight = FigureEight(scale=eight_reader.number("scale", minimum=0.0, exclusive=True))
    eight_reader.done()

    torus_reader = reader.section("torus")
    torus = Torus(
        R=torus_reader.number("R", minimum=0.0, exclusive=True),
        r=torus_reader.number("r", minimum=0.0, exclusive=True),
        sampling=torus_reader.string("sampling", choices=("rejection", "naive")),
    )
    torus_reader.done()

    sphere_reader = reader.section("sphere")
    sphere = Sphere(
        radius=sphere_reader.number("radius", minimum=0.0, exclusive=True),
        sampling=sphere_reader.string("sampling", choices=("muller", "naive_angles")),
    )
    sphere_reader.done()

    klein_reader = reader.section("klein_bottle")
    klein_bottle = KleinBottle(
        # R^3 admits only an immersion with a self-intersection, which destroys the
        # ground truth (docs/method.md §1.1), so the option is rejected outright.
        embedding=klein_reader.string("embedding", choices=("r4_tube",)),
        a=klein_reader.number("a", minimum=0.0, exclusive=True),
        b=klein_reader.number("b", minimum=0.0, exclusive=True),
        sampling=klein_reader.string("sampling", choices=("rejection", "naive")),
    )
    klein_reader.done()
    if klein_bottle.a <= klein_bottle.b:
        raise ConfigError(
            f"{reader.origin}: static_clouds.klein_bottle requires a > b, got "
            f"a={klein_bottle.a}, b={klein_bottle.b}; otherwise the factor (a + b cos v) "
            "reaches zero and the surface degenerates (docs/method.md §1.1)"
        )

    hopf_reader = reader.section("hopf_link")
    hopf_link = HopfLink(
        radius=hopf_reader.number("radius", minimum=0.0, exclusive=True),
        offset=hopf_reader.number("offset", minimum=0.0, exclusive=True),
    )
    hopf_reader.done()

    reader.done()
    return StaticClouds(
        n_points=n_points,
        n_points_by_generator=n_points_by_generator,
        noise_sigma=noise_sigma,
        ambient_noise=ambient_noise,
        circle=circle,
        two_circles=two_circles,
        figure_eight=figure_eight,
        torus=torus,
        sphere=sphere,
        klein_bottle=klein_bottle,
        hopf_link=hopf_link,
    )


def _network_from_cloud(reader: _Reader) -> NetworkFromCloud:
    section = NetworkFromCloud(
        method=reader.string("method", choices=("gaussian_kernel", "knn")),
        kernel_sigma=reader.number("kernel_sigma", minimum=0.0, exclusive=True),
        knn_k=reader.integer("knn_k", minimum=1),
        distance_from_similarity=reader.string(
            "distance_from_similarity", choices=("sqrt_neg_log", "one_minus")
        ),
    )
    reader.done()
    return section


def _dynamic_events(reader: _Reader) -> DynamicEvents:
    n_nodes = reader.integer("n_nodes", minimum=3)
    n_snapshots = reader.integer("n_snapshots", minimum=1)
    edge_weight = reader.number("edge_weight", minimum=0.0, exclusive=True)
    threshold = reader.number("threshold", minimum=0.0)

    scenarios_reader = reader.section("scenarios")
    lifecycle_reader = scenarios_reader.section("cycle_lifecycle")
    cycle_lifecycle = CycleLifecycle(
        path_closes_at=lifecycle_reader.integer("path_closes_at", minimum=0),
        chord_added_at=lifecycle_reader.integer("chord_added_at", minimum=0),
        edge_removed_at=lifecycle_reader.integer("edge_removed_at", minimum=0),
        component_splits_at=lifecycle_reader.integer("component_splits_at", minimum=0),
        component_merges_at=lifecycle_reader.integer("component_merges_at", minimum=0),
    )
    lifecycle_reader.done()
    scenarios_reader.done()

    loop_reader = reader.section("geometric_loop")
    geometric_loop = GeometricLoop(
        n_points=loop_reader.integer("n_points", minimum=3),
        radius_schedule=loop_reader.number_list(
            "radius_schedule", minimum=0.0, exclusive=True, min_length=1
        ),
        gap_open_at=loop_reader.integer("gap_open_at", minimum=0),
        gap_close_at=loop_reader.integer("gap_close_at", minimum=0),
    )
    loop_reader.done()

    reader.done()
    if threshold >= edge_weight:
        raise ConfigError(
            f"{reader.origin}: dynamic_events.threshold ({threshold}) must be below "
            f"edge_weight ({edge_weight}), otherwise no edge ever enters the complex"
        )
    return DynamicEvents(
        n_nodes=n_nodes,
        n_snapshots=n_snapshots,
        edge_weight=edge_weight,
        threshold=threshold,
        scenarios=Scenarios(cycle_lifecycle=cycle_lifecycle),
        geometric_loop=geometric_loop,
    )


def _dynamic_sbm(reader: _Reader) -> DynamicSbm:
    section = DynamicSbm(
        n_nodes=reader.integer("n_nodes", minimum=1),
        n_snapshots=reader.integer("n_snapshots", minimum=1),
        block_sizes=reader.integer_list("block_sizes", minimum=1, min_length=1),
        p_in=reader.number("p_in", minimum=0.0, maximum=1.0),
        p_out=reader.number("p_out", minimum=0.0, maximum=1.0),
        merge_at=reader.integer("merge_at", minimum=0),
        split_at=reader.integer("split_at", minimum=0),
    )
    reader.done()
    if section.p_out >= section.p_in:
        raise ConfigError(
            f"{reader.origin}: dynamic_sbm.p_out ({section.p_out}) must be below "
            f"p_in ({section.p_in}), otherwise the blocks are not communities"
        )
    for key in ("merge_at", "split_at"):
        if getattr(section, key) >= section.n_snapshots:
            raise ConfigError(
                f"{reader.origin}: dynamic_sbm.{key} ({getattr(section, key)}) must be "
                f"below n_snapshots ({section.n_snapshots})"
            )
    return section


def _oscillators(reader: _Reader) -> Oscillators:
    model = reader.string("model", choices=("kuramoto", "roessler"))
    n_channels = reader.integer("n_channels", minimum=2)
    n_timepoints = reader.integer("n_timepoints", minimum=2)
    dt = reader.number("dt", minimum=0.0, exclusive=True)

    kuramoto_reader = reader.section("kuramoto")
    kuramoto = Kuramoto(
        K=kuramoto_reader.number("K", minimum=0.0),
        omega_dist=kuramoto_reader.string("omega_dist", choices=("normal",)),
        omega_scale=kuramoto_reader.number("omega_scale", minimum=0.0),
    )
    kuramoto_reader.done()

    roessler_reader = reader.section("roessler")
    roessler = Roessler(
        a=roessler_reader.number("a"),
        b=roessler_reader.number("b"),
        c=roessler_reader.number("c"),
        epsilon=roessler_reader.number("epsilon", minimum=0.0),
    )
    roessler_reader.done()

    integrator_reader = reader.section("integrator")
    integrator = Integrator(
        method=integrator_reader.string("method", choices=("RK45", "LSODA")),
        rtol=integrator_reader.number("rtol", minimum=0.0, exclusive=True),
        atol=integrator_reader.number("atol", minimum=0.0, exclusive=True),
    )
    integrator_reader.done()

    schedule_reader = reader.section("coupling_schedule")
    coupling_schedule = CouplingSchedule(
        regime_length=schedule_reader.integer("regime_length", minimum=1),
        regimes=schedule_reader.string_list("regimes", min_length=1),
    )
    schedule_reader.done()

    reader.done()
    return Oscillators(
        model=model,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        dt=dt,
        kuramoto=kuramoto,
        roessler=roessler,
        integrator=integrator,
        coupling_schedule=coupling_schedule,
    )


def _sliding_window(reader: _Reader) -> SlidingWindow:
    section = SlidingWindow(
        # Two points are the minimum a Pearson correlation is defined on.
        length=reader.integer("length", minimum=2),
        step=reader.integer("step", minimum=1),
        fisher_z=reader.boolean("fisher_z"),
        max_length_fraction_of_regime=reader.number(
            "max_length_fraction_of_regime", minimum=0.0, maximum=1.0, exclusive=True
        ),
    )
    reader.done()
    if section.step > section.length:
        raise ConfigError(
            f"{reader.origin}: sliding_window.step ({section.step}) exceeds length "
            f"({section.length}), which would skip part of the series"
        )
    return section


def _noise(reader: _Reader) -> Noise:
    levels = {}
    for key in (
        "weight_jitter_sigma",
        "spurious_edge_fraction",
        "missing_edge_fraction",
        "point_jitter_sigma",
    ):
        levels[key] = reader.number_list(key, minimum=0.0, min_length=1)
    n_repeats = reader.integer("n_repeats", minimum=1)
    reader.done()

    for key, values in levels.items():
        # The stability tests compare diagrams across levels and need the clean case
        # first and the levels ordered, otherwise "monotone in the noise level" has
        # no meaning.
        if values[0] != 0.0:
            raise ConfigError(
                f"{reader.origin}: noise.{key} must start at 0.0 (the clean reference), "
                f"got {values[0]}"
            )
        if any(later <= earlier for earlier, later in zip(values, values[1:])):
            raise ConfigError(f"{reader.origin}: noise.{key} must be strictly increasing: {values}")
    for key in ("spurious_edge_fraction", "missing_edge_fraction"):
        if levels[key][-1] > 1.0:
            raise ConfigError(f"{reader.origin}: noise.{key} holds a fraction above 1.0")
    return Noise(n_repeats=n_repeats, **levels)


def _null_models(reader: _Reader) -> NullModels:
    er_reader = reader.section("erdos_renyi")
    erdos_renyi = ErdosRenyi(
        match_density_to=er_reader.string("match_density_to", choices=_STATIC_CLOUD_GENERATORS)
    )
    er_reader.done()

    rg_reader = reader.section("random_geometric")
    random_geometric = RandomGeometric(
        n_nodes=rg_reader.integer("n_nodes", minimum=1),
        match_density_to=rg_reader.string("match_density_to", choices=_STATIC_CLOUD_GENERATORS),
    )
    rg_reader.done()

    n_realizations = reader.integer("n_realizations", minimum=1)
    reader.done()
    return NullModels(
        erdos_renyi=erdos_renyi,
        random_geometric=random_geometric,
        n_realizations=n_realizations,
    )


def _naming(reader: _Reader) -> Naming:
    section = Naming(float_precision=reader.integer("float_precision", minimum=1, maximum=17))
    reader.done()
    return section


def _check_cross_section(config: Config, *, origin: str) -> None:
    """Checks that span sections; each one restates a constraint from ``docs/``."""
    clouds = config.static_clouds
    if clouds.two_circles.separation <= 2 * clouds.two_circles.radius:
        raise ConfigError(
            f"{origin}: static_clouds.two_circles.separation "
            f"({clouds.two_circles.separation}) must exceed 2 * radius "
            f"({2 * clouds.two_circles.radius}), otherwise the two components touch "
            "and beta_0 is not 2"
        )
    if clouds.torus.R <= clouds.torus.r:
        raise ConfigError(
            f"{origin}: static_clouds.torus requires R > r, got R={clouds.torus.R}, "
            f"r={clouds.torus.r}"
        )
    # The second circle passes through the disc of the first exactly when its
    # centre offset lies strictly between 0 and 2r; otherwise the circles either
    # intersect or are not linked at all (docs/method.md §1.1).
    if not 0.0 < clouds.hopf_link.offset < 2 * clouds.hopf_link.radius:
        raise ConfigError(
            f"{origin}: static_clouds.hopf_link.offset ({clouds.hopf_link.offset}) must lie "
            f"strictly between 0 and 2 * radius ({2 * clouds.hopf_link.radius}), "
            "otherwise the circles are not linked"
        )

    topology = config.topology_check
    if topology.cross_check_n_points > topology.reference_max_nodes:
        raise ConfigError(
            f"{origin}: topology_check.cross_check_n_points ({topology.cross_check_n_points}) "
            f"exceeds reference_max_nodes ({topology.reference_max_nodes}); the reference "
            "implementation would refuse the subsample"
        )
    smallest_cloud = min(
        [config.static_clouds.n_points, *config.static_clouds.n_points_by_generator.values()]
    )
    if topology.cross_check_n_points > smallest_cloud:
        raise ConfigError(
            f"{origin}: topology_check.cross_check_n_points ({topology.cross_check_n_points}) "
            f"exceeds the smallest configured cloud ({smallest_cloud}); there is "
            "nothing to subsample from"
        )

    if config.dynamic_events.n_nodes > config.topology_check.reference_max_nodes:
        raise ConfigError(
            f"{origin}: dynamic_events.n_nodes ({config.dynamic_events.n_nodes}) exceeds "
            f"topology_check.reference_max_nodes ({config.topology_check.reference_max_nodes}); "
            "the reference implementation could not verify the Betti curve"
        )

    lifecycle = config.dynamic_events.scenarios.cycle_lifecycle
    schedule = [
        ("path_closes_at", lifecycle.path_closes_at),
        ("chord_added_at", lifecycle.chord_added_at),
        ("edge_removed_at", lifecycle.edge_removed_at),
        ("component_splits_at", lifecycle.component_splits_at),
        ("component_merges_at", lifecycle.component_merges_at),
    ]
    for (earlier_key, earlier), (later_key, later) in zip(schedule, schedule[1:]):
        if later <= earlier:
            raise ConfigError(
                f"{origin}: dynamic_events.scenarios.cycle_lifecycle.{later_key} ({later}) "
                f"must come after {earlier_key} ({earlier}); events sharing a snapshot "
                "cannot be told apart in the Betti curve"
            )
    if schedule[-1][1] >= config.dynamic_events.n_snapshots:
        raise ConfigError(
            f"{origin}: dynamic_events.scenarios.cycle_lifecycle.{schedule[-1][0]} "
            f"({schedule[-1][1]}) is outside n_snapshots ({config.dynamic_events.n_snapshots})"
        )

    loop = config.dynamic_events.geometric_loop
    if not loop.gap_open_at < loop.gap_close_at < config.dynamic_events.n_snapshots:
        raise ConfigError(
            f"{origin}: dynamic_events.geometric_loop requires "
            f"gap_open_at < gap_close_at < n_snapshots, got {loop.gap_open_at}, "
            f"{loop.gap_close_at}, {config.dynamic_events.n_snapshots}"
        )

    if sum(config.dynamic_sbm.block_sizes) != config.dynamic_sbm.n_nodes:
        raise ConfigError(
            f"{origin}: dynamic_sbm.block_sizes sum to "
            f"{sum(config.dynamic_sbm.block_sizes)} but n_nodes is {config.dynamic_sbm.n_nodes}"
        )

    # docs/method.md §3.4: a window longer than a regime smears the event across the
    # regime boundary and it can no longer be localised.
    regime_length = config.oscillators.coupling_schedule.regime_length
    allowed = config.sliding_window.max_length_fraction_of_regime * regime_length
    if config.sliding_window.length > allowed:
        raise ConfigError(
            f"{origin}: sliding_window.length ({config.sliding_window.length}) exceeds "
            f"max_length_fraction_of_regime * regime_length ({allowed}); "
            "the window must be much shorter than a coupling regime"
        )


# --------------------------------------------------------------------------- reader


class _Reader:
    """A mapping reader that tracks which keys were consumed and where it is."""

    def __init__(self, data: Mapping[str, Any], *, origin: str, path: str) -> None:
        self.origin = origin
        self.path = path
        if not isinstance(data, Mapping):
            raise ConfigError(f"{self._where()}: expected a mapping, got {_type_name(data)}")
        self._data = data
        self._seen: set[str] = set()

    def section(self, key: str) -> _Reader:
        value = self._take(key)
        child_path = f"{self.path}.{key}" if self.path else key
        return _Reader(value, origin=self.origin, path=child_path)

    def integer(self, key: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
        value = self._take(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{self._where(key)}: expected an int, got {_type_name(value)}")
        self._check_range(key, value, minimum, maximum, exclusive=False)
        return value

    def number(
        self,
        key: str,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive: bool = False,
    ) -> float:
        value = self._take(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self._where(key)}: expected a number, got {_type_name(value)}")
        value = float(value)
        self._check_range(key, value, minimum, maximum, exclusive=exclusive)
        return value

    def string(self, key: str, *, choices: Sequence[str] | None = None) -> str:
        value = self._take(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(
                f"{self._where(key)}: expected a non-empty string, got {_type_name(value)}"
            )
        if choices is not None and value not in choices:
            raise ConfigError(
                f"{self._where(key)}: expected one of {list(choices)}, got {value!r}"
            )
        return value

    def boolean(self, key: str) -> bool:
        value = self._take(key)
        if not isinstance(value, bool):
            raise ConfigError(f"{self._where(key)}: expected a bool, got {_type_name(value)}")
        return value

    def integer_list(
        self, key: str, *, minimum: int | None = None, min_length: int = 0
    ) -> tuple[int, ...]:
        values = self._take_sequence(key, min_length)
        result = []
        for index, item in enumerate(values):
            if not isinstance(item, int) or isinstance(item, bool):
                raise ConfigError(
                    f"{self._where(key)}[{index}]: expected an int, got {_type_name(item)}"
                )
            if minimum is not None and item < minimum:
                raise ConfigError(f"{self._where(key)}[{index}]: expected >= {minimum}, got {item}")
            result.append(item)
        return tuple(result)

    def number_list(
        self,
        key: str,
        *,
        minimum: float | None = None,
        exclusive: bool = False,
        min_length: int = 0,
    ) -> tuple[float, ...]:
        values = self._take_sequence(key, min_length)
        result = []
        for index, item in enumerate(values):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ConfigError(
                    f"{self._where(key)}[{index}]: expected a number, got {_type_name(item)}"
                )
            item = float(item)
            if minimum is not None and (item <= minimum if exclusive else item < minimum):
                bound = "> " if exclusive else ">= "
                raise ConfigError(
                    f"{self._where(key)}[{index}]: expected {bound}{minimum}, got {item}"
                )
            result.append(item)
        return tuple(result)

    def integer_mapping(
        self, key: str, *, minimum: int | None = None, allowed_keys: Sequence[str]
    ) -> dict[str, int]:
        """Read a mapping of names to ints, rejecting names outside ``allowed_keys``."""
        value = self._take(key)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{self._where(key)}: expected a mapping, got {_type_name(value)}")
        unknown = set(value) - set(allowed_keys)
        if unknown:
            raise ConfigError(
                f"{self._where(key)}: unknown names {sorted(unknown)}; "
                f"expected any of {list(allowed_keys)}"
            )
        result = {}
        for name, item in value.items():
            if not isinstance(item, int) or isinstance(item, bool):
                raise ConfigError(
                    f"{self._where(key)}.{name}: expected an int, got {_type_name(item)}"
                )
            if minimum is not None and item < minimum:
                raise ConfigError(
                    f"{self._where(key)}.{name}: expected >= {minimum}, got {item}"
                )
            result[name] = item
        return result

    def string_list(self, key: str, *, min_length: int = 0) -> tuple[str, ...]:
        values = self._take_sequence(key, min_length)
        for index, item in enumerate(values):
            if not isinstance(item, str) or not item:
                raise ConfigError(
                    f"{self._where(key)}[{index}]: expected a non-empty string, "
                    f"got {_type_name(item)}"
                )
        return tuple(values)

    def done(self) -> None:
        """Raise if the mapping held keys nobody asked for."""
        unknown = set(self._data) - self._seen
        if unknown:
            raise ConfigError(
                f"{self._where()}: unknown keys {sorted(unknown)}. A key that nothing reads "
                "is either a typo or a parameter the code ignores; both are silent bugs."
            )

    def _take(self, key: str) -> Any:
        self._seen.add(key)
        try:
            return self._data[key]
        except KeyError:
            raise ConfigError(f"{self._where()}: missing required key {key!r}") from None

    def _take_sequence(self, key: str, min_length: int) -> list[Any]:
        value = self._take(key)
        if not isinstance(value, list):
            raise ConfigError(f"{self._where(key)}: expected a list, got {_type_name(value)}")
        if len(value) < min_length:
            raise ConfigError(
                f"{self._where(key)}: expected at least {min_length} items, got {len(value)}"
            )
        return value

    def _check_range(
        self,
        key: str,
        value: float,
        minimum: float | None,
        maximum: float | None,
        *,
        exclusive: bool,
    ) -> None:
        if minimum is not None and (value <= minimum if exclusive else value < minimum):
            bound = "> " if exclusive else ">= "
            raise ConfigError(f"{self._where(key)}: expected {bound}{minimum}, got {value}")
        if maximum is not None and (value >= maximum if exclusive else value > maximum):
            bound = "< " if exclusive else "<= "
            raise ConfigError(f"{self._where(key)}: expected {bound}{maximum}, got {value}")

    def _where(self, key: str | None = None) -> str:
        path = self.path
        if key is not None:
            path = f"{path}.{key}" if path else key
        return f"{self.origin}: {path}" if path else self.origin


def _type_name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    factor = 2
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 1
    return True
