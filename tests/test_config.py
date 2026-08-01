"""Validation of ``config.yaml``: required keys, types, ranges, cross-section rules."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tdadyn.synthetic import load_config, parse_config
from tdadyn.synthetic.config import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def mutate(raw: dict, dotted: str, value) -> dict:
    """Return a copy of ``raw`` with one dotted key set to ``value``."""
    result = copy.deepcopy(raw)
    node = result
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    node[leaf] = value
    return result


def drop(raw: dict, dotted: str) -> dict:
    """Return a copy of ``raw`` with one dotted key removed."""
    result = copy.deepcopy(raw)
    node = result
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node[key]
    del node[leaf]
    return result


def test_repository_config_is_valid():
    config = load_config(CONFIG_PATH)
    assert config.seed == 20260201
    assert config.topology_check.coeff_field == 2
    assert config.static_clouds.torus.sampling == "rejection"
    assert config.dynamic_events.scenarios.cycle_lifecycle.path_closes_at == 12


def test_output_root_resolves_against_the_config_file_not_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(CONFIG_PATH)
    assert config.output_root == REPO_ROOT / "data" / "synthetic"
    assert config.output_root.is_absolute()


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "dotted",
    [
        "seed",
        "topology_check.coeff_field",
        "static_clouds.torus.R",
        "oscillators.integrator.rtol",
        "naming.float_precision",
        "sliding_window.max_length_fraction_of_regime",
    ],
)
def test_missing_required_key_names_the_key(raw, dotted):
    with pytest.raises(ConfigError, match=dotted.split(".")[-1]):
        parse_config(drop(raw, dotted), config_path=CONFIG_PATH)


@pytest.mark.parametrize(
    ("dotted", "value", "message"),
    [
        ("seed", "20260201", "expected an int"),
        ("static_clouds.n_points", 1.5, "expected an int"),
        ("static_clouds.torus.R", "two", "expected a number"),
        ("sliding_window.fisher_z", "yes", "expected a bool"),
        ("dynamic_sbm.block_sizes", 60, "expected a list"),
        ("oscillators.coupling_schedule.regimes", [1, 2], "expected a non-empty string"),
        ("topology_check", [1, 2], "expected a mapping"),
    ],
)
def test_wrong_type_is_reported_with_its_path(raw, dotted, value, message):
    with pytest.raises(ConfigError) as excinfo:
        parse_config(mutate(raw, dotted, value), config_path=CONFIG_PATH)
    assert message in str(excinfo.value)
    assert dotted.split(".")[-1] in str(excinfo.value)


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("seed", -1),
        ("topology_check.max_homology_dim", 3),
        ("topology_check.significance_ratio", 0.0),
        ("topology_check.significance_ratio", 1.0),
        ("topology_check.reference_max_nodes", 0),
        ("static_clouds.n_points", 0),
        ("static_clouds.noise_sigma", -0.1),
        ("dynamic_sbm.p_in", 1.5),
        ("oscillators.dt", 0.0),
        ("naming.float_precision", 0),
        ("sliding_window.step", 0),
    ],
)
def test_out_of_range_value_is_rejected(raw, dotted, value):
    with pytest.raises(ConfigError):
        parse_config(mutate(raw, dotted, value), config_path=CONFIG_PATH)


@pytest.mark.parametrize(
    ("dotted", "value"),
    [
        ("static_clouds.torus.sampling", "uniform_angles"),
        ("static_clouds.klein_bottle.embedding", "r3"),
        ("network_from_cloud.distance_from_similarity", "exp"),
        ("oscillators.model", "lorenz"),
        ("oscillators.integrator.method", "Euler"),
        ("null_models.erdos_renyi.match_density_to", "moebius"),
    ],
)
def test_value_outside_the_allowed_set_is_rejected(raw, dotted, value):
    with pytest.raises(ConfigError, match="expected one of"):
        parse_config(mutate(raw, dotted, value), config_path=CONFIG_PATH)


def test_unknown_key_is_an_error_rather_than_ignored(raw):
    # A key nothing reads is either a typo or a parameter the code silently ignores.
    polluted = copy.deepcopy(raw)
    polluted["static_clouds"]["noise_sgima"] = 0.5
    with pytest.raises(ConfigError, match="noise_sgima"):
        parse_config(polluted, config_path=CONFIG_PATH)


def test_unknown_top_level_key_is_an_error(raw):
    with pytest.raises(ConfigError, match="whatever"):
        parse_config({**raw, "whatever": 1}, config_path=CONFIG_PATH)


def test_format_version_must_match_the_package(raw):
    with pytest.raises(ConfigError, match="format_version"):
        parse_config(mutate(raw, "format_version", 2), config_path=CONFIG_PATH)


# --- cross-section constraints, each restating a rule from docs/ ------------------


def test_two_circles_must_not_touch(raw):
    radius = raw["static_clouds"]["two_circles"]["radius"]
    with pytest.raises(ConfigError, match="separation"):
        parse_config(
            mutate(raw, "static_clouds.two_circles.separation", 2 * radius),
            config_path=CONFIG_PATH,
        )


def test_torus_requires_major_radius_above_minor(raw):
    with pytest.raises(ConfigError, match="R > r"):
        parse_config(mutate(raw, "static_clouds.torus.r", 3.0), config_path=CONFIG_PATH)


@pytest.mark.parametrize("offset", [0.0001, 2.5])
def test_hopf_link_offset_must_keep_the_circles_linked(raw, offset):
    # radius is 1.0, so the circles are linked only for 0 < offset < 2.
    if offset >= 2 * raw["static_clouds"]["hopf_link"]["radius"]:
        with pytest.raises(ConfigError, match="not linked"):
            parse_config(mutate(raw, "static_clouds.hopf_link.offset", offset),
                         config_path=CONFIG_PATH)
    else:
        parse_config(mutate(raw, "static_clouds.hopf_link.offset", offset),
                     config_path=CONFIG_PATH)


def test_coefficient_field_must_be_prime(raw):
    with pytest.raises(ConfigError, match="prime"):
        parse_config(mutate(raw, "topology_check.coeff_field", 4), config_path=CONFIG_PATH)


def test_dynamic_graph_must_fit_the_reference_implementation(raw):
    limit = raw["topology_check"]["reference_max_nodes"]
    with pytest.raises(ConfigError, match="reference_max_nodes"):
        parse_config(mutate(raw, "dynamic_events.n_nodes", limit + 1), config_path=CONFIG_PATH)


def test_scheduled_events_must_be_strictly_ordered(raw):
    lifecycle = "dynamic_events.scenarios.cycle_lifecycle"
    path_closes = raw["dynamic_events"]["scenarios"]["cycle_lifecycle"]["path_closes_at"]
    with pytest.raises(ConfigError, match="chord_added_at"):
        parse_config(
            mutate(raw, f"{lifecycle}.chord_added_at", path_closes), config_path=CONFIG_PATH
        )


def test_scheduled_events_must_fit_in_the_snapshots(raw):
    n_snapshots = raw["dynamic_events"]["n_snapshots"]
    with pytest.raises(ConfigError, match="outside n_snapshots"):
        parse_config(
            mutate(
                raw,
                "dynamic_events.scenarios.cycle_lifecycle.component_merges_at",
                n_snapshots,
            ),
            config_path=CONFIG_PATH,
        )


def test_geometric_loop_gap_must_open_before_it_closes(raw):
    with pytest.raises(ConfigError, match="gap_open_at < gap_close_at"):
        parse_config(
            mutate(raw, "dynamic_events.geometric_loop.gap_open_at", 50),
            config_path=CONFIG_PATH,
        )


def test_sbm_block_sizes_must_sum_to_the_node_count(raw):
    with pytest.raises(ConfigError, match="block_sizes"):
        parse_config(mutate(raw, "dynamic_sbm.block_sizes", [15, 15, 15]), config_path=CONFIG_PATH)


def test_window_must_be_much_shorter_than_a_regime(raw):
    # docs/method.md §3.4: a longer window smears the event across the regime boundary.
    regime = raw["oscillators"]["coupling_schedule"]["regime_length"]
    fraction = raw["sliding_window"]["max_length_fraction_of_regime"]
    too_long = int(fraction * regime) + 1
    with pytest.raises(ConfigError, match="much shorter than a coupling regime"):
        parse_config(mutate(raw, "sliding_window.length", too_long), config_path=CONFIG_PATH)


def test_noise_levels_must_start_clean_and_increase(raw):
    with pytest.raises(ConfigError, match="must start at 0.0"):
        parse_config(
            mutate(raw, "noise.weight_jitter_sigma", [0.05, 0.1]), config_path=CONFIG_PATH
        )
    with pytest.raises(ConfigError, match="strictly increasing"):
        parse_config(
            mutate(raw, "noise.weight_jitter_sigma", [0.0, 0.2, 0.1]), config_path=CONFIG_PATH
        )


def test_edge_threshold_must_admit_the_edge_weight(raw):
    weight = raw["dynamic_events"]["edge_weight"]
    with pytest.raises(ConfigError, match="threshold"):
        parse_config(mutate(raw, "dynamic_events.threshold", weight), config_path=CONFIG_PATH)
