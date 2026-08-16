"""Build every dataset from ``config.yaml`` and register it in ``manifest.json``.

    python -m tdadyn.synthetic.build --config config.yaml [--family ...]

The datasets are a deterministic function of the configuration and the seed, so this is a
rebuild rather than an accumulation: the manifest is replaced, not appended to.  A stale
row would otherwise describe a directory that no longer exists, and the reproducibility
check would compare against a registry nobody has.

**What reaches disk.**  Families 1 to 3 write one dataset each.  Family 4 writes only
``repeat = 0`` of each mechanism and level: the averaging over repeats is what the tests do,
through ``rng_for(seed, stream, repeat)``, and materialising every repeat would multiply the
tree without adding an object anyone reads.  Every perturbed dataset stores the perturbed
array alone and names its source in ``noise_level["base"]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
)
from .config import Config, load_config
from .contracts import Dataset
from .io import save_dataset
from .manifest import ManifestEntry, make_entry, write_manifest
from .naming import make_dataset_id
from .noise import make_jittered_cloud, make_jittered_network, make_noisy_snapshots
from .rng import rng_for

FAMILY_CHOICES = ("static_cloud", "dynamic_graph", "series", "noisy", "null")

NOISE_BASE_CLOUD = "circle"
"""Family-1 object the cloud noise is built over.

One exemplar is enough on disk: the stability curve of result 6 is computed over repeats
in the tests and over several objects, not read back from here.  Graph noise uses *both*
family-2 scenarios instead, because their edge rules point in opposite directions and a
single one would exercise only half the arithmetic."""


def build_all(config: Config, families: tuple[str, ...]) -> list[ManifestEntry]:
    """Build the requested families and return their manifest entries.

    Parameters
    ----------
    config : Config
        Loaded configuration.
    families : tuple of str
        Subset of :data:`FAMILY_CHOICES`.

    Returns
    -------
    list of ManifestEntry
        One entry per written dataset, in write order.
    """
    entries: list[ManifestEntry] = []
    precision = config.naming.float_precision

    def write(dataset: Dataset) -> None:
        family = dataset.ground_truth.family
        directory = save_dataset(
            dataset, config.output_root / family, float_precision=precision, overwrite=True
        )
        entries.append(make_entry(directory, config.output_root, dataset.ground_truth))

    # Bases are built once and reused: family 4 sits on top of them, and rebuilding a base
    # per noisy dataset would re-run the reference implementation dozens of times.
    clouds = {name: build_static_cloud(name, config) for name in STATIC_GENERATORS}
    graphs = {name: build_dynamic_graph(name, config) for name in DYNAMIC_GENERATORS}

    if "static_cloud" in families:
        for dataset in clouds.values():
            write(dataset)

    if "dynamic_graph" in families:
        for dataset in graphs.values():
            write(dataset)
        write(build_dynamic_sbm(config))

    if "series" in families:
        for name in SERIES_GENERATORS:
            write(build_series(name, config))

    if "noisy" in families:
        for dataset in _noisy_datasets(config, clouds, graphs):
            write(dataset)

    if "null" in families:
        write(build_null_cloud(config))
        write(build_null_graph(config, graphs["cycle_lifecycle"]))

    return entries


def _noisy_datasets(config: Config, clouds: dict, graphs: dict) -> list[Dataset]:
    """Every family-4 perturbation of the built bases, at ``repeat = 0``."""
    noise = config.noise
    precision = config.naming.float_precision
    datasets: list[Dataset] = []

    base = clouds[NOISE_BASE_CLOUD]
    base_id = make_dataset_id(base.ground_truth, float_precision=precision)
    network = build_network(base.points, config)

    for level in noise.point_jitter_sigma:
        stream = rng_for(config.seed, noise_stream("point_jitter", NOISE_BASE_CLOUD), 0)
        datasets.append(make_jittered_cloud(base.points, base_id, level, 0, stream))
    for level in noise.weight_jitter_sigma:
        stream = rng_for(config.seed, noise_stream("weight_jitter", NOISE_BASE_CLOUD), 0)
        datasets.append(
            make_jittered_network(network, base_id, level, GAUSSIAN_KERNEL_RANGE, 0, stream)
        )

    for name, graph in graphs.items():
        truth = graph.ground_truth
        graph_id = make_dataset_id(truth, float_precision=precision)
        for kind, levels in (
            ("spurious_edges", noise.spurious_edge_fraction),
            ("missing_edges", noise.missing_edge_fraction),
        ):
            for level in levels:
                stream = rng_for(config.seed, noise_stream(kind, name), 0)
                datasets.append(
                    make_noisy_snapshots(
                        graph.snapshots,
                        graph_id,
                        kind,
                        level,
                        truth.edge_rule,
                        truth.edge_threshold,
                        0,
                        stream,
                    )
                )
    return datasets


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Build every synthetic dataset and write the manifest"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--family",
        action="append",
        choices=FAMILY_CHOICES,
        help="build only this family; repeatable, default all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="write here instead of config.output_root",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.output_root is not None:
        config = replace_output_root(config, args.output_root.resolve())
    families = tuple(args.family) if args.family else FAMILY_CHOICES

    entries = build_all(config, families)
    path = write_manifest(config.output_root, entries)
    print(f"{len(entries)} datasets -> {config.output_root}")
    print(f"manifest -> {path}")


def replace_output_root(config: Config, output_root: Path) -> Config:
    """Return the configuration with a different output root.

    Used by ``--output-root`` and by the reproducibility test, which builds twice into
    temporary directories rather than over the working data.
    """
    import dataclasses

    return dataclasses.replace(config, output_root=output_root)


if __name__ == "__main__":
    main()
