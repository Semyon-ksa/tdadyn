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

STABILITY_OBJECTS = ("circle", "torus", "sphere")
"""Objects the stability curve is measured on, and one circle is not enough.

The history of the significance criterion in S4 is that anisotropic and compact objects
fail in *opposite* directions: the old rule lost :math:`H_2` on the torus and the Klein
bottle while inventing loops on the sphere.  A stability claim backed by a circle with
:math:`\\beta_1 = 1` says nothing about :math:`H_2`, which is the most fragile part of the
pipeline.  The torus supplies :math:`H_2` and anisotropy, the sphere the compact case."""

RESULTS_DIR = "results"
STABILITY_FILE = "stability.json"


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


def stability_curve(config: Config) -> dict:
    """Bottleneck distance from the clean diagram as a function of the noise level.

    Parameters
    ----------
    config : Config
        Supplies the objects' geometry, the noise levels and the repeat count.

    Returns
    -------
    dict
        ``{object: {"levels": [...], "mean": [...], "spread": [...], "n_repeats": n}}``,
        plus the dimension the distances were taken in.

    Notes
    -----
    This is produced by the build, not by the test suite.  The tests *verify* the curve —
    that it is monotone and that it respects :math:`d_B \\le 2 d_H` — while the reported
    numbers of result 6 are an output of the pipeline.  Deriving them as a side effect of
    ``pytest -m slow`` would make regenerating the report require running the tests, and
    would leave the artefact unproduced whenever an unrelated test failed.

    Reported in both :math:`H_1` and :math:`H_2`.  :math:`H_0` is left out because the
    bound holds there in both its forms and so says nothing (``docs/method.md`` §4.1);
    :math:`H_2` is in because the torus is on the list precisely for it.

    **The point count is the family-1 default, not each object's configured size.**  The
    torus is configured at 900 points so that its :math:`H_2` bar *separates from the
    noise* — that is what Betti recovery needs.  This curve measures something else: how
    far the diagram moves under perturbation, which is defined for every bar whether or not
    it is significant.  At 900 points a single diagram costs 25 seconds and the curve would
    take a quarter of an hour; at the default it is minutes, and nothing about the
    stability statement depends on the difference.  The count used is recorded in the
    artefact so the numbers stay interpretable.
    """
    import numpy as np
    from persim import bottleneck

    from .persistence import pairwise_distances, persistence_diagrams

    n_points = config.static_clouds.n_points
    dims = (1, 2)

    def diagrams(points):
        return persistence_diagrams(
            pairwise_distances(points),
            max_homology_dim=max(dims),
            coeff_field=config.topology_check.coeff_field,
        )

    def finite(diagram):
        return diagram[np.isfinite(diagram[:, 1])]

    curve = {}
    for name in STABILITY_OBJECTS:
        base = build_static_cloud(name, config, noise_sigma=0.0, n_points=n_points)
        clean = diagrams(base.points)
        per_dim = {dim: {"mean": [], "spread": []} for dim in dims}
        for level in config.noise.point_jitter_sigma:
            values = {dim: [] for dim in dims}
            for repeat in range(config.noise.n_repeats):
                stream = rng_for(config.seed, noise_stream("point_jitter", name), repeat)
                noisy = make_jittered_cloud(base.points, "curve", level, repeat, stream)
                noisy_diagrams = diagrams(noisy.points)
                for dim in dims:
                    values[dim].append(
                        float(bottleneck(finite(clean[dim]), finite(noisy_diagrams[dim])))
                    )
            for dim in dims:
                per_dim[dim]["mean"].append(float(np.mean(values[dim])))
                per_dim[dim]["spread"].append(float(np.std(values[dim])))
        curve[name] = {
            "levels": [float(level) for level in config.noise.point_jitter_sigma],
            "n_repeats": int(config.noise.n_repeats),
            "H1": per_dim[1],
            "H2": per_dim[2],
        }
    return {"n_points": int(n_points), "objects": curve}


def write_stability(config: Config) -> Path:
    """Compute the stability curve and write it beside the manifest."""
    from .io import write_json

    path = config.output_root / RESULTS_DIR / STABILITY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, stability_curve(config))
    return path


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
    parser.add_argument(
        "--results",
        action="store_true",
        help="also compute the stability curve of result 6 into results/stability.json",
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
    if args.results:
        print(f"stability curve -> {write_stability(config)}")


def replace_output_root(config: Config, output_root: Path) -> Config:
    """Return the configuration with a different output root.

    Used by ``--output-root`` and by the reproducibility test, which builds twice into
    temporary directories rather than over the working data.
    """
    import dataclasses

    return dataclasses.replace(config, output_root=output_root)


if __name__ == "__main__":
    main()
