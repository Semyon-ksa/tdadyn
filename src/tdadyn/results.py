"""Build the artefact of Result 2.

    python -m tdadyn.results --config config.yaml --results

**The artefact is produced by the build, never by pytest.**  A reported number that appears
as a side effect of the slow test run is a number nobody can reproduce on demand, which is
the rule S9 arrived at and the reason this module exists rather than a fixture.

**It sits above the layers rather than inside one, and that was not the first plan.**  The
module began as ``tdadyn.zigzag.run`` and the layering guard rejected it: producing this
artefact needs null realisations, and drawing them means calling the generators — the zigzag
making its own data.  The rule it broke is the one added in Z1 precisely so that the zigzag
reads truth through :mod:`tdadyn.core` instead of producing it.

The artefact needs both layers, and a thing that needs both belongs above both.  This is a
composition root: the one module allowed to know the generators and the algorithm at once,
and outside the four layers the guard names for exactly that reason.

Everything here is assembly and serialisation.  The computing was done in Z1–Z5 and is
reached through :func:`tdadyn.zigzag.analysis.analyse`, :func:`degradation_row` and
:func:`measure_null`; the oracle comes from :mod:`tdadyn.reference`.  Nothing is recomputed a
second way here, because a second way would be a second answer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from tdadyn.core import (
    Config,
    load_config,
    load_dataset,
    replace_output_root,
    write_json,
)
from tdadyn.core.manifest import package_version
from tdadyn.core.manifest import truth_digest as _canonical_digest
from tdadyn.core.schema import jsonify
from tdadyn.synthetic import build_dynamic_graph, build_null_graph
from tdadyn.zigzag import UNION_RULE, dionysus_barcode, index_set, zigzag_complexes
from tdadyn.zigzag.analysis import analyse, degradation_row, measure_null

ARTEFACT_FORMAT_VERSION = 1
RESULTS_DIR = "results"
ARTEFACT_FILE = "zigzag_events.json"

INDEX_CONVENTION = (
    "barcodes live on I = {0, 1/2, 1, ..., T-1}: integer positions are snapshots, "
    "half-integer ones the unions between them. An event is placed on the first snapshot "
    "index at or after a birth and strictly after a death (docs/zigzag.md 4.2)."
)
"""Recorded beside every barcode.

``docs/zigzag.md`` §7: barcodes obtained under different conventions are not comparable, so
the convention is published with them.  Leaving it to be inferred from the commit is how a
number outlives the assumption it was computed under.
"""

_UNREPRODUCIBLE = ("generated_at", "package_version", "content_sha256")
"""Fields excluded from the digest.

Two of them differ between runs by construction and the third is the digest itself.  Exactly
the list ``entry_payload`` keeps for the manifest, and for the same reason: the digest must
answer "does this say the same thing", not "was this written twice".
"""


def artefact_digest(payload: dict) -> str:
    """sha256 of the artefact over its canonical serialisation.

    Reuses :func:`tdadyn.core.manifest.truth_digest` — the same canonical dump S9 settled on
    for ground truth — rather than a second hashing routine.  Hashing the file bytes instead
    would fold indentation, key order and float ``repr`` into the digest, and a cosmetic
    reformat would read as changed results.
    """
    return _canonical_digest(
        {key: value for key, value in payload.items() if key not in _UNREPRODUCIBLE}
    )


def build_artefact(config: Config) -> dict:
    """Compute everything Result 2 reports and return it as a JSON-ready record.

    Parameters
    ----------
    config : Config
        Its ``output_root`` says where the datasets are read from.

    Returns
    -------
    dict
        The artefact, with ``content_sha256`` already filled in.
    """
    root = Path(config.output_root)
    clean = _clean_datasets(root)
    bases = {path.name: load_dataset(path) for path in (root / "dynamic_graph").glob("*")}

    datasets = [_clean_entry(name, dataset, config) for name, dataset in clean.items()]
    datasets.extend(_noisy_entries(root, bases, config))

    payload = {
        "format_version": ARTEFACT_FORMAT_VERSION,
        "generated_at": _utc_now(),
        "package_version": package_version(),
        "convention": {
            "union_rule": UNION_RULE,
            "index_convention": INDEX_CONVENTION,
            "coeff_field": config.zigzag.coeff_field,
            "max_homology_dim": config.zigzag.max_homology_dim,
            "backend": "dionysus",
            "event_matching": {
                "tolerance": config.zigzag.event_matching.tolerance,
                "match_dimension": config.zigzag.event_matching.match_dimension,
                "match_type": config.zigzag.event_matching.match_type,
            },
        },
        "datasets": datasets,
        "null_model": _null_entry(config),
        "agreement": {
            "datasets": len(datasets),
            "bars_compared": sum(e["oracle_agreement"]["bars"] for e in datasets),
            "discrepancies": sum(e["oracle_agreement"]["discrepancies"] for e in datasets),
        },
    }
    payload = jsonify("artefact", payload)
    payload["content_sha256"] = artefact_digest(payload)
    return payload


def write_artefact(config: Config) -> Path:
    """Write the artefact under ``output_root/results`` and return its path."""
    path = Path(config.output_root) / RESULTS_DIR / ARTEFACT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, build_artefact(config))
    return path


# --- the pieces ---------------------------------------------------------------------


def _clean_datasets(root: Path) -> dict[str, object]:
    found = {}
    for path in sorted((root / "dynamic_graph").glob("*")):
        dataset = load_dataset(path)
        if dataset.ground_truth.truth_kind == "betti_curve":
            found[path.name] = dataset
    return found


def _clean_entry(dataset_id: str, dataset, config: Config) -> dict:
    report = analyse(
        dataset, config=config, truth=dataset.ground_truth.events, with_fine_path=True
    )
    return {
        "dataset_id": dataset_id,
        "kind": "clean",
        "base": None,
        "noise": None,
        **_common(dataset, report, config, with_events=True),
    }


def _noisy_entries(root: Path, bases: dict, config: Config) -> list[dict]:
    entries = []
    for path in sorted((root / "noisy").glob("*")):
        dataset = load_dataset(path)
        if not hasattr(dataset, "snapshots"):
            # point_jitter and weight_jitter sit on a family-1 cloud and have no snapshots.
            continue
        noise = dataset.ground_truth.noise_level
        base = bases[noise["base"]]
        row = degradation_row(
            dataset, config=config, truth=base.ground_truth.events, with_fine_path=True
        )
        entries.append(
            {
                "dataset_id": path.name,
                "kind": "noisy",
                "base": noise["base"],
                "noise": {"kind": noise["kind"], "level": float(noise["level"])},
                **_common(dataset, row.report, config, with_events=False),
            }
        )
    return entries


def _common(dataset, report, config: Config, *, with_events: bool) -> dict:
    """The barcode, the metrics and the counters of one dataset.

    ``with_events`` is off for the noisy datasets, and the reason is size rather than
    principle: at noise level 0.1 one of them carries 1802 events, and the sixteen together
    came to 450 kB of a 609 kB file — bulk that no one reads and that re-diffs whole on every
    rebuild.  What Result 2 reports about them is recall, precision and the counters of §6,
    and those stay.  The barcode stays too (9 kB apiece), so the events remain derivable;
    ``n_events`` records how many there were, so the omission is visible rather than silent.
    """
    ceiling = config.zigzag.max_homology_dim
    diagram = zigzag_complexes(
        dataset,
        max_homology_dim=ceiling,
        max_nodes=config.topology_check.reference_max_nodes,
    )
    indices = index_set(len(dataset.snapshots))
    complexes = [diagram[index] for index in indices]
    barcodes = dionysus_barcode(complexes, max_homology_dim=ceiling)
    events = [
        {
            "snapshot": event.snapshot,
            "dim": event.dim,
            "type": event.type,
            "censored": event.censored,
            "from_half_integer": event.from_half_integer,
        }
        for event in sorted(
            report.events,
            key=lambda event: (event.snapshot, event.dim, event.type, event.censored),
        )
    ]
    return {
        "barcode": {
            str(dim): [[birth, death] for birth, death in bars]
            for dim, bars in barcodes.items()
        },
        "oracle_agreement": _agreement(complexes, barcodes, ceiling),
        "n_events": len(events),
        "events": events if with_events else None,
        "metrics": _metrics(report.metrics),
        "counters": {
            "invisible_bars": report.invisible_bars,
            "censored_ends": report.censored_ends,
            "events_from_half_integer_ends": report.events_from_half_integer_ends,
            "ephemeral_bars": report.ephemeral_bars,
            "ephemeral_in_prologue": report.ephemeral_in_prologue,
            "ephemeral_outside_prologue": report.ephemeral_outside_prologue(),
        },
    }


def _agreement(complexes, barcodes, ceiling: int) -> dict:
    """Oracle against backend, bar for bar, on **every** dataset — not only the clean ones.

    The claim of Result 2 is that two independent computations agree, so the evidence belongs
    in the artefact rather than in the test suite's word for it.  Checking it only where the
    barcode has four bars would be the weakest available version of that claim: an error in
    the adapter — ordering, presence runs, the half-open translation — is far likelier to
    show where a noisy dataset produces nine hundred.

    Measured: 265 s across the sixteen noisy datasets, 89 s of it on the worst one.  That is
    most of what ``make results`` costs, and it is the part worth paying for.
    """
    from tdadyn.reference import barcode, module_from_complexes

    compared = discrepancies = 0
    for dim in range(ceiling + 1):
        oracle = barcode(module_from_complexes(complexes, dim))
        compared += len(oracle)
        if tuple(barcodes[dim]) != oracle:
            discrepancies += 1
    return {"bars": compared, "discrepancies": discrepancies}


def _metrics(metrics) -> dict:
    def one(value) -> dict:
        return {
            "n_truth": value.n_truth,
            "n_predicted": value.n_predicted,
            "n_matched": value.n_matched,
            "recall": value.recall,
            "precision": value.precision,
            "mean_localisation_error": value.mean_localisation_error,
            "max_localisation_error": value.max_localisation_error,
        }

    return {
        **one(metrics),
        "by_dimension": {str(dim): one(value)
                         for dim, value in metrics.by_dimension.items()},
    }


def _null_entry(config: Config) -> dict:
    datasets = [
        build_null_graph(
            config, build_dynamic_graph("cycle_lifecycle", config, seed=seed), repeat=repeat
        )
        for seed in config.topology_check.calibration_seeds
        for repeat in range(config.null_models.n_realizations)
    ]
    measurement = measure_null(datasets, config=config)

    def rate(value) -> dict:
        lower, upper = value.interval
        return {"hits": value.hits, "trials": value.trials,
                "rate": value.rate, "interval": [lower, upper]}

    return {
        "seeds": list(config.topology_check.calibration_seeds),
        "realizations": config.null_models.n_realizations,
        "per_snapshot": rate(measurement.per_snapshot),
        "per_realisation": rate(measurement.per_realisation),
        "events": measurement.events,
        "events_by_dimension": {str(dim): count
                                for dim, count in measurement.events_by_dimension.items()},
        "events_per_snapshot": measurement.events_per_snapshot,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compute the barcodes, event logs and metrics of result 2"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="read datasets from here and write the artefact here",
    )
    parser.add_argument(
        "--results", action="store_true",
        help=f"also write {RESULTS_DIR}/{ARTEFACT_FILE}",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.output_root is not None:
        config = replace_output_root(config, args.output_root.resolve())

    artefact = build_artefact(config)
    _print_tables(artefact)
    if args.results:
        path = Path(config.output_root) / RESULTS_DIR / ARTEFACT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, artefact)
        print(f"artefact -> {path}")


def _print_tables(artefact: dict) -> None:
    clean = [entry for entry in artefact["datasets"] if entry["kind"] == "clean"]
    noisy = [entry for entry in artefact["datasets"] if entry["kind"] == "noisy"]
    print(f"convention: union_rule={artefact['convention']['union_rule']}, "
          f"coeff_field={artefact['convention']['coeff_field']}, "
          f"backend={artefact['convention']['backend']}")
    print()
    print(f"{'dataset':<20}{'recall':<9}{'precision':<12}{'invisible':<11}{'ephemeral (outside)'}")
    for entry in clean:
        metrics = entry["metrics"]
        print(f"{entry['dataset_id'].split('_N')[0]:<20}{metrics['recall']:<9.3f}"
              f"{metrics['precision']:<12.4f}{entry['counters']['invisible_bars']:<11}"
              f"{entry['counters']['ephemeral_outside_prologue']}")
    print()
    print(f"{'noise':<20}{'level':<8}{'recall':<9}{'precision':<12}{'invisible'}")
    for entry in sorted(noisy, key=lambda item: (item["noise"]["kind"], item["base"],
                                                 item["noise"]["level"])):
        metrics = entry["metrics"]
        label = f"{entry['noise']['kind'][:8]} {entry['base'].split('_N')[0][:10]}"
        print(f"{label:<20}{entry['noise']['level']:<8}{metrics['recall']:<9.3f}"
              f"{metrics['precision']:<12.4f}{entry['counters']['invisible_bars']}")
    print()
    null = artefact["null_model"]
    for name in ("per_snapshot", "per_realisation"):
        rate = null[name]
        low, high = rate["interval"]
        print(f"null, {name:<16}{rate['rate']:.4f} "
              f"({rate['hits']}/{rate['trials']}, 95% [{low:.4f}, {high:.4f}])")
    print(f"null, events per snapshot: {null['events_per_snapshot']:.1f}")
    agreement = artefact["agreement"]
    print(f"oracle vs backend: {agreement['bars_compared']} bars over "
          f"{agreement['datasets']} datasets, "
          f"{agreement['discrepancies']} discrepancies")


if __name__ == "__main__":  # pragma: no cover - exercised through the command line
    main()
