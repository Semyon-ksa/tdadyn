"""The build command, the manifest, and what "a rebuild reproduces the hashes" means.

The datasets are a deterministic function of the configuration and the seed, so a rebuild
must land byte for byte where the previous one did — **within one environment**.  Across
environments it need not: the Rössler system is chaotic at :math:`c = 5.7`, and a different
scipy version, method or tolerance moves the whole series (``docs/method.md`` §3.3).  The
check here is therefore stated as "two builds in one environment agree", which is what can
be verified; promising more would be promising something the data cannot deliver.
"""

from __future__ import annotations

import json

import pytest

from tdadyn.core import entry_payload, read_manifest, truth_digest
from tdadyn.core.config import replace_output_root
from tdadyn.synthetic.build import FAMILY_CHOICES, build_all
from tdadyn.core.io import GROUND_TRUTH_FILE, read_json
from tdadyn.core.manifest import write_manifest


@pytest.fixture(scope="module")
def built(config, tmp_path_factory):
    """One full build, shared by the checks that only read it."""
    root = tmp_path_factory.mktemp("build")
    local = replace_output_root(config, root)
    entries = build_all(local, FAMILY_CHOICES)
    write_manifest(root, entries)
    return local, entries


def test_every_family_is_represented(built):
    _, entries = built
    families = {entry.family for entry in entries}
    assert families == set(FAMILY_CHOICES)


def test_the_manifest_registers_every_dataset(built):
    config, entries = built
    manifest = read_manifest(config.output_root)
    assert len(manifest["datasets"]) == len(entries)
    assert [row["dataset_id"] for row in manifest["datasets"]] == sorted(
        row["dataset_id"] for row in manifest["datasets"]
    )


def test_every_registered_directory_exists(built):
    config, entries = built
    for entry in entries:
        directory = config.output_root / entry.path
        assert directory.is_dir()
        assert (directory / GROUND_TRUTH_FILE).exists()
        assert {file.name for file in directory.glob("*.npy")} == set(entry.arrays)


# --- the layout rule of family 4 (docs/data.md 1) -----------------------------------


def test_a_noisy_dataset_stores_only_the_perturbed_array(built):
    config, entries = built
    expected = {
        "point_jitter": {"points.npy"},
        "weight_jitter": {"weights.npy"},
        "spurious_edges": {"snapshots.npy"},
        "missing_edges": {"snapshots.npy"},
    }
    seen = set()
    for entry in entries:
        if entry.family != "noisy":
            continue
        seen.add(entry.generator)
        assert set(entry.arrays) == expected[entry.generator], entry.dataset_id
    assert seen == set(expected), "not every mechanism was built"


def test_a_clean_cloud_carries_no_network(built):
    _, entries = built
    for entry in entries:
        if entry.family == "static_cloud":
            assert set(entry.arrays) == {"points.npy"}, entry.dataset_id


# --- the free check the manifest makes possible -------------------------------------


def test_zero_noise_reproduces_the_base_hash(built):
    """At level 0 the perturbed array must hash to the base's array, byte for byte.

    Both digests are already in the manifest, so this costs nothing — and it catches, in
    one assertion, a base wired to the wrong dataset, a random stream that shifts between
    levels, and a ``clip`` applied unconditionally.  Any of those would otherwise surface
    much later as an unexplained step at the start of the stability curve.
    """
    config, entries = built
    by_id = {entry.dataset_id: entry for entry in entries}
    checked = 0
    for entry in entries:
        if entry.family != "noisy" or entry.params["level"] != 0.0:
            continue
        base = by_id[entry.params["base"]]
        array = next(iter(entry.arrays))
        if array not in base.arrays:
            # weight_jitter's base is the cloud; its network is derived, not stored, so
            # there is no base digest to compare against.
            continue
        assert entry.arrays[array] == base.arrays[array], (
            f"{entry.dataset_id} at level 0 does not reproduce {base.dataset_id}"
        )
        checked += 1
    assert checked > 0, "no zero-level dataset was compared"


# --- the truth digest ----------------------------------------------------------------


def test_the_truth_digest_tracks_the_record_not_its_formatting(built):
    config, entries = built
    entry = entries[0]
    payload = read_json(config.output_root / entry.path / GROUND_TRUTH_FILE)
    assert truth_digest(payload) == entry.truth_sha256
    # Reformatting must not move the digest: it is taken over the parsed record.
    reformatted = json.loads(json.dumps(payload, indent=8, sort_keys=False))
    assert truth_digest(reformatted) == entry.truth_sha256


def test_the_truth_digest_notices_a_changed_expectation(built):
    config, entries = built
    entry = next(e for e in entries if e.family == "static_cloud")
    payload = read_json(config.output_root / entry.path / GROUND_TRUTH_FILE)
    payload["expected_betti"]["1"] += 1
    # The array digests would not move — the arrays did not change. Only this catches it,
    # and the expected Betti numbers are the whole value of the dataset.
    assert truth_digest(payload) != entry.truth_sha256


# --- reproducibility -----------------------------------------------------------------


# --- the stability artefact (result 6) -----------------------------------------------


def reduced(config, n_points, levels, repeats):
    """The configuration shrunk so the curve is testable in seconds."""
    import dataclasses

    return dataclasses.replace(
        config,
        static_clouds=dataclasses.replace(config.static_clouds, n_points=n_points),
        noise=dataclasses.replace(
            config.noise, point_jitter_sigma=levels, n_repeats=repeats
        ),
    )


def test_the_stability_curve_covers_more_than_one_object(config, tmp_path):
    """One circle is not enough, and the shape of the artefact has to show it.

    ``docs/method.md`` §4.2a: anisotropic and compact objects fail in opposite directions,
    so a stability claim from a circle alone says nothing about :math:`H_2`.  Measured at
    the configured size, the circle's :math:`H_2` curve is flat near 0.02 while the
    sphere's runs to 0.30 — the dimension that matters is invisible on the circle.
    """
    from tdadyn.synthetic.build import STABILITY_OBJECTS, stability_curve

    curve = stability_curve(reduced(config, 60, (0.0, 0.05), 2))
    assert set(curve["objects"]) == set(STABILITY_OBJECTS)
    assert len(STABILITY_OBJECTS) > 1
    for name, entry in curve["objects"].items():
        assert set(entry) >= {"levels", "n_repeats", "H1", "H2"}, name
        for dim in ("H1", "H2"):
            assert len(entry[dim]["mean"]) == len(entry["levels"])
            assert entry[dim]["mean"][0] == 0.0, f"{name} {dim} at level 0"


def test_the_artefact_is_written_by_the_build_not_by_the_tests(config, tmp_path):
    # The tests verify the curve; the CLI produces it. Otherwise regenerating the report
    # would require running the slow suite, and an unrelated failure would leave it
    # unproduced.
    from tdadyn.synthetic.build import RESULTS_DIR, STABILITY_FILE, write_stability

    local = reduced(replace_output_root(config, tmp_path), 60, (0.0, 0.05), 2)
    path = write_stability(local)
    assert path == tmp_path / RESULTS_DIR / STABILITY_FILE
    assert json.loads(path.read_text())["n_points"] == 60


@pytest.mark.slow
def test_a_rebuild_reproduces_the_same_hashes(config, tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first = build_all(replace_output_root(config, first_root), FAMILY_CHOICES)
    second = build_all(replace_output_root(config, second_root), FAMILY_CHOICES)

    # Sets in both directions: looking each new entry up in the old registry would miss a
    # dataset that disappeared.
    left = {json.dumps(entry_payload(entry), sort_keys=True) for entry in first}
    right = {json.dumps(entry_payload(entry), sort_keys=True) for entry in second}
    assert left == right


@pytest.mark.slow
def test_the_manifest_is_replaced_not_accumulated(config, tmp_path):
    # update_manifest never prunes, so a full rebuild must overwrite: otherwise rows for
    # datasets that no longer exist would survive a configuration change and the
    # comparison above would run against a tree nobody has.
    root = tmp_path / "root"
    entries = build_all(replace_output_root(config, root), FAMILY_CHOICES)
    write_manifest(root, entries)
    write_manifest(root, entries[:3])
    assert len(read_manifest(root)["datasets"]) == 3
