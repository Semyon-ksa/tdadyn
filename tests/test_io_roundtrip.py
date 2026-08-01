"""Saving, loading and registering datasets."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tdadyn.synthetic import (
    dataset_equal,
    entry_payload,
    load_dataset,
    make_entry,
    read_manifest,
    save_dataset,
    update_manifest,
)
from tdadyn.synthetic.io import GROUND_TRUTH_FILE

FLOAT_PRECISION = 6


@pytest.fixture
def output_root(tmp_path):
    return tmp_path / "synthetic"


def store(dataset, output_root, **kwargs):
    family_dir = output_root / dataset.ground_truth.family
    return save_dataset(dataset, family_dir, float_precision=FLOAT_PRECISION, **kwargs)


def test_round_trip_returns_an_identical_dataset(any_fake_dataset, output_root):
    path = store(any_fake_dataset, output_root)
    restored = load_dataset(path, float_precision=FLOAT_PRECISION)
    assert dataset_equal(restored, any_fake_dataset)


def test_round_trip_is_bitwise_not_approximate(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    restored = load_dataset(path)
    assert restored.points.tobytes() == fake_point_cloud.points.tobytes()
    assert restored.weights.tobytes() == fake_point_cloud.weights.tobytes()


def test_two_saves_of_the_same_dataset_are_byte_identical(any_fake_dataset, tmp_path):
    first = store(any_fake_dataset, tmp_path / "one")
    second = store(any_fake_dataset, tmp_path / "two")
    assert first.name == second.name
    names = sorted(item.name for item in first.iterdir())
    assert names == sorted(item.name for item in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_existing_directory_is_not_overwritten_silently(fake_point_cloud, output_root):
    store(fake_point_cloud, output_root)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        store(fake_point_cloud, output_root)


def test_overwrite_replaces_the_directory_whole(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    (path / "stale.npy").write_bytes(b"leftover")
    store(fake_point_cloud, output_root, overwrite=True)
    assert not (path / "stale.npy").exists()


def test_a_failed_save_leaves_no_staging_directory(fake_point_cloud, output_root, monkeypatch):
    import tdadyn.synthetic.io as io_module

    def explode(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(io_module.np, "save", explode)
    with pytest.raises(RuntimeError):
        store(fake_point_cloud, output_root)
    family_dir = output_root / fake_point_cloud.ground_truth.family
    assert list(family_dir.iterdir()) == []


def test_renamed_directory_is_detected(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    renamed = path.parent / "torus_N400_seed1"
    path.rename(renamed)
    with pytest.raises(ValueError, match="renamed by hand"):
        load_dataset(renamed)


def test_hand_edited_params_are_detected(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    payload = json.loads((path / GROUND_TRUTH_FILE).read_text(encoding="utf-8"))
    payload["params"]["n_points"] = 999
    (path / GROUND_TRUTH_FILE).write_text(json.dumps(payload), encoding="utf-8")

    load_dataset(path)  # without the precision only the directory name is checked
    with pytest.raises(ValueError, match="does not follow from the stored params"):
        load_dataset(path, float_precision=FLOAT_PRECISION)


def test_json_is_written_in_the_canonical_form(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    text = (path / GROUND_TRUTH_FILE).read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "Фиктивный" in text  # ensure_ascii=False
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_arrays_are_stored_without_pickle(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    array = np.load(path / "points.npy", allow_pickle=False)
    assert array.dtype == np.float64


# --- manifest --------------------------------------------------------------------


def test_manifest_records_the_hash_of_every_array(fake_point_cloud, output_root):
    path = store(fake_point_cloud, output_root)
    entry = make_entry(path, output_root, fake_point_cloud.ground_truth)
    assert set(entry.arrays) == {"points.npy", "weights.npy"}
    for file_name, digest in entry.arrays.items():
        assert digest == hashlib.sha256((path / file_name).read_bytes()).hexdigest()
    assert entry.path == f"{fake_point_cloud.ground_truth.family}/{path.name}"


def test_manifest_replaces_an_entry_and_stays_sorted(
    fake_point_cloud, fake_series, output_root
):
    for dataset in (fake_series, fake_point_cloud):
        path = store(dataset, output_root)
        update_manifest(output_root, make_entry(path, output_root, dataset.ground_truth))

    path = store(fake_point_cloud, output_root, overwrite=True)
    update_manifest(output_root, make_entry(path, output_root, fake_point_cloud.ground_truth))

    datasets = read_manifest(output_root)["datasets"]
    ids = [item["dataset_id"] for item in datasets]
    assert len(ids) == len(set(ids)) == 2
    assert ids == sorted(ids)


def test_reproducibility_is_defined_without_the_timestamp(fake_point_cloud, tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first = store(fake_point_cloud, first_root)
    second = store(fake_point_cloud, second_root)

    first_entry = make_entry(first, first_root, fake_point_cloud.ground_truth)
    second_entry = make_entry(second, second_root, fake_point_cloud.ground_truth)

    assert entry_payload(first_entry) == entry_payload(second_entry)
    assert first_entry.arrays == second_entry.arrays


def test_manifest_of_a_directory_without_arrays_is_refused(fake_point_cloud, output_root, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no arrays"):
        make_entry(empty, tmp_path, fake_point_cloud.ground_truth)
