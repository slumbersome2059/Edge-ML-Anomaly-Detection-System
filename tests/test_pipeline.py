import json
import pickle

import numpy as np
import pandas as pd

from f1_can.prepareData import (
    feature_columns,
    generate_evaluation_dataset,
    prepare_datasets,
    save_processed_data,
)
from quantise import TelemetryDataReader
from edge_ids_runner import load_preprocessing, load_raw_windows


def make_telemetry(path):
    rows = []
    for segment in range(10):
        for row in range(25):
            rows.append({"RPM": 8000 + segment * 100 + row, "Speed": 150 + row,
                         "Throttle": float((row * 5) % 100), "nGear": float((row % 8) + 1),
                         "DeltaTime": 0.1, "segment_id": f"segment-{segment}"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_preparation_persists_disjoint_raw_edge_artifacts(tmp_path):
    csv_path, output = tmp_path / "telemetry.csv", tmp_path / "processed"
    make_telemetry(csv_path)
    artifacts = prepare_datasets(csv_path, [f"segment-{i}" for i in range(10)], manifest_path=output / "preparation_manifest.json")
    save_processed_data(output, *artifacts)
    manifest = json.loads((output / "preparation_manifest.json").read_text())
    splits = manifest["split_segment_ids"]
    assert not (set(splits["train"]) & set(splits["validation"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert (output / "validation_raw_windows.npy").is_file()
    assert np.load(output / "validation_raw_windows.npy").shape[1:] == (20, 5)
    with (output / "val_loader.pkl").open("rb") as source:
        assert next(iter(pickle.load(source)))[0].shape[1:] == (5, 20)


def test_manifest_reuse_keeps_seeded_segments(tmp_path):
    csv_path, manifest = tmp_path / "telemetry.csv", tmp_path / "manifest.json"
    make_telemetry(csv_path)
    first = prepare_datasets(csv_path, [f"segment-{i}" for i in range(10)], manifest_path=manifest)
    save_processed_data(tmp_path / "processed", *first)
    second = prepare_datasets(csv_path, [f"segment-{i}" for i in range(10)], manifest_path=manifest)
    assert first[-1]["split_segment_ids"] == second[-1]["split_segment_ids"]


def test_evaluation_windows_are_raw_and_deterministic(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    make_telemetry(csv_path)
    frame = pd.read_csv(csv_path)
    one, labels_one, tags_one = generate_evaluation_dataset(frame, feature_columns(), seed=7)
    two, labels_two, tags_two = generate_evaluation_dataset(frame, feature_columns(), seed=7)
    np.testing.assert_array_equal(one, two)
    np.testing.assert_array_equal(labels_one, labels_two)
    assert tags_one == tags_two
    assert np.max(one[:, :, 0]) > 1000  # not StandardScaler-normalized


def test_pi_artifact_loaders_and_quantizer_feed(tmp_path):
    preprocessing = {"feature_columns": feature_columns(), "window_size": 20,
                     "mean": [1, 2, 3, 4, 5], "scale": [1, 1, 1, 1, 1]}
    preprocessing_path = tmp_path / "preprocessing.json"
    preprocessing_path.write_text(json.dumps(preprocessing))
    assert load_preprocessing(preprocessing_path)["mean"].shape == (5,)
    windows_path = tmp_path / "windows.npy"
    np.save(windows_path, np.ones((2, 20, 5), dtype=np.float32), allow_pickle=False)
    assert load_raw_windows(windows_path).shape == (2, 20, 5)
    reader = TelemetryDataReader(np.ones((2, 1, 5, 20), dtype=np.float32), "input_telemetry")
    assert reader.get_next()["input_telemetry"].shape == (1, 5, 20)
