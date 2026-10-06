"""Deterministic preparation of training and edge-inference telemetry artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import BATCH_SIZE, PROCESSED_COLUMNS, STRIDE, WINDOW_SIZE
from .sensors import Sensors

NUM_FEATURES = len(Sensors.SENSOR_NAME_COLUMNS + PROCESSED_COLUMNS)
SPLIT_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15


def file_sha256(path: Path) -> str:
    """Return a stable digest used to bind generated artifacts to their source."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_columns() -> list[str]:
    """Return the model's canonical raw-feature order."""
    return Sensors.SENSOR_NAME_COLUMNS + PROCESSED_COLUMNS


def validate_dataframe_columns(df: pd.DataFrame) -> None:
    expected = set(feature_columns() + ["segment_id"])
    actual = set(df.columns)
    if actual != expected:
        raise ValueError(
            "telemetry columns do not match the model schema: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


def extract_windows(dframe: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Create windows without crossing a race-driver segment boundary."""
    windows: list[np.ndarray] = []
    for _, group in dframe.groupby("segment_id", sort=False):
        values = group[columns].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
            windows.append(values[start : start + WINDOW_SIZE])
    if not windows:
        return np.empty((0, WINDOW_SIZE, len(columns)), dtype=np.float32)
    return np.stack(windows).astype(np.float32, copy=False)


def _legacy_seeded_split(segment_ids: np.ndarray, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Preserve the project's original ``np.random.seed(); shuffle()`` split."""
    shuffled = np.array(segment_ids, dtype=object, copy=True)
    legacy_rng = np.random.RandomState(seed)
    legacy_rng.shuffle(shuffled)
    train_end = int(len(shuffled) * TRAIN_FRACTION)
    validation_end = train_end + int(len(shuffled) * VALIDATION_FRACTION)
    return (
        shuffled[:train_end].tolist(),
        shuffled[train_end:validation_end].tolist(),
        shuffled[validation_end:].tolist(),
    )


def _load_or_create_manifest(
    csv_path: Path,
    sorted_segment_ids: list[str],
    manifest_path: Path,
    *,
    seed: int,
    regenerate_split: bool,
) -> dict[str, Any]:
    """Persist split membership so later preparation cannot silently reshuffle it."""
    raw_hash = file_sha256(csv_path)
    sorted_hash = hashlib.sha256(json.dumps(sorted_segment_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    if manifest_path.exists() and not regenerate_split:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("raw_csv_sha256") != raw_hash:
            raise RuntimeError(
                "raw telemetry differs from the existing preparation manifest; "
                "pass --regenerate-split to intentionally create a new split"
            )
        if manifest.get("feature_columns") != feature_columns() or manifest.get("window_size") != WINDOW_SIZE:
            raise RuntimeError("existing preparation manifest is incompatible with the current feature schema")
        if manifest.get("sorted_segment_ids_sha256") != sorted_hash:
            raise RuntimeError("sorted segment order differs from the existing preparation manifest")
        return manifest

    df = pd.read_csv(csv_path, usecols=["segment_id"])
    all_segments = df["segment_id"].drop_duplicates().to_numpy()
    train_ids, validation_ids, test_ids = _legacy_seeded_split(all_segments, seed)
    train_set = set(train_ids)
    calibration_id = next((segment for segment in sorted_segment_ids if segment in train_set), None)
    if calibration_id is None:
        raise RuntimeError("no sorted calibration segment belongs to the training split")
    return {
        "schema_version": 1,
        "raw_csv_sha256": raw_hash,
        "sorted_segment_ids_sha256": sorted_hash,
        "seed": seed,
        "feature_columns": feature_columns(),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "split_fractions": {"train": TRAIN_FRACTION, "validation": VALIDATION_FRACTION, "test": 1 - TRAIN_FRACTION - VALIDATION_FRACTION},
        "split_segment_ids": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "calibration_segment_id": calibration_id,
    }


def inject_fault(raw_window: np.ndarray, fault_type: str, rng: np.random.Generator) -> np.ndarray:
    """Inject one bounded synthetic fault into an unscaled telemetry window."""
    result = np.array(raw_window, dtype=np.float32, copy=True)
    duration = int(rng.choice((2, 3, 4, 5)))
    start = int(rng.integers(0, WINDOW_SIZE - duration + 1))
    affected = slice(start, start + duration)
    if fault_type == Sensors.RPM.fault.value:
        sensor = Sensors.RPM
        result[affected, sensor.index] = np.clip(result[affected, sensor.index] * rng.uniform(1.45, 1.9), 0, sensor.max_val)
    elif fault_type == Sensors.SPEED.fault.value:
        sensor = Sensors.SPEED
        result[affected, sensor.index] = np.clip(result[affected, sensor.index] + rng.choice((-1, 1)) * rng.uniform(55, 90), 0, sensor.max_val)
    elif fault_type == Sensors.THROTTLE.fault.value:
        sensor = Sensors.THROTTLE
        result[affected, sensor.index] = rng.choice((0, sensor.max_val))
    elif fault_type == Sensors.GEAR.fault.value:
        sensor = Sensors.GEAR
        result[affected, sensor.index] = np.clip(result[affected, sensor.index] + rng.choice((-3, -2, 2, 3)), 0, sensor.max_val)
    else:
        raise ValueError(f"unknown fault type: {fault_type}")
    return result


def generate_evaluation_dataset(
    test_df: pd.DataFrame,
    columns: list[str],
    *,
    anomaly_ratio: float = 0.5,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return deterministic raw held-out windows and labels; do not scale them here."""
    if not 0 <= anomaly_ratio <= 1:
        raise ValueError("anomaly_ratio must be between zero and one")
    rng = np.random.default_rng(seed)
    windows = extract_windows(test_df, columns)
    result = np.empty_like(windows)
    labels = np.zeros(len(windows), dtype=np.int8)
    fault_tags: list[str] = []
    for index, window in enumerate(windows):
        if rng.random() < anomaly_ratio:
            fault = rng.choice(Sensors.FAULT_TYPES)
            result[index] = inject_fault(window, fault.value, rng)
            labels[index] = 1
            fault_tags.append(fault.value)
        else:
            result[index] = window
            fault_tags.append("clean")
    return result, labels, fault_tags


def prepare_datasets(
    csv_path: str | Path,
    sorted_segment_ids: list[str],
    *,
    manifest_path: str | Path | None = None,
    seed: int = SPLIT_SEED,
    regenerate_split: bool = False,
) -> tuple[DataLoader, DataLoader, StandardScaler, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    """Prepare training loaders and unscaled edge validation/test artifacts."""
    source_path = Path(csv_path)
    target_manifest = Path(manifest_path) if manifest_path else source_path.with_suffix(".manifest.json")
    manifest = _load_or_create_manifest(source_path, sorted_segment_ids, target_manifest, seed=seed, regenerate_split=regenerate_split)
    df = pd.read_csv(source_path)
    validate_dataframe_columns(df)
    columns = feature_columns()
    split_ids = manifest["split_segment_ids"]
    train_df = df[df["segment_id"].isin(split_ids["train"])]
    validation_df = df[df["segment_id"].isin(split_ids["validation"])]
    test_df = df[df["segment_id"].isin(split_ids["test"])]
    if train_df.empty or validation_df.empty or test_df.empty:
        raise RuntimeError("each persisted split must contain telemetry rows")

    scaler = StandardScaler().fit(train_df[columns].to_numpy())
    float_columns = {column: np.float64 for column in columns}
    scaled_train = train_df.copy().astype(float_columns)
    scaled_validation = validation_df.copy().astype(float_columns)
    scaled_train.loc[:, columns] = scaler.transform(train_df[columns].to_numpy())
    scaled_validation.loc[:, columns] = scaler.transform(validation_df[columns].to_numpy())
    train_windows = extract_windows(scaled_train, columns)
    validation_windows = extract_windows(scaled_validation, columns)
    validation_raw_windows = extract_windows(validation_df, columns)
    calibration_df = df[df["segment_id"] == manifest["calibration_segment_id"]]
    calibration_windows = extract_windows(calibration_df, columns)
    if not len(train_windows) or not len(validation_windows) or not len(calibration_windows):
        raise RuntimeError("a split does not contain enough samples for one telemetry window")
    calibration_scaled = scaler.transform(calibration_windows.reshape(-1, NUM_FEATURES)).reshape(calibration_windows.shape).astype(np.float32)
    calibration_for_reader = np.expand_dims(np.transpose(calibration_scaled, (0, 2, 1)), axis=1)
    test_raw_windows, labels, fault_tags = generate_evaluation_dataset(test_df, columns, seed=seed)
    train_tensor = torch.from_numpy(np.transpose(train_windows, (0, 2, 1)).astype(np.float32))
    validation_tensor = torch.from_numpy(np.transpose(validation_windows, (0, 2, 1)).astype(np.float32))
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_tensor), batch_size=BATCH_SIZE, shuffle=False)
    manifest["artifact_counts"] = {
        "train_windows": int(len(train_windows)), "validation_windows": int(len(validation_windows)),
        "test_windows": int(len(test_raw_windows)), "calibration_windows": int(len(calibration_for_reader)),
    }
    return train_loader, validation_loader, scaler, validation_raw_windows, test_raw_windows, labels, fault_tags, calibration_for_reader, manifest


def save_processed_data(
    output_dir: Path, train_loader: DataLoader, validation_loader: DataLoader, scaler: StandardScaler,
    validation_raw_windows: np.ndarray, test_raw_windows: np.ndarray, labels: np.ndarray,
    fault_tags: list[str], calibration_for_reader: np.ndarray, manifest: dict[str, Any],
) -> None:
    """Write Colab-training and Pi-inference artifacts with explicit scaling ownership."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (("train_loader.pkl", train_loader), ("val_loader.pkl", validation_loader), ("X_calib_t_for_reader.pkl", calibration_for_reader)):
        with (output_dir / name).open("wb") as destination:
            pickle.dump(value, destination)
    np.save(output_dir / "validation_raw_windows.npy", validation_raw_windows, allow_pickle=False)
    np.save(output_dir / "test_raw_windows.npy", test_raw_windows, allow_pickle=False)
    np.save(output_dir / "test_labels.npy", labels, allow_pickle=False)
    (output_dir / "test_fault_tags.json").write_text(json.dumps(fault_tags), encoding="utf-8")
    preprocessing = {"schema_version": 1, "feature_columns": feature_columns(), "window_size": WINDOW_SIZE,
                     "mean": np.asarray(scaler.mean_, dtype=float).tolist(), "scale": np.asarray(scaler.scale_, dtype=float).tolist()}
    (output_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    (output_dir / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
