"""Raspberry Pi ONNX Runtime validation and held-out detection runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

EXPECTED_FEATURES = ["RPM", "Speed", "Throttle", "nGear", "DeltaTime"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessing(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("feature_columns") != EXPECTED_FEATURES or config.get("window_size") != 20:
        raise ValueError("preprocessing schema does not match the deployed autoencoder")
    mean, scale = np.asarray(config.get("mean"), dtype=np.float32), np.asarray(config.get("scale"), dtype=np.float32)
    if mean.shape != (5,) or scale.shape != (5,) or not np.all(np.isfinite(mean)) or np.any(scale <= 0):
        raise ValueError("preprocessing mean or scale is invalid")
    config["mean"], config["scale"] = mean, scale
    return config


def load_raw_windows(path: Path) -> np.ndarray:
    windows = np.load(path, allow_pickle=False)
    if windows.ndim != 3 or windows.shape[1:] != (20, 5) or not len(windows) or not np.all(np.isfinite(windows)):
        raise ValueError(f"{path} must contain finite [N, 20, 5] windows")
    return windows.astype(np.float32, copy=False)


class EdgeIDSInferenceEngine:
    """Single-threaded batch-one inference suitable for a Raspberry Pi 3."""

    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input_telemetry" or inputs[0].shape != [1, 5, 20]:
            raise RuntimeError("model input must be input_telemetry with shape [1, 5, 20]")
        if len(outputs) != 1:
            raise RuntimeError("model must have exactly one reconstruction output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict_window(self, scaled_window: np.ndarray) -> tuple[float, float]:
        if scaled_window.shape != (20, 5) or not np.all(np.isfinite(scaled_window)):
            raise ValueError("scaled window must be finite with shape [20, 5]")
        tensor = np.transpose(scaled_window, (1, 0))[np.newaxis, :, :].astype(np.float32, copy=False)
        started = time.perf_counter()
        reconstruction = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return float(np.mean((tensor - reconstruction) ** 2)), (time.perf_counter() - started) * 1000


def evaluate_windows(raw_windows: np.ndarray, engine: EdgeIDSInferenceEngine, preprocessing: dict[str, Any], *, warmup_windows: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Scale each saved raw window immediately before batch-one inference."""
    errors, latencies = np.empty(len(raw_windows), dtype=np.float64), []
    for index, raw_window in enumerate(raw_windows):
        scaled = (raw_window - preprocessing["mean"]) / preprocessing["scale"]
        errors[index], latency = engine.predict_window(scaled)
        if index >= warmup_windows:
            latencies.append(latency)
    return errors, np.asarray(latencies, dtype=np.float64)


def latency_summary(latencies: np.ndarray) -> dict[str, float]:
    if not len(latencies):
        return {key: 0.0 for key in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms")}
    return {"mean_ms": float(np.mean(latencies)), "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)), "p99_ms": float(np.percentile(latencies, 99)), "max_ms": float(np.max(latencies))}


def validate(model: Path, preprocessing_path: Path, windows_path: Path, threshold_output: Path) -> dict[str, Any]:
    errors, latencies = evaluate_windows(load_raw_windows(windows_path), EdgeIDSInferenceEngine(model), load_preprocessing(preprocessing_path))
    record = {"schema_version": 1, "model_sha256": sha256(model), "preprocessing_sha256": sha256(preprocessing_path),
              "validation_windows_sha256": sha256(windows_path), "percentile": 99.0, "threshold": float(np.percentile(errors, 99)),
              "validation_error_mean": float(np.mean(errors)), "validation_error_max": float(np.max(errors)),
              "windows_evaluated": int(len(errors)), "latency": latency_summary(latencies)}
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    threshold_output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def test(model: Path, preprocessing_path: Path, windows_path: Path, threshold_path: Path) -> dict[str, Any]:
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold.get("model_sha256") != sha256(model) or threshold.get("preprocessing_sha256") != sha256(preprocessing_path):
        raise RuntimeError("threshold belongs to a different model or preprocessing artifact")
    errors, latencies = evaluate_windows(load_raw_windows(windows_path), EdgeIDSInferenceEngine(model), load_preprocessing(preprocessing_path))
    return {"windows_evaluated": int(len(errors)), "anomalies_flagged": int(np.count_nonzero(errors > threshold["threshold"])),
            "threshold": float(threshold["threshold"]), "latency": latency_summary(latencies)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validate", help="calculate an INT8 threshold from clean validation windows")
    validation.add_argument("--model", type=Path, required=True)
    validation.add_argument("--preprocessing", type=Path, required=True)
    validation.add_argument("--windows", type=Path, required=True)
    validation.add_argument("--threshold-output", type=Path, required=True)
    testing = commands.add_parser("test", help="report test detection count and inference latency")
    testing.add_argument("--model", type=Path, required=True)
    testing.add_argument("--preprocessing", type=Path, required=True)
    testing.add_argument("--windows", type=Path, required=True)
    testing.add_argument("--threshold", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.model, args.preprocessing, args.windows, args.threshold_output) if args.command == "validate" else test(args.model, args.preprocessing, args.windows, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
