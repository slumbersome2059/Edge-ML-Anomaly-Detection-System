import time
import pickle
import argparse
from pathlib import Path
from typing import Any
import numpy as np

import onnxruntime as ort

# ==========================================
# EDGE RUNNER CONFIGURATION
# ==========================================
FEATURE_COLUMNS = ["RPM", "Speed", "Throttle", "nGear", "DeltaTime"]# For Codex: try getting this from test data in pickle if possible 
WINDOW_SIZE = 20# For Codex: try getting this from test data in pickle if possible 


class EdgeIDSInferenceEngine:
    def __init__(self, model_path: Path):
        # Single-threaded execution optimization for ARM Cortex-A53
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1 #see how these thread options go, if it doesn't research more, context switching might be more expensive than advantage from parallelisation so you reduce number of threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL #whether operators in the graph get executed sequentially or in parallel
        #For models where there are many branches you would want to execute in parallel
        
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        #InferenceSession is used to load and run the model, it also does other things like optimise the graph
        #Provider contains code to run operators for specific targets like (CPU, GPU)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input_telemetry" or inputs[0].shape != [1, 5, 20]:
            raise RuntimeError("model input must be input_telemetry with shape [1, 5, 20]")
        if len(outputs) != 1:
            raise RuntimeError("model must have exactly one reconstruction output")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def predict_window(self, scaled_window: np.ndarray) -> tuple[float, float]:
        """
        Input shape: (WINDOW_SIZE, 5)
        Returns: (MSE reconstruction loss, inference latency in ms)
        """
        if scaled_window.shape != (20, 5) or not np.all(np.isfinite(scaled_window)):
            raise ValueError("scaled window must be finite with shape [20, 5]")
        # Reshape to (1, 5, 20) expected by Conv1D ONNX graph, it does a transposition and expansion and casting here
        tensor_input = np.transpose(scaled_window, (1, 0))[np.newaxis, :, :].astype(np.float32)

        start_time = time.perf_counter()
        reconstruction = self.session.run([self.output_name], {self.input_name: tensor_input})[0]#the results of running the model on input are returned by run
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Mean Squared Error
        mse = float(np.mean((tensor_input - reconstruction) ** 2))#mean will be taken over flattened array

        return mse, latency_ms

def evaluate_windows(raw_windows: np.ndarray, engine: EdgeIDSInferenceEngine, scaler_vals: dict[str, np.ndarray], *, warmup_windows: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Scale each saved raw window immediately before batch-one inference."""
    errors, latencies = np.empty(len(raw_windows), dtype=np.float64), []
    for index, raw_window in enumerate(raw_windows):
        scaled = (raw_window - scaler_vals["mean"]) / scaler_vals["scale"] #there is a mean and a scale(this is standard deviation) for each feature so numpy duplicates these 5 element arrays 20 times and does stuff element wise
        errors[index], latency = engine.predict_window(scaled)
        if index >= warmup_windows:
            latencies.append(latency)
    return errors, np.asarray(latencies, dtype=np.float64)

def latency_summary(latencies: np.ndarray) -> dict[str, float]:
    if len(latencies) == 0:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {"mean_ms": float(np.mean(latencies)), "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)), "p99_ms": float(np.percentile(latencies, 99)), "max_ms": float(np.max(latencies))}

def load_scaler_vals(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        config = pickle.load(source)
    if not isinstance(config, dict):
        raise ValueError("scaler values must be stored as a dictionary")
    mean = np.asarray(config.get("mean"), dtype=np.float32)
    scale = np.asarray(config.get("scale"), dtype=np.float32)
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != (len(FEATURE_COLUMNS),):
        raise ValueError("scaler mean and scale must contain one value per feature")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scaler mean and scale must be finite, with positive scales")
    config["mean"] = mean
    config["scale"] = scale
    return config

def load_raw_windows(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        windows = pickle.load(source)
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, len(FEATURE_COLUMNS)):
        raise ValueError("windows must have shape [number_of_windows, window_size, num_features]")
    if len(windows) == 0 or not np.all(np.isfinite(windows)):#isfinite tests elementwise for whether an element is infinity or not
        raise ValueError("windows must be non-empty and finite")
    return windows
