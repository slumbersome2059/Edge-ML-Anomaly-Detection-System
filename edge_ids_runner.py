import time
from collections import deque
import numpy as np
import pandas as pd
import onnxruntime as ort

# ==========================================
# EDGE RUNNER CONFIGURATION
# ==========================================
ONNX_MODEL_PATH = "conv_autoencoder_ids.onnx"
FEATURE_COLUMNS = ["Speed", "RPM", "Throttle", "nGear", "DeltaTime"]# For Codex: try getting this from test data in pickle if possible 
WINDOW_SIZE = 20# For Codex: try getting this from test data in pickle if possible 
DETECTION_THRESHOLD = 0.04500  # Set from Phase 3 calculated 3-Sigma / 99th percentile

# Pre-fitted StandardScaler parameters (Extracted from training Phase 1)
# Replace these values with scaler.mean_ and scaler.scale_ output from your training set
SCALER_MEAN = np.array([180.5, 9200.0, 45.0, 4.0, 0.10], dtype=np.float32)
SCALER_SCALE = np.array([65.2, 3100.0, 38.5, 1.8, 0.01], dtype=np.float32)


class EdgeIDSInferenceEngine:
    def __init__(self, model_path: str, threshold: float):
        self.threshold = threshold
        # Single-threaded execution optimization for ARM Cortex-A53
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_window(self, scaled_window: np.ndarray) -> tuple[float, float, bool]:
        """
        Input shape: (WINDOW_SIZE, 5)
        Returns: (MSE reconstruction loss, inference latency in ms, is_anomaly flag)
        """
        # Reshape to (1, 5, 20) expected by Conv1D ONNX graph
        tensor_input = np.transpose(scaled_window, (1, 0))[np.newaxis, :, :].astype(np.float32)

        start_time = time.perf_counter()
        reconstruction = self.session.run([self.output_name], {self.input_name: tensor_input})[0]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Mean Squared Error
        mse = float(np.mean((tensor_input - reconstruction) ** 2))
        is_anomaly = mse > self.threshold

        return mse, latency_ms, is_anomaly


def simulate_csv_can_stream(csv_path: str, engine: EdgeIDSInferenceEngine):
    """Simulates real-time CAN bus streaming row-by-row on the Raspberry Pi."""
    df = pd.read_csv(csv_path) #For Codex: you should be able to use pickle instead pandas I think
    telemetry_stream = df[FEATURE_COLUMNS].values

    buffer = deque(maxlen=WINDOW_SIZE)
    latencies = []
    anomaly_count = 0

    print("\n--- Starting Edge CAN Bus Streaming Simulation ---")
    for row_idx, raw_sample in enumerate(telemetry_stream):
        # 1. Standardize incoming sample on the fly
        scaled_sample = (raw_sample - SCALER_MEAN) / SCALER_SCALE
        buffer.append(scaled_sample)

        # 2. Wait until window buffer is fully populated (20 samples)
        if len(buffer) < WINDOW_SIZE:
            continue

        # 3. Construct window array (20, 5)
        current_window = np.array(buffer, dtype=np.float32)

        # 4. Perform Inference
        mse, latency_ms, is_anomaly = engine.predict_window(current_window)
        latencies.append(latency_ms)

        if is_anomaly:
            anomaly_count += 1
            print(
                f"[INTRUSION DETECTED] Row: {row_idx:05d} | "
                f"Reconstruction MSE: {mse:.6f} | Latency: {latency_ms:.2f} ms"
            )

    print("\n==================================================")
    print("      RASPBERRY PI BENCHMARKING SUMMARY          ")
    print("==================================================")
    print(f"Total Windows Evaluated: {len(latencies)}")
    print(f"Total Anomalies Flagged: {anomaly_count}")
    print(f"Mean Latency per Window: {np.mean(latencies):.3f} ms")
    print(f"Max Latency:             {np.max(latencies):.3f} ms")
    print(f"99th Percentile Latency: {np.percentile(latencies, 99):.3f} ms")


if __name__ == "__main__":
    # Initialize ONNX Engine
    ids_engine = EdgeIDSInferenceEngine(
        model_path=ONNX_MODEL_PATH, 
        threshold=DETECTION_THRESHOLD
    )

    # Run Streaming Simulation on Test Data
    # simulate_csv_can_stream("test_telemetry.csv", ids_engine)