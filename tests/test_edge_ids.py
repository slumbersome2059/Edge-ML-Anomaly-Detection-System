import pickle

import numpy as np

from edge_ids_run_funcs import latency_summary, load_raw_windows, load_scaler_vals
from quantise import TelemetryDataReader


def test_edge_artifact_loaders_validate_shapes(tmp_path):
    scaler_path = tmp_path / "scaler_vals.pkl"
    windows_path = tmp_path / "windows.pkl"
    with scaler_path.open("wb") as stream:
        pickle.dump({"mean": np.zeros(5), "scale": np.ones(5)}, stream)
    with windows_path.open("wb") as stream:
        pickle.dump(np.ones((2, 20, 5), dtype=np.float32), stream)

    assert load_scaler_vals(scaler_path)["mean"].shape == (5,)
    assert load_raw_windows(windows_path).shape == (2, 20, 5)


def test_latency_summary_handles_only_warmup_windows():
    assert latency_summary(np.array([]))["p99_ms"] == 0.0


def test_calibration_reader_uses_onnx_input_name():
    reader = TelemetryDataReader(np.ones((2, 1, 5, 20), dtype=np.float32))
    assert reader.get_next()["input_telemetry"].shape == (1, 5, 20)
