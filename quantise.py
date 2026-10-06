"""Statically quantize a checked ONNX autoencoder using training-only inputs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from f1_can.prepareData import file_sha256


class TelemetryDataReader(CalibrationDataReader):
    """Yield ONNX Runtime calibration feeds from ``[N, 1, 5, 20]`` data."""

    def __init__(self, calibration_data: np.ndarray, input_name: str) -> None:
        if calibration_data.ndim != 4 or calibration_data.shape[1:] != (1, 5, 20):
            raise ValueError("calibration data must have shape [N, 1, 5, 20]")
        self._input_name = input_name
        self._iterator = iter(calibration_data.astype(np.float32, copy=False))

    def get_next(self) -> dict[str, np.ndarray] | None:
        sample = next(self._iterator, None)
        return None if sample is None else {self._input_name: sample}


def quantize_onnx_model(calibration_data: np.ndarray, input_onnx_path: Path, output_onnx_path: Path) -> None:
    session = ort.InferenceSession(str(input_onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "input_telemetry" or inputs[0].shape != [1, 5, 20]:
        raise RuntimeError("the FP32 ONNX model must expose input_telemetry with shape [1, 5, 20]")
    quantize_static(model_input=str(input_onnx_path), model_output=str(output_onnx_path),
                    calibration_data_reader=TelemetryDataReader(calibration_data, inputs[0].name),
                    quant_format=QuantFormat.QDQ, per_channel=False,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    onnx.checker.check_model(str(output_onnx_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", type=Path, default=Path("models/conv_autoencoder_ids.onnx"))
    parser.add_argument("--calibration", type=Path, default=Path("data/processed/X_calib_t_for_reader.pkl"))
    parser.add_argument("--output-onnx", type=Path, default=Path("models/conv_autoencoder_ids_quantized.onnx"))
    args = parser.parse_args()
    with args.calibration.open("rb") as source:
        calibration_data = pickle.load(source)
    quantize_onnx_model(calibration_data, args.input_onnx, args.output_onnx)
    metadata = {"source_fp32_onnx_sha256": file_sha256(args.input_onnx), "quantized_onnx_sha256": file_sha256(args.output_onnx),
                "calibration_sha256": file_sha256(args.calibration), "calibration_windows": int(len(calibration_data)),
                "format": "QDQ", "per_channel": False, "weight_type": "QInt8", "activation_type": "QUInt8"}
    args.output_onnx.with_suffix(args.output_onnx.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Quantized and checked ONNX model: '{args.output_onnx}'.")


if __name__ == "__main__":
    main()
