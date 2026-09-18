"""Export a frozen autoencoder checkpoint to a checked FP32 ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from f1_can import WINDOW_SIZE
from f1_can.prepareData import NUM_FEATURES, file_sha256
from model_integration.Autoencoder import ConvAutoencoder1D


def export_torch_to_onnx(model: torch.nn.Module, output_path: Path) -> None:
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, NUM_FEATURES, WINDOW_SIZE, dtype=torch.float32), str(output_path),
                      export_params=True, opset_version=14, input_names=["input_telemetry"],
                      output_names=["reconstructed_telemetry"])
    onnx.checker.check_model(str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/autoencoder_ids.pth"))
    parser.add_argument("--output-onnx", type=Path, default=Path("models/conv_autoencoder_ids.onnx"))
    args = parser.parse_args()
    model = ConvAutoencoder1D(in_channels=NUM_FEATURES)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    export_torch_to_onnx(model, args.output_onnx)
    metadata = {"source_checkpoint_sha256": file_sha256(args.model_path), "onnx_sha256": file_sha256(args.output_onnx),
                "input_name": "input_telemetry", "output_name": "reconstructed_telemetry",
                "input_shape": [1, NUM_FEATURES, WINDOW_SIZE], "opset": 14}
    args.output_onnx.with_suffix(args.output_onnx.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported and checked ONNX model: '{args.output_onnx}'.")


if __name__ == "__main__":
    main()
