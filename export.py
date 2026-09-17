import argparse
from pathlib import Path
import torch
from model_integration.Autoencoder import ConvAutoencoder1D
from f1_can.prepareData import WINDOW_SIZE, NUM_FEATURES


def export_torch_to_onnx(model: torch.nn.Module, output_path: str = "models/conv_autoencoder_ids.onnx", in_channels: int = 5):
    """Exports trained PyTorch model to ONNX format."""
    model.eval()
    # Dummy input matching shape: (Batch=1, Features=5, Window_Size=20)
    dummy_input = torch.randn(1, NUM_FEATURES, WINDOW_SIZE, dtype=torch.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    #what this does is produce a graph of all the computation that needs to happen to get an output
    torch.onnx.export(
        model,
        dummy_input,#executes graph given by pytorch for which it needs input, the pytorch operations that happen are tracked and converted to ONNX format 
        str(output_path),
        export_params=True,#exports parameters(like weights) as well
        opset_version=14,#opset version describes the operations that can be performed on the runtime
        input_names=["input_telemetry"],#these are names of the inputs and outputs on the graph
        output_names=["reconstructed_telemetry"],
    )
    print(f"Successfully exported ONNX model to: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Export trained model weights to ONNX format.")
    parser.add_argument("--model-path", type=Path, default=Path("models/autoencoder_ids.pth"))
    parser.add_argument("--output-onnx", type=Path, default=Path("models/conv_autoencoder_ids.onnx"))
    args = parser.parse_args()

    model = ConvAutoencoder1D(in_channels=NUM_FEATURES)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))

    export_torch_to_onnx(model, str(args.output_onnx), in_channels=args.in_channels)


if __name__ == "__main__":
    main()