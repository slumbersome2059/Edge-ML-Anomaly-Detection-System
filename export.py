import torch

def export_torch_to_onnx(model: torch.nn.Module, output_path: str = "conv_autoencoder_ids.onnx"):
    """Exports trained PyTorch model to ONNX format for ONNX Runtime on ARM CPU."""
    model.eval()
    # Dummy input matching shape: (Batch=1, Features=5, Window_Size=20)
    dummy_input = torch.randn(1, 5, 20, dtype=torch.float32)

    #what this does is produce a graph of all the computation that needs to happen to get an output
    torch.onnx.export(
        model,
        dummy_input,#executes graph given by pytorch for which it needs input, the pytorch operations that happen are tracked and converted to ONNX format 
        output_path,
        export_params=True,#exports parameters(like weights) as well
        opset_version=14, #opset version describes the operations that can be performed on the runtime
        input_names=["input_telemetry"],#these are names of the inputs and outputs on the graph
        output_names=["reconstructed_telemetry"]
    )
    print(f"Successfully exported model to ONNX: '{output_path}'")

# Do any necessary optimisation of the model
# Gather the data
# Arrange data as iterator where each elem is (batch_size, channels, window_size)