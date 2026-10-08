import argparse
import pickle
import onnxruntime.quantization as quant
import numpy as np
from pathlib import Path
from constants import NUM_FEATURES, WINDOW_SIZE

class TelemetryDataReader(quant.CalibrationDataReader):
    def __init__(self, X_calib_t_for_reader:np.ndarray):
        """
        - X_calib_t_for_reader is (N, 1, 5, 20) because the quantize needs an extra array wrapped around each window 
        - This calibration step is used to calculate the quantisation parameters 
        """
        if len(X_calib_t_for_reader) == 0 or X_calib_t_for_reader.ndim != 4 or X_calib_t_for_reader.shape[1:] != (1, NUM_FEATURES, WINDOW_SIZE):
            raise ValueError("calibration data must have shape [N, 1, num_features, window_size]")
        self.data_iter = iter(X_calib_t_for_reader)
    def get_next(self):
        sample = next(self.data_iter, None)
        return None if sample is None else {"input_telemetry": sample}

def quantize_onnx_model(
    calib_data: np.ndarray,
    input_onnx_path: Path = Path("models/conv_autoencoder_ids.onnx"),
    output_onnx_path: Path = Path("models/conv_autoencoder_ids_quantized.onnx"),
    
):
    """Applies static INT8 quantization using training-only calibration samples.

    - calib_data is (N, 1, 5, 20) because the quantize needs an extra array wrapped around each window 
    - This calibration step is used to calculate the quantisation parameters 
    
    """
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    data_reader = TelemetryDataReader(calib_data)

    quant.quantize_static(
        model_input=str(input_onnx_path),
        model_output=str(output_onnx_path),
        calibration_data_reader=data_reader,
        quant_format=quant.QuantFormat.QDQ,
        per_channel=False,
        #when per_channel = True
        #this gives quantisation parameters for each channel rather than whole layer, 
        # It is more useful when you have different distributions for data per channel.
        # Even with similar distribution, this reduces impact of outlier in one channel killing the precision for whole layer 
        # but takes more memory on the Pi to calculate. 
        # Its inefficient on A53 processors on some processors it can be sped up and on GPUs this is marginal
        # This conversation gives the answer to why https://gemini.google.com/u/1/app/8821eed9766b1b43?pageId=none
        weight_type=quant.QuantType.QInt8,#eventual types the weights and activations get quantised to
        activation_type=quant.QuantType.QUInt8
    )
    print(f"Successfully quantized model -> '{output_onnx_path}'")

def main() -> None:
    parser = argparse.ArgumentParser(description="Quantised model is generated in output-onnx path from an existing onnx model.")
    parser.add_argument("--input-onnx", type=Path, default=Path("models/conv_autoencoder_ids.onnx"))
    parser.add_argument("--calibration", type=Path, default=Path("data/processed/X_calib_t_for_reader.pkl"))
    parser.add_argument("--output-onnx", type=Path, default=Path("models/conv_autoencoder_ids_quantized.onnx"))
    args = parser.parse_args()
    with args.calibration.open("rb") as source:
        calibration_data = pickle.load(source)
    quantize_onnx_model(calibration_data, args.input_onnx, args.output_onnx)
    print(f"Quantized and checked ONNX model: '{args.output_onnx}'.")


if __name__ == "__main__":
    main()
