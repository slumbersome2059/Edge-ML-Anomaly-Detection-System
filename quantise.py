import onnxruntime.quantization as quant

import onnxruntime.quantization as quant
import numpy as np

class TelemetryDataReader(quant.CalibrationDataReader):
    def __init__(self, X_calib_t_for_reader:np.array):
        """
        - X_calib_t_for_reader is (N, 1, 5, 20) because the quantize needs an extra array wrapped around each window 
        - This calibration step is used to calculate the quantisation parameters 
        """
        self.data_iter = iter(X_calib_t_for_reader)
    def get_next(self):
        return next(self.data_iter, None)

def quantize_onnx_model(
    calib_data: np.array,
    input_onnx_path: str = "conv_autoencoder_ids.onnx",
    output_onnx_path: str = "conv_autoencoder_ids_quantized.onnx",
    
):
    """Applies static INT8 quantization using calibration samples from test.csv.

    - calib_data is (N, 1, 5, 20) because the quantize needs an extra array wrapped around each window 
    - This calibration step is used to calculate the quantisation parameters 
    
    """
    data_reader = TelemetryDataReader(calib_data)

    quant.quantize_static(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
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