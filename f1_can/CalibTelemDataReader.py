import onnxruntime.quantization as quant
import numpy as np
from . import WINDOW_SIZE, BATCH_SIZE, STRIDE
class TelemetryDataReader(quant.CalibrationDataReader):
    def __init__(self, X_calib_t_for_reader:np.array):
        """
        - X_calib_t_for_reader is (N, 1, 5, 20) because the quantize needs an extra array wrapped around each window 
        - This calibration step is used to calculate the quantisation parameters 
        """
        self.data_iter = iter(X_calib_t_for_reader)
    def get_next(self):
        return next(self.data_iter, None)
