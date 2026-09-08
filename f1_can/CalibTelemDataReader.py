import onnxruntime.quantization as quant
from . import WINDOW_SIZE, BATCH_SIZE, STRIDE
class TelemetryDataReader(quant.CalibrationDataReader):
    def __init__(self, training_windows):
        """
        - training_windows is the windows produced in prepare dataset
        - It is of the shape (N, 20, 5)
        - To build a representative dataset for calibration we sample data randomly
        - This calibration step is helpful to determine the 
        """
        len_training_windows = len(training_windows)
        len_sel_windows_indices = 10
        sel_windows_indices = range(0, len_training_windows, len_training_windows//len_sel_windows_indices)
        calib_batches = []
        for i in sel_windows_indices:
            # Adding batches which are of (Batch_size, num_features, window_size)
            calib_batches.append({"input" : training_windows[i : i + BATCH_SIZE].transpose([1,2])}) 
        self.calib_batches = iter(calib_batches)
        
    def get_next(self):
        return next(self.data_iter, None)
