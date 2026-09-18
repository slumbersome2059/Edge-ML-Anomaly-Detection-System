# Google Colab training and quantization

Locally, run only:

```bash
make prepare
make colab-bundle
```

Upload `output/colab_training_bundle.zip` to Google Drive. In a GPU-enabled Colab runtime, mount Drive, copy and unpack the archive, then run:

```bash
pip install -r requirements-colab.txt
python train.py --processed-dir data/processed --model-output models/autoencoder_ids.pth
python export.py --model-path models/autoencoder_ids.pth --output-onnx models/conv_autoencoder_ids.onnx
python quantise.py --input-onnx models/conv_autoencoder_ids.onnx --calibration data/processed/X_calib_t_for_reader.pkl --output-onnx models/conv_autoencoder_ids_quantized.onnx
```

Copy the quantized model, its metadata file, `edge_ids_runner.py`, `requirements-pi.txt`, `data/processed/preprocessing.json`, `data/processed/validation_raw_windows.npy`, and `data/processed/test_raw_windows.npy` to the 64-bit Raspberry Pi. On the Pi:

```bash
python3 -m pip install -r requirements-pi.txt
python3 edge_ids_runner.py validate --model models/conv_autoencoder_ids_quantized.onnx --preprocessing data/processed/preprocessing.json --windows data/processed/validation_raw_windows.npy --threshold-output output/pi_threshold.json
python3 edge_ids_runner.py test --model models/conv_autoencoder_ids_quantized.onnx --preprocessing data/processed/preprocessing.json --windows data/processed/test_raw_windows.npy --threshold output/pi_threshold.json
```

The Pi test command intentionally does not load labels or calculate accuracy metrics.
