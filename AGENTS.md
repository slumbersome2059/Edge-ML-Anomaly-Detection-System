# Repository Guidelines

## Project structure

- `extract.py` and `f1_can/telemetry.py` download and clean FastF1 race-driver telemetry.
- `prepare.py` and `f1_can/prepareData.py` create deterministic segment splits, PyTorch loaders, raw Pi windows, scaler metadata, and the calibration artifact.
- `train.py`, `model_integration/Autoencoder.py`, and `export.py` train in Google Colab and export checked FP32 ONNX.
- `quantise.py` produces static QDQ INT8 ONNX from training-only calibration windows.
- `edge_ids_runner.py` is the Pi runner: it scales raw windows, derives the deployment threshold from validation data, then reports test detection count and latency.

Generated telemetry, pickles, models, thresholds, Colab bundles, and caches belong in `data/`, `models/`, `output/`, or `.cache/`; do not commit them.

## Commands

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
make extract
make prepare
make colab-bundle
make export
```

Never run `make train` on this laptop. Upload the Colab bundle to Google Drive and follow `COLAB.md` for GPU training, ONNX export, and static quantization. Use `requirements-pi.txt` only on the 64-bit Raspberry Pi.

## Data and model invariants

- Feature order is always `RPM`, `Speed`, `Throttle`, `nGear`, `DeltaTime`; model input is `[batch, 5, 20]`.
- Fit `StandardScaler` only on training segments. Colab loaders are scaled; Pi validation and test artifacts are raw and must be scaled in `edge_ids_runner.py`.
- Preserve `preparation_manifest.json`; it fixes split IDs, source hash, seed, and calibration segment. Validation and test segments must never be used for training or calibration.
- The deployment threshold is the 99th percentile of clean validation MSE from the quantized model on the Pi. Bind threshold records to model and preprocessing hashes.
- Synthetic labels and fault tags remain outside model inputs and are not loaded by the Pi test command.

## Tests and style

Use Python 3, four-space indentation, type hints for public functions, and pytest files named `tests/test_*.py`. Unit tests must use temporary/synthetic data and require neither FastF1 downloads, Colab, nor Pi hardware. Cover split persistence, scaling ownership, raw artifact shapes, fault determinism, quantizer feeds, and runner threshold compatibility.
