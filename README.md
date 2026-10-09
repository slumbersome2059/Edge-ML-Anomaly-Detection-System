# Edge ML Anomaly Detection System

This is an edge-oriented anomaly-detection prototype. It downloads 2024 race telemetry with FastF1, prepares race-driver-aware time-series windows, trains a compact PyTorch convolutional autoencoder on normal behaviour, and flags injected sensor anomalies from reconstruction error.

The project uses `RPM`, `Speed`, `Throttle`, `nGear`, and elapsed time between readings as telemetry features. Formula 1 data is used as an accessible public time-series source; this project does **not** claim to implement a vehicle-specific CAN database or production intrusion-detection system.

## Why this project

Connected vehicles need monitoring that can work close to the data source, where compute and latency budgets are constrained. This repository explores that idea with an unsupervised model: learn the shape of normal telemetry, then use unusually high reconstruction error as an anomaly signal.

It is also an example of how I work as an **AI-native intern**. I naturally use an LLM, notebook-style experimentation, and models to move quickly through unfamiliar areas such as PyTorch, telemetry processing, and test design. Codex and Google Gemini helped generate and explain parts of the implementation and tests; I reviewed the code and unit tests, traced the data flow, and made changes where needed(as you can read from my commits). For example, Codex suggested interpolating data to make it look like data was sampled at a frequent rate but instead I created a column with delta time to combat the data being recorded at inconsistent times. 

## Completed Features

- Extracts 2024 Formula 1 race telemetry via FastF1 and caches downloaded sessions locally.
- Retains and validates core signals: RPM, speed, throttle position, and gear.
- Keeps data grouped by race-driver segment to prevent sliding windows crossing unrelated drives.
- Builds overlapping 20-sample windows (two seconds at a 10 Hz source cadence) for 1D convolutional learning.
- Fits `StandardScaler` on training segments only, avoiding validation/test leakage.
- Trains a lightweight 1D convolutional autoencoder to reconstruct normal telemetry.
- Uses validation reconstruction error to calculate both three-sigma and 99th-percentile thresholds.
- Generates repeatable evaluation examples with RPM spikes, speed offsets, stuck throttle, and gear manipulation.
- Includes metric reporting for precision, recall, F1 score, false-positive rate, and the confusion matrix.
- Exports a trained model to ONNX for a lightweight inference deployment path.

## Features that I am Working on
- Quantization of the ONNX model
- Testing on a Raspbery Pi

## Built with

- **Python** — pipeline and CLI orchestration
- **FastF1** and **pandas** — Formula 1 telemetry extraction and preparation
- **NumPy** and **scikit-learn** — numerical processing and train-only normalisation
- **PyTorch** — 1D convolutional autoencoder training and inference
- **pytest** — regression tests using synthetic telemetry fixtures
- **ONNX** — portable model-export target

## Project structure

```text
.
├── Makefile                      # Extract, prepare, export, and quantise workflow
├── f1_can/
│   ├── telemetry.py              # FastF1 extraction and telemetry validation
│   ├── prepareData.py            # Splits, scaling, windows, and fault injection
│   └── sensors.py                # Signal bounds and anomaly types
├── model_integration/
│   ├── Autoencoder.py            # 1D convolutional autoencoder
│   └── evaluation.py             # Reconstruction-error evaluation metrics
├── tests/                        # Synthetic-data regression tests
└── export.py                     # ONNX export helper
```

## Usage

### 1. Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2. Run the pipeline

Run from the repository root. The first run downloads FastF1 sessions and writes the resulting telemetry CSV to `data/fastf1_2024/`; FastF1 downloads are cached in `.cache/fastf1/`.

```bash
make prepare
```

This extracts missing telemetry and creates deterministic training, validation, test, scaler, and calibration artifacts. It does not train a model.

For a short extraction smoke test, limit the number of race sessions:

```bash
python3 extract.py --max-sessions 2
```

To reuse a telemetry CSV instead of downloading data again:

```bash
python3 prepare.py --raw-csv path/to/fastf1_telemetry.csv
```

Train on Colab from the generated `data/processed/train_loader.pkl` and `val_loader.pkl`; do not run the training target locally.

### 3. Export an existing model

Once weights are available, request the export path:

```bash
python3 export.py --model-path models/autoencoder_ids.pth --output-onnx models/conv_autoencoder_ids.onnx
python3 quantise.py --input-onnx models/conv_autoencoder_ids.onnx --calibration data/processed/X_calib_t_for_reader.pkl --output-onnx models/conv_autoencoder_ids_quantized.onnx
```

This produces FP32 and static-QDQ INT8 ONNX models suitable for an ONNX Runtime-based inference experiment on constrained hardware.

### 4. Run model

```bash
python3 edge_ids_validate.py   --model models/conv_autoencoder_ids_quantized.onnx   --scaler-vals data/processed/scaler_vals.pkl   --windows data/processed/val_unscaled_windows.pkl   --threshold-output output/int8_threshold.pkl

python3 edge_ids_test.py   --model models/conv_autoencoder_ids_quantized.onnx   --scaler-vals data/processed/scaler_vals.pkl   --windows data/processed/test_unscaled_windows.pkl   --threshold output/int8_threshold.pkl --test-output output/int8_test_out.pkl

python3 evaluate.py --test-output output/int8_test_out.pkl --y-true data/processed/anomalies.pkl --fault-tags data/processed/anomaly_types.pkl --output-results-path output/int8_results.pkl
```

The test suite uses generated telemetry fixtures, so it does not require a FastF1 download, GPU, or physical CAN hardware.

## Results

When I got to running the model on the Pi I realised that my Pi wasn't working so I had to run it on my laptop.

### Benefits of Quantisation
- Unfortunately, I didn't see many differences in performance with quantisation 
- With size the quantised version had a size of 7kb and the unquantised one had a size of 7.2kb
- This is probably due to the overhead of quantisation making a large difference because the model is small
- With bigger models you could probably make out a large difference in size
- In terms of time the unquantised one ran at 0.014 ms per window prediction while the quantised one was 0.016 ms
- This was me running it on my laptop so we may see different results on the PI
- Again, I think it is the overhead of quantisation making a difference here
### Model Results
- The reason the recall is low is because of the failure to identify throttle_stuck and gear issues
- This is because it is hard to inject actual anomalies for throttle_stuck and gear(it just ends up looking like normal data) and I have got to improve the structure of the autoencoder as well
- Quantised does do worse than unquantised with accuracy as expected but interestingly it is the false negatives where it is much worse 
#### Unquantised
Confusion Matrix:
[[30710   336]
 [15004 15723]]
True Positives (Detected Attacks): 15723 | False Positives: 336
True Negatives (Clean Windows):    30710 | False Negatives: 15004

Precision: 0.9791
Recall:    0.5117
F1-Score:  0.6721
FPR:       1.0823%

--- Recall Breakdown by Injection Type ---
rpm_spike      : 92.26% detected (7016/7605)

speed_offset   : 83.54% detected (6486/7764)

throttle_stuck : 1.33% detected (103/7771)

gear           : 27.92% detected (2118/7587)
#### Quantised
Confusion Matrix:
[[30707   339]
 [17034 13693]]
True Positives (Detected Attacks): 13693 | False Positives: 339
True Negatives (Clean Windows):    30707 | False Negatives: 17034

Precision: 0.9758
Recall:    0.4456
F1-Score:  0.6119
FPR:       1.0919%

--- Recall Breakdown by Injection Type ---
rpm_spike      : 89.72% detected (6823/7605)
speed_offset   : 67.97% detected (5277/7764)
throttle_stuck : 1.35% detected (105/7771)
gear           : 19.61% detected (1488/7587)

## How anomaly detection works

1. Race-driver telemetry is cleaned and partitioned by `segment_id` into train, validation, and test groups.
2. The scaler is fitted only on training telemetry, then applied to all partitions.
3. Each partition becomes overlapping `[features, 20]` windows for the autoencoder.
4. The model learns to reconstruct normal training windows.
5. Mean squared reconstruction error on clean validation data establishes candidate thresholds.
6. A separate evaluation generator injects a seeded mix of faults and compares their errors with the threshold.

## Fault scenarios

| Fault | Effect |
| --- | --- |
| `rpm_spike` | Temporarily amplifies RPM within its configured maximum. |
| `speed_offset` | Adds or subtracts a large speed offset. |
| `throttle_stuck` | Holds throttle at either 0% or 100%. |
| `gear` | Alters the selected gear by multiple positions. |

These are deterministic test-time synthetic scenarios, not claims about a particular vehicle or CAN implementation.

## Notes and next steps

- Need to improve model architecture and anomaly injection(look at first few bps on model results)
- Need to make my Pi work so that I can run this on the Pi

