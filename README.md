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
├── main.py                       # Extract, prepare, train, and export workflow
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
python3 main.py
```

For a short extraction smoke test, limit the number of race sessions:

```bash
python3 main.py --max-sessions 2
```

To reuse a telemetry CSV instead of downloading data again:

```bash
python3 main.py --raw-csv path/to/fastf1_telemetry.csv
```

The workflow trains for up to 20 epochs, applying early stopping when validation loss stops improving, and saves the trained weights as `autoencoder_ids.pth`.

### 3. Export an existing model

Once weights are available, request the export path:

```bash
python3 main.py --raw-csv path/to/fastf1_telemetry.csv --trained-model autoencoder_ids.pth
```

This produces `conv_autoencoder_ids.onnx`, which is suitable for an ONNX Runtime-based inference experiment on constrained hardware.

### 4. Run tests

```bash
python3 -m pytest
```

The test suite uses generated telemetry fixtures, so it does not require a FastF1 download, GPU, or physical CAN hardware.

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

This is a learning and prototyping project. A production deployment would need vehicle-specific signal definitions, a validated threat model, data from the target platform, model calibration under real operating conditions, and hardware-in-the-loop testing.

Generated telemetry, caches, checkpoints, and exported models are intentionally local artefacts and should not be committed.
