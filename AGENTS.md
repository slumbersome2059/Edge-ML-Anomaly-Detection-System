# Repository Guidelines

## Project Structure & Module Organization

This repository prepares Formula 1 telemetry for anomaly testing and replays
it as synthetic CAN traffic. Keep each pipeline role isolated:

- `f1_can/telemetry.py` extracts and resamples FastF1 race-driver segments.
- `f1_can/dataset.py` creates the normal/held-out window archives.
- `f1_can/faults.py` injects deterministic test-only anomalies.
- `f1_can/can_frames.py` and `f1_can/replay.py` implement the fixed CAN layout
  and SocketCAN sender.
- `model_integration/` contains copy-in files for the external autoencoder;
  `tests/` contains offline unit tests.

Generated datasets, model outputs, and FastF1 caches belong under `data/`,
`output/`, or `.cache/` and must not be committed. Keep labels out of CAN
payloads and model feature tensors.

## Build, Test, and Development Commands

Install the pipeline dependencies before extraction or replay.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -v
python3 build_fastf1_dataset.py --output-dir data/fastf1_2024
python3 -m f1_can.replay --archive data/fastf1_2024/test_windows.npz --dry-run
```

The test command is hardware/network independent. Dataset generation downloads
and caches FastF1 data; use a fixed `--seed` when comparing outputs.

## Coding Style & Naming Conventions

Use Python 3 with four-space indentation, type hints for public functions, and
standard-library modules where practical. Follow PEP 8 naming: `snake_case` for
functions, variables, modules, and JSON fields; `UPPER_CASE` for CAN IDs and
other constants. Keep CLI parsing inside `main()` and protect it with
`if __name__ == "__main__":`.

Document signal units, bounds, scaling ownership, and byte order beside an
encoding definition. The `0x100` payload is synthetic unless a DBC explicitly
replaces it; do not present it as a vehicle-specific CAN definition.

## Testing Guidelines

Write `unittest` tests named `test_<behavior>` in `tests/test_*.py`. Cover
deterministic splits/faults, archive shapes, scaler ownership, payload
round-trips, and invalid values. Use temporary data or fixtures; do not require
FastF1 downloads, SocketCAN, or a GPU in the unit suite.

## Commit & Pull Request Guidelines

The existing history uses descriptive, sentence-style commit subjects. Use a
concise imperative subject, for example: `Add gear fault injection`. Keep each
commit focused. Pull requests should explain dataset/schema or CAN-ID changes,
list validation commands, and include sample output when payload encoding
changes.
