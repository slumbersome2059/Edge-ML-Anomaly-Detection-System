"""Create deterministic training and Raspberry Pi inference artifacts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from f1_can.prepareData import prepare_datasets, save_processed_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=Path("data/fastf1_2024/fastf1_telemetry.csv"))
    parser.add_argument("--sorted-segment-ids", type=Path, default=Path("data/fastf1_2024/sorted_segment_ids.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regenerate-split", action="store_true")
    args = parser.parse_args()
    with args.sorted_segment_ids.open("rb") as source:
        sorted_segment_ids = pickle.load(source)
    artifacts = prepare_datasets(args.raw_csv, sorted_segment_ids, manifest_path=args.output_dir / "preparation_manifest.json", seed=args.seed, regenerate_split=args.regenerate_split)
    save_processed_data(args.output_dir, *artifacts)
    print(f"Prepared deterministic artifacts in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
