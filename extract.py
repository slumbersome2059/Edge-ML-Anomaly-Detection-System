import argparse
from pathlib import Path
from f1_can.telemetry import extract_2024_races
import pickle


def main():
    parser = argparse.ArgumentParser(description="Extract FastF1 race telemetry data.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/fastf1_2024/fastf1_telemetry.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/fastf1"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-sessions", type=int, default=42)
    args = parser.parse_args()

    print(f"Extracting FastF1 telemetry for year {args.year}...")
    count = extract_2024_races(args.output_csv, args.cache_dir, year=args.year, max_sessions=args.max_sessions)
    print(f"Successfully extracted {count} telemetry rows to '{args.output_csv}'.")


if __name__ == "__main__":
    main()
    