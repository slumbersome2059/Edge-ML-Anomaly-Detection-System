import argparse
from pathlib import Path
import pickle
from f1_can.prepareData import prepare_datasets, save_processed_data


def main():
    parser = argparse.ArgumentParser(description="Prepare and serialize telemetry datasets.")
    parser.add_argument("--raw-csv", type=Path, default=Path("data/fastf1_2024/fastf1_telemetry.csv"))
    parser.add_argument("--sorted_segment_ids", type=Path, default=Path("data/fastf1_2024/sorted_segment_ids.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    
    with open(args.sorted_segment_ids, "rb") as f: sorted_segment_ids = pickle.load(f)
    

    print(f"Processing raw CSV: '{args.raw_csv}'...")
    train_loader, val_loader, scaler, val_scaled_windows, test_scaled_windows, anomalies, anomaly_types, X_calib_t_for_reader = prepare_datasets(str(args.raw_csv), sorted_segment_ids)

    print(f"Saving processed pickle artifacts to '{args.output_dir}'...")
    save_processed_data(args.output_dir, train_loader, val_loader, scaler, val_scaled_windows, test_scaled_windows, anomalies, anomaly_types, X_calib_t_for_reader)
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
    #sdf