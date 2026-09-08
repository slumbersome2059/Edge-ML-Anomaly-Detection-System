"""Extract 2024 FastF1 race telemetry and build model-ready raw archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from f1_can.telemetry import extract_2024_races
from f1_can.prepareData import *
from f1_can.sensors import *

from model_integration.Autoencoder import *

from export import *

import pandas as pd



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/fastf1_2024"))
    parser.add_argument("--raw-csv", type=Path, help="reuse a previously extracted telemetry CSV")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/fastf1"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-sessions", type=int, help="limit downloads for a smoke run", default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trained-model", type=Path, help="reuse previously made model")
    parser.add_argument("--export-tofp32", type=Path, help="reuse previously made model")
    args = parser.parse_args()
    raw_csv = args.raw_csv or args.output_dir / "fastf1_telemetry.csv" 
    #if there is no raw_csv created before we create a new one at path on right(adding the data to csv is done below)
    if not args.raw_csv:
        count = extract_2024_races(raw_csv, args.cache_dir, year=args.year,
                                   max_sessions=args.max_sessions)
        print(f"Extracted {count} raw telemetry rows to {raw_csv}")
    
    train_loader, val_loader, scaler, X_val_t, X_test, X_train_t = (
            prepare_datasets(raw_csv)
        )
    model = ConvAutoencoder1D(in_channels=(list(X_train_t.size())[2]))
    #Data will be form of (Batch, Channels/Features, Window_Size) because this data sent to Conv1D which requires it like this
    if(not args.trained_model):
        
        # 2. Instantiate Model
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Autoencoder Model Initialized (Total Parameters: {total_params})")

        # 3. Train Model
        print("\nStarting Autoencoder Training...")
        train_autoencoder(
            model, train_loader, val_loader, epochs=20
        )

        # 4. Compute Anomaly Threshold
        calculate_anomaly_threshold(
            model, X_val_t
        )
        # Save trained weights for Phase 4 & Phase 5
        torch.save(model.state_dict(), "autoencoder_ids.pth")
    else:
        model.load_state_dict(torch.load(args.trained_model, weights_only=True))#weights-only means tensors and primitive types not normal weights
    if(not args.export_tofp32):
        export_torch_to_onnx(model)
    
    #result = build_archives(raw_csv, args.output_dir, seed=args.seed)
    #print(f"Wrote {result['train']} normal, {result['test']} test, and {result['faulted']} faulted examples to {args.output_dir}")
    
if __name__ == "__main__":
    main()