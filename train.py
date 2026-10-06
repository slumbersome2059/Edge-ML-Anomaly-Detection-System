import argparse
import pickle
from pathlib import Path
import torch
from f1_can.prepareData import NUM_FEATURES

from model_integration.Autoencoder import ConvAutoencoder1D, train_autoencoder, calculate_anomaly_threshold


def main():
    parser = argparse.ArgumentParser(description="Train 1D Conv Autoencoder model.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-output", type=Path, default=Path("models/autoencoder_ids.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    print("Loading training and validation pickle artifacts...")
    with open(args.processed_dir / "train_loader.pkl", "rb") as f:
        train_loader = pickle.load(f)
    with open(args.processed_dir / "val_loader.pkl", "rb") as f:
        val_loader = pickle.load(f)

    
    model = ConvAutoencoder1D(in_channels=NUM_FEATURES)
    total_params = sum(p.numel() for p in model.parameters())#numel gives you the total number of elements in the tensor
    print(f"Autoencoder initialized with {NUM_FEATURES} input channels ({total_params} total parameters).")

    print(f"Starting training for {args.epochs} epochs...")
    train_autoencoder(model, train_loader, val_loader, epochs=args.epochs)


    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    print(f"Model state saved to '{args.model_output}'.")


if __name__ == "__main__":
    main()