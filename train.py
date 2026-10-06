"""Train the autoencoder from a frozen preparation bundle (for Google Colab)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from f1_can.prepareData import NUM_FEATURES, file_sha256
from model_integration.Autoencoder import ConvAutoencoder1D, train_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-output", type=Path, default=Path("models/autoencoder_ids.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    with (args.processed_dir / "train_loader.pkl").open("rb") as source:
        train_loader = pickle.load(source)
    with (args.processed_dir / "val_loader.pkl").open("rb") as source:
        validation_loader = pickle.load(source)
    manifest_path = args.processed_dir / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["feature_columns"] != ["RPM", "Speed", "Throttle", "nGear", "DeltaTime"]:
        raise RuntimeError("training bundle feature schema is not supported by this model")
    model = ConvAutoencoder1D(in_channels=NUM_FEATURES)
    print(f"Autoencoder initialized with {NUM_FEATURES} channels and {sum(p.numel() for p in model.parameters())} parameters.")
    train_autoencoder(model, train_loader, validation_loader, epochs=args.epochs)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    metadata = {"preparation_manifest_sha256": file_sha256(manifest_path), "feature_columns": manifest["feature_columns"],
                "window_size": manifest["window_size"], "epochs_requested": args.epochs,
                "checkpoint_sha256": file_sha256(args.model_output)}
    args.model_output.with_suffix(args.model_output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Model state and metadata saved to '{args.model_output}'.")


if __name__ == "__main__":
    main()
