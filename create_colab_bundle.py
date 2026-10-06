"""Package frozen loaders and matching source files for a Google Colab run."""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

from f1_can.prepareData import file_sha256

DATA_FILES = ("train_loader.pkl", "val_loader.pkl", "X_calib_t_for_reader.pkl", "preparation_manifest.json", "preprocessing.json")
SOURCE_FILES = ("train.py", "export.py", "quantise.py", "requirements-colab.txt", "f1_can/__init__.py", "f1_can/prepareData.py", "f1_can/sensors.py", "model_integration/Autoencoder.py")


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("output/colab_training_bundle.zip"))
    args = parser.parse_args()
    missing = [name for name in DATA_FILES if not (args.processed_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"prepare artifacts first; missing: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "git_revision": git_revision(), "processed_files": {name: file_sha256(args.processed_dir / name) for name in DATA_FILES}}
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in DATA_FILES:
            archive.write(args.processed_dir / name, Path("data/processed") / name)
        for name in SOURCE_FILES:
            source = Path(name)
            if source.is_file():
                archive.write(source, source)
        archive.writestr("colab_bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Created Colab bundle: '{args.output}'.")


if __name__ == "__main__":
    main()
