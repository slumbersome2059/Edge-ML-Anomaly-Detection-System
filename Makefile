# Path Definitions
RAW_DATA = data/fastf1_2024/fastf1_telemetry.csv
PROCESSED_DIR = data/processed
STAMP_FILE = $(PROCESSED_DIR)/.stamp
PREPARATION_MANIFEST = $(PROCESSED_DIR)/preparation_manifest.json
MODEL_WEIGHTS = models/autoencoder_ids.pth
ONNX_MODEL = models/conv_autoencoder_ids.onnx
QUANTIZED_ONNX_MODEL = models/conv_autoencoder_ids_quantized.onnx
SORTED_SEGMENT_IDS = data/fastf1_2024/sorted_segment_ids.pkl
COLAB_BUNDLE = output/colab_training_bundle.zip

.PHONY: extract prepare train export quantize colab-bundle clean


# 1. Extraction: Download telemetry and write CSV
$(RAW_DATA): extract.py f1_can/telemetry.py
	python3 extract.py --output-csv $(RAW_DATA)

extract: $(RAW_DATA)

# 2. Preparation: Transform CSV and output pickle files
$(PREPARATION_MANIFEST): $(RAW_DATA) $(SORTED_SEGMENT_IDS) prepare.py f1_can/prepareData.py
	python3 prepare.py --raw-csv $(RAW_DATA) --output-dir $(PROCESSED_DIR)

prepare: $(PREPARATION_MANIFEST)

# 3. Training: Train autoencoder using processed training loader
$(MODEL_WEIGHTS): $(PREPARATION_MANIFEST) train.py model_integration/Autoencoder.py
	python3 train.py --processed-dir $(PROCESSED_DIR) --model-output $(MODEL_WEIGHTS)

train: $(MODEL_WEIGHTS)
	

# 5. Export: Convert PyTorch model weights to ONNX format
$(ONNX_MODEL): $(MODEL_WEIGHTS) export.py
	python3 export.py --model-path $(MODEL_WEIGHTS) --output-onnx $(ONNX_MODEL)

exportCommand: $(ONNX_MODEL)

export: $(ONNX_MODEL)

$(QUANTIZED_ONNX_MODEL): $(ONNX_MODEL) $(PROCESSED_DIR)/X_calib_t_for_reader.pkl quantise.py
	python3 quantise.py --input-onnx $(ONNX_MODEL) --calibration $(PROCESSED_DIR)/X_calib_t_for_reader.pkl --output-onnx $(QUANTIZED_ONNX_MODEL)

quantize: $(QUANTIZED_ONNX_MODEL)

$(COLAB_BUNDLE): $(PREPARATION_MANIFEST) create_colab_bundle.py requirements-colab.txt train.py export.py quantise.py f1_can/prepareData.py f1_can/sensors.py model_integration/Autoencoder.py
	python3 create_colab_bundle.py --processed-dir $(PROCESSED_DIR) --output $(COLAB_BUNDLE)

colab-bundle: $(COLAB_BUNDLE)

# Clean up all generated data, models, and cache files
clean:
	rm -rf $(RAW_DATA) $(PROCESSED_DIR) models/ .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
