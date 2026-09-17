# Path Definitions
RAW_DATA = data/fastf1_2024/fastf1_telemetry.csv
PROCESSED_DIR = data/processed
STAMP_FILE = $(PROCESSED_DIR)/.stamp
MODEL_WEIGHTS = models/autoencoder_ids.pth
ONNX_MODEL = models/conv_autoencoder_ids.onnx
SORTED_SEGMENT_IDS = data/fastf1_2024/sorted_segment_ids.pkl

.PHONY: extract prepare train evaluate export clean


# 1. Extraction: Download telemetry and write CSV
$(RAW_DATA): extract.py f1_can/telemetry.py
	python3 extract.py --output-csv $(RAW_DATA)

extract: $(RAW_DATA)

# 2. Preparation: Transform CSV and output pickle files
$(STAMP_FILE): $(RAW_DATA) prepare.py f1_can/prepareData.py
	python3 prepare.py --raw-csv $(RAW_DATA) --output-dir $(PROCESSED_DIR)
	touch $(STAMP_FILE)

prepare: $(STAMP_FILE)

# 3. Training: Train autoencoder using processed training loader
$(MODEL_WEIGHTS): $(STAMP_FILE) train.py model_integration/Autoencoder.py
	python3 train.py --processed-dir $(PROCESSED_DIR) --model-output $(MODEL_WEIGHTS)

train: $(MODEL_WEIGHTS)
	

# 5. Export: Convert PyTorch model weights to ONNX format
$(ONNX_MODEL): $(MODEL_WEIGHTS) export.py
	python3 export.py --model-path $(MODEL_WEIGHTS) --output-onnx $(ONNX_MODEL)

exportCommand: $(ONNX_MODEL)

# Clean up all generated data, models, and cache files
clean:
	rm -rf $(RAW_DATA) $(PROCESSED_DIR) models/ .cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +