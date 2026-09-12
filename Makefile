# Define paths (modify these if your structure differs)
RAW_DATA = data/fastf1_2024/fastf1_telemetry.csv #actual place of data
PROCESSED_DATA = pickles of transposed tensor, loader, normal windows for test, train, calib and validation data

SRC_EXTRACT = src/extract.py
SRC_TRANSFORM = src/transform.py
SRC_LOAD = src/load.py
 
# Phony targets: These are not files (avoids conflicts with real files named "run")
.PHONY: run clean
 
# Default target: Run when you type "make" without arguments
train: $(PROCESSED_DATA) (SRC_TRAIN)
	python SRC_TRAIN
 
export: calib_data_path $(export_code) 
	python export_code
# Transform depends on raw data and the transform script
$(PROCESSED_DATA): $(RAW_DATA) $(code_in_prepare_datasets)
	
	python $(code_in_prepare_datasets)
 
# Extract depends on the extract script (no raw data yet)
$(RAW_DATA): $(SRC_EXTRACT)
	
	python $(SRC_EXTRACT)
 
# Clean up generated data (for fresh runs)
clean:
	rm -f $(RAW_DATA) $(PROCESSED_DATA) $(FINAL_DATA)
	