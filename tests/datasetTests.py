import io
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

# Import your functions and constants from the source modules
import f1_can
from f1_can.prepareData import prepare_datasets, inject_fault, WINDOW_SIZE, generate_evaluation_dataset, split_segments, get_loaders_for_window_lists, windows_df, give_scaler
from f1_can.telemetry import _resample_car_data, keepRequiredColumns
from f1_can.sensors import Sensors


# =====================================================================
# FIXTURES & MOCKS
# =====================================================================


@pytest.fixture
def raw_car_data():
    """Generates synthetic FastF1 car telemetry DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=WINDOW_SIZE, freq="100ms")
    return pd.DataFrame({
        "Date": dates,
        "RPM": [10000 + i * 100 for i in range(WINDOW_SIZE)],
        "Speed": [200 + i for i in range(WINDOW_SIZE)],
        "Throttle": [50.0, 60.0, 105.0, -5.0, 70.0, 80.0, 90.0, 100.0, 0.0, 50.0,
                     50.0, 60.0, 20.0, 5.0, 70.0, 80.0, 90.0, 100.0, 0.0, 50.0],  # Includes invalid values
        "nGear": [1, 2, 3, 1, 4, 5, 6, 7, 8, 8,
                  1, 2, 3, 1, 4, 5, 6, 7, 8, 8],                                 # Includes invalid values
    })

@pytest.fixture
def sample_window():
    """Returns a single clean window of shape (20, 4)."""
    # Columns: RPM (2000), Speed (60), Throttle (20), Gear (3)
    
    #then on each dimension the array is repeated the same number of times
    return pd.DataFrame({
        "RPM":[2000.0 for i in range(WINDOW_SIZE)],
        "Speed":[60.0 for i in range(WINDOW_SIZE)],
        "Throttle":[20.0 for i in range(WINDOW_SIZE)],
        "nGear":[3.0 for i in range(WINDOW_SIZE)]
    })

@pytest.fixture
def sample_test_dataset(sample_window):
    """Returns a batch of raw windows of shape (10, 20, 4)."""
    rng = np.random.default_rng(0) #creates a generator
    return [sample_window for i in range(10)] #an nd array created with the size given, the values generated from low = 10, high = 100


@pytest.fixture
def synthetic_telemetry_csv(tmp_path):
    """Creates a temporary valid CSV file for dataset preparation tests."""
    csv_file = tmp_path / "test_telemetry.csv"
    
    # Build 10 segments to allow split partitioning (70/15/15)
    records = []
    dates = pd.date_range("2024-01-01", periods=30, freq="100ms")
    
    for seg_idx in range(10):
        seg_id = f"2024-R01-driver_{seg_idx}"
        for d in dates:
            records.append({
                "RPM": np.random.randint(8000, 12000),
                "Speed": np.random.uniform(150, 300),
                "Throttle": np.random.uniform(0, 100),
                "nGear": np.random.randint(1, 8),
                "DeltaTime": np.random.rand()/2,
                "segment_id": seg_id,
            })
            
    df = pd.DataFrame(records)
    df.to_csv(csv_file, index=False)
    return csv_file

# =====================================================================
# UNIT TESTS: 
# =====================================================================
class TestKeepRequiredColumns:
    def test_missing_required_columns_raises_value_error(self, raw_car_data):
        """Should raise ValueError when required feature columns are missing from input."""
        df_missing = raw_car_data.drop(columns=["RPM"])
        with pytest.raises(ValueError, match="FastF1 car data is missing columns"):
            keepRequiredColumns(df_missing)


# =====================================================================
# UNIT TESTS: _resample_car_data
# =====================================================================

class TestResampleCarData:
    
    def test_throttle_and_gear_filtering(self, raw_car_data):
        """Ensures out-of-range Throttle (>100 or <0) and nGear (<0) are dropped."""
        resampled = _resample_car_data(raw_car_data)
        
        # Verify bounded conditions
        assert (resampled["Throttle"] <= 100).all()
        assert (resampled["Throttle"] >= 0).all()
        assert (resampled["nGear"] >= 0).all()
        assert (resampled["nGear"] <= 8).all()
        assert len(resampled) < len(raw_car_data)

    def test_delta_time_calculation_and_first_row_drop(self, raw_car_data):
        """Validates that DeltaTime is computed accurately and first row with NaN is dropped."""
        
        raw_car_data.loc[3, "Throttle"] = 50
        raw_car_data.loc[2, "Throttle"] = 50
        raw_car_data.loc[3, "nGear"] = 2
        resampled = _resample_car_data(raw_car_data)
        
        
        assert "DeltaTime" in resampled.columns
        print(resampled)
        assert not resampled["DeltaTime"].isna().any()
        # Expect ~0.1s diff between consecutive 100ms timestamps
        np.testing.assert_allclose(resampled["DeltaTime"].values, 0.1, rtol=1e-3)

    def test_empty_dataframe_handling(self):
        """Checks graceful handling when receiving an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["Date", "RPM", "Speed", "Throttle", "nGear"])
        result = _resample_car_data(empty_df)
        assert result.empty



# =====================================================================
# UNIT TESTS: prepare_datasets
# =====================================================================

class TestPrepareDatasets:

    def test_dataset_split_and_dataloader_shapes(self, synthetic_telemetry_csv):
        """Verifies DataLoader creation, batch shapes, and tensor transpositions."""
        train_loader, val_loader, scaler, X_val_t, X_test, X_train_t = prepare_datasets(
            str(synthetic_telemetry_csv)
        )

        # 1. Output types check
        assert isinstance(X_val_t, torch.Tensor)

        # 2. Shape verification: Conv1D requires (Batch, Features/Channels, Window_Size)
        num_features = 5
        window_size = WINDOW_SIZE
        assert X_train_t.shape[1] == num_features
        assert X_train_t.shape[2] == window_size


        # 3. DataLoader batch sampling test
        batch = next(iter(train_loader))[0]
        assert batch.shape[1] == num_features
        assert batch.shape[2] == window_size

class TestInjectFault:    
    def test_valid_values(self, raw_car_data):
        resampled = _resample_car_data(raw_car_data)
        sensor_configs = [Sensors.RPM, Sensors.SPEED, Sensors.THROTTLE, Sensors.GEAR]
        
        # Set boundaries using column names
        for sensor in sensor_configs:
            resampled.iloc[0:10, resampled.columns.get_loc(sensor.name)] = sensor.max_val
            resampled.iloc[10:, resampled.columns.get_loc(sensor.name)] = 0
            
        for sensor in sensor_configs:
            # Pass the DataFrame directly instead of converting to np.array
            injected_df = inject_fault(resampled, sensor.fault.value, np.random.default_rng())
            
            # Verify bounds on the DataFrame column
            assert (injected_df[sensor.name] <= sensor.max_val).all()
            assert (injected_df[sensor.name] >= 0).all()


    @pytest.mark.parametrize("fault_type", ["rpm_spike", "speed_offset", "throttle_stuck", "gear"])
    def test_inject_fault_modifies_data(self, sample_window, fault_type):
        """Verify that inject_fault actually modifies the returned array without mutating original."""
        rng = np.random.default_rng(42)
        sample_window = pd.DataFrame(sample_window)
        original_copy = np.copy(sample_window)
        
        modified = inject_fault(sample_window, fault_type, rng)
        
        # Check that original window remained unchanged
        np.testing.assert_array_equal(sample_window, original_copy)
        # Check that output array is actually modified
        assert not np.array_equal(sample_window, modified)

    @pytest.mark.parametrize("fault_type", ["rpm_spike", "speed_offset", "throttle_stuck", "gear"])
    def test_inject_fault_output_shape_and_type(self, sample_window, fault_type):
        """Verify output shape and type preservation for all fault types."""
        rng = np.random.default_rng(42)
        
        modified = inject_fault(sample_window, fault_type, rng)
        
        assert isinstance(modified, pd.DataFrame)
        assert modified.shape == sample_window.shape

class TestGenerateEvaluationDataset:
    def test_generate_evaluation_dataset_output_shapes(self, sample_test_dataset):
        """Verify output tensor dimensions match PyTorch Conv1D expectation: (Batch, Features, Window_Size)."""
        num_windows, window_size, n_features = (len(sample_test_dataset),len(sample_test_dataset[0]),len(sample_test_dataset[0].loc[0]))
        
        # Fit dummy scaler
        scaler = StandardScaler()
        totalDF = pd.DataFrame()
        for df in sample_test_dataset:
            totalDF = pd.concat([totalDF, df])
        scaler.fit(totalDF)
        #Here the 3D array gets turned into 2D array because of the tuple of length 2
        #The -1 means inference so it infers that length of array must firstDim * secondDim both of sample_test_dataset
        X_tensor, y_test, fault_tags = generate_evaluation_dataset(
            sample_test_dataset, scaler, anomaly_ratio=0.5, seed=42
        )

        assert isinstance(X_tensor, torch.Tensor)
        assert X_tensor.shape == (num_windows, n_features, window_size)
        assert y_test.shape == (num_windows,)
        assert len(fault_tags) == num_windows


    def test_generate_evaluation_dataset_reproducibility(self, sample_test_dataset):
        """Verify that passing the same seed yields deterministic outputs."""
        scaler = StandardScaler()
        totalDF = pd.DataFrame()
        for df in sample_test_dataset:
            totalDF = pd.concat([totalDF, df])
            scaler.fit(totalDF)
        X1, y1, tags1 = generate_evaluation_dataset(sample_test_dataset, scaler, seed=123)
        X2, y2, tags2 = generate_evaluation_dataset(sample_test_dataset, scaler, seed=123)

        torch.testing.assert_close(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert tags1 == tags2
