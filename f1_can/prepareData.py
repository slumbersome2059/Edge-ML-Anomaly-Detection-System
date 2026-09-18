import pickle
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .sensors import Sensors

from . import WINDOW_SIZE, BATCH_SIZE, STRIDE, PROCESSED_COLUMNS
NUM_FEATURES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
import time
start_time = time.time()

def give_first_train_segment_id(sorted_whole_segment_ids: list, train_segment_ids: list):
    """
    To get training data you split dataset meaning the first finishing race maybe split so you need to look at first in training data.
    You need this in calibration so that you can cut out unnecessary clipping(first finisher probably has highest speed).
    I trained the model first before knowing about needing calibration data for quantisation so if I force first to be in train_segs it messes up 
    the test and validation data so that I may test on data trained on.
     
    This is why I did this slightly slower method.
    """

    for i in sorted_whole_segment_ids:
        if i in train_segment_ids:
            return i 
def extract_windows(dframe, feature_cols):
    windows = []
    for _, group in dframe.groupby("segment_id"):#groupby is usually used with something that brings data to one cell
        #if not what it does is give you an iterable with the original dframe split by segment_id so you end up having many dframes 
        arr = group[feature_cols].to_numpy(dtype=np.float32)
        n_rows = len(group)
        segment_windows = [
            arr[start : start + WINDOW_SIZE]
            for start in range(0, n_rows - WINDOW_SIZE + 1, STRIDE)
        ]
        windows.extend(segment_windows)
    return windows
def setup_sensors(feature_cols):
    for sensor in Sensors.ALL_SENSORS:
        try:
            sensor.index = feature_cols.index(sensor.name)
        except ValueError:
            raise RuntimeError("The columns of the dataframe to be fed into the " \
            "training model does not match the ones present in the sensors class ")
def validate_dframe_cols(df):
    #This makes sure the code uses all the dframe columns and dframe columns are all code uses
    if set(df.columns) != set(Sensors.SENSOR_NAME_COLUMNS + PROCESSED_COLUMNS + ["segment_id"]):
        raise RuntimeError(
            f"Feature columns mismatch. Expected {Sensors.SENSOR_NAME_COLUMNS + PROCESSED_COLUMNS + ["segment_id"]}, "
            f"but got {df.columns}. Modify sensor_name_columns, processed_columns in init."
        )
def prepare_datasets(csv_path: str, sorted_segment_ids):
    """Processes raw CSV telemetry and extracts training loader, validation tensors,

    scaler, and raw unscaled validation/test windows.
    """
    np.random.seed(42)
    df = pd.read_csv(csv_path)
    # 1. Segment-based dataset split (70% Train, 15% Val, 15% Test)
    # Splitting by segment_id avoids temporal correlation leakage between splits
    segments = df["segment_id"].unique()
    np.random.shuffle(segments)
    print("--- %s seconds ---" % (time.time() - start_time))

    n_train = int(len(segments) * 0.70)
    n_val = int(len(segments) * 0.15)

    train_segs = segments[:n_train]
    val_segs = segments[n_train : n_train + n_val]
    test_segs = segments[n_train + n_val :]

    train_df = df[df["segment_id"].isin(train_segs)].copy()
    val_df = df[df["segment_id"].isin(val_segs)].copy()
    test_df = df[df["segment_id"].isin(test_segs)].copy()

    calib_df = df[df["segment_id"] == give_first_train_segment_id(sorted_segment_ids, train_segs)].copy()

    feature_cols = [c for c in df.columns if c != "segment_id"]

    validate_dframe_cols(df)

    setup_sensors(feature_cols)
    NUM_FEATURES = len(feature_cols)#USED IN OTHER FILE

    print("--- %s seconds ---" % (time.time() - start_time))

    # 3. Fit Scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    #this calcualates mean and SD, later used in transform to scale things
    #The scaling/transformation does is z = x - \mu/\sigma, the z score stuff
    #It is really important you fit on training data, fitting on the other data means 
    #you gain info about something that is meant to be unknown(test and val are unseen data)

    # Transform DataFrames for training and thresholding
    for dframe in [train_df, val_df, calib_df]:
        dframe[feature_cols] = scaler.transform(dframe[feature_cols].values)#this is fine the columns aren't also added to nparray

    print("--- %s seconds ---" % (time.time() - start_time))

    X_train = extract_windows(train_df, feature_cols)
    X_val = extract_windows(val_df, feature_cols)
    X_calib = extract_windows(calib_df, feature_cols)

    print("--- %s seconds ---" % (time.time() - start_time))

    X_test, anomalies, anomaly_types = generate_scaled_evaluation_dataset(test_df, scaler, feature_cols)

    print("--- %s seconds ---" % (time.time() - start_time))
    # Currently the shape is (1, Window_Size, Channels/Features)
    # Convert to PyTorch Conv1D shape: (Batch, Channels/Features, Window_Size)
    X_train_t = torch.tensor(np.array(X_train)).transpose(1, 2)
    X_val_t = torch.tensor(np.array(X_train)).transpose(1, 2)
    X_calib_t = np.transpose(np.array(X_calib), (0, 2, 1))
    X_calib_t_for_reader = np.expand_dims(X_calib_t, 1)

    # A Dataset is a way to store samples and if you need to you can store labels associated with tensors 
    # so "pos" might be associated with "good review"
    # Below we don't have any labels and just store the sample
    # A Dataset is something that represents where data is stored(it could be in 
    # some file, some nparray), the class requires only a getItem(int idx) method 
    # which should give you the (sample, label) or just sample if label is not there
    train_loader = DataLoader(
        TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
            TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False
    )
    # DataLoader is an iterable and when do next on it you end up getting a 
    # (batch_size, ...) tensor for (N, ...) shaped Dataset
    """
    - We normally pass data in batches of batch_size during training(from this we determine 
    the change in weights and biases) rather than using the whole set of data to produce 
     one change in weights and biases which means we can get many changes when we iterate 
     through one set of data
    - Everytime we iterate through data we select new batch to use to determine change 
    in weights and biases and eventually you will exhaust all the data(at that point you finish one epoch)
    - For the next epoch, you should shuffle the batches, using DataLoader has functionality to do this
    """
    # the shuffle is saying reshuffle at the end of every epoch

    return train_loader, val_loader, scaler, X_val, X_test, anomalies, anomaly_types, X_calib_t_for_reader


def inject_fault(raw_window: np.ndarray, fault_type: str, rng: np.random.Generator) -> pd.DataFrame:
    """Injects CAN bus sensor faults into raw unscaled window DataFrame."""
    result = raw_window.copy()
    randSize = rng.choice([2,3,4,5])
    start = rng.integers(0, WINDOW_SIZE - randSize)  # Fault begins partway through window

    
    # index list is, so list that stores the stuff before , on loc 
    # this list could be string labels
    # if you want to use integers no matter the index use iloc, which is loc but works only with integers
    slice_idx = slice(start, start + randSize)
    if fault_type == Sensors.RPM.fault.value:
        sensor = Sensors.RPM
        print(result[slice_idx, sensor.index])
        result[slice_idx, sensor.index] = np.clip(
            result[slice_idx, sensor.index] * rng.uniform(1.45, 1.9),
            0,
            sensor.max_val,
        )
        print(result[slice_idx, sensor.index])
    elif fault_type == Sensors.SPEED.fault.value:
        sensor = Sensors.SPEED
        result[slice_idx, sensor.index] = np.clip(
            result[slice_idx, sensor.index] + rng.choice((-1, 1)) * rng.uniform(55, 90),
            0,
            sensor.max_val,
        )
    elif fault_type == Sensors.THROTTLE.fault.value:
        sensor = Sensors.THROTTLE
        result[slice_idx, sensor.index] = rng.choice((0, sensor.max_val))
    elif fault_type == Sensors.GEAR.fault.value:  # gear manipulation
        sensor = Sensors.GEAR
        result[slice_idx, sensor.index] = np.clip(
            result[slice_idx, sensor.index] + rng.choice((-3, -2, 2, 3)),
            0,
            sensor.max_val,
        )

    return result


def generate_scaled_evaluation_dataset(#gives a scaled, UNtransposed 3D array from the dataframe
    test_df: pd.DataFrame, scaler, feature_cols, anomaly_ratio: float = 0.5, seed: int = 42
):
    """Generates test windows with an equal mix of clean data and injected fault types."""
    rng = np.random.default_rng(seed)
    processed_windows = []
    labels = []  # 0: Normal, 1: Anomaly
    fault_tags = []
    print(test_df.columns)
    
    test_windows = extract_windows(test_df, feature_cols) #in test_windows each window is a dataframe
    print(len(test_windows))
    randNums = rng.random((len(test_windows)))
    for (randNum, window) in zip(randNums, test_windows):
        is_anomaly = randNum < anomaly_ratio #this is faster than rng.choice, even faster to create all the randoms at once
        if is_anomaly:
            fault = rng.choice(Sensors.FAULT_TYPES)
            fault_str = fault.value
            modified_window = inject_fault(window, fault, rng)
            label = 1
        else:
            fault_str = "clean"
            modified_window = window.copy()
            label = 0

        scaled_window = scaler.transform(np.array(modified_window))
        processed_windows.append(scaled_window)
        labels.append(label)
        fault_tags.append(fault_str)

    X_test = np.array(processed_windows, dtype=np.float32)
    y_test = np.array(labels, dtype=int)

    return X_test, y_test, fault_tags


def save_processed_data(output_dir: Path, train_loader, val_loader, scaler, val_scaled_windows, test_scaled_windows, anomalies, anomaly_types, X_calib_t_for_reader):
    """Serializes dataset splits and scaler into pickle files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_loader.pkl", "wb") as f:
        pickle.dump(train_loader, f)
    with open(output_dir / "val_loader.pkl", "wb") as f:
        pickle.dump(val_loader, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "val_scaled_windows.pkl", "wb") as f:
        pickle.dump(val_scaled_windows, f)
    with open(output_dir / "test_scaled_windows.pkl", "wb") as f:
        pickle.dump(test_scaled_windows, f)
    with open(output_dir / "anomalies.pkl", "wb") as f:
            pickle.dump(anomalies, f)
    with open(output_dir / "anomaly_types.pkl", "wb") as f:
            pickle.dump(anomaly_types, f)
    with open(output_dir / "X_calib_t_for_reader.pkl", "wb") as f:
            pickle.dump(X_calib_t_for_reader, f)
    