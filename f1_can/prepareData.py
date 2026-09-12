import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .sensors import Sensors
import pickle
from . import WINDOW_SIZE, BATCH_SIZE, STRIDE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# PHASE 1: DATA PREPROCESSING & WINDOWING
# ==========================================
def load_and_split_segments(csv_path: str):
    """Handles loading and segment-based splitting."""
    np.random.seed(42)
    df = pd.read_csv(csv_path)

    # 1. Segment-based dataset split (70% Train, 15% Val, 15% Test)
    # Splitting by segment_id avoids temporal correlation leakage between splits
    segments = df["segment_id"].unique()
    np.random.shuffle(segments)

    n_train = int(len(segments) * 0.70)
    n_val = int(len(segments) * 0.15)

    return (
        segments[:n_train],
        segments[n_train : n_train + n_val],
        segments[n_train + n_val:]
    )
def extract_windows(dframe, feature_cols):
    windows = []
    for _, group in dframe.groupby("segment_id"):
        for start in range(0, len(group) - WINDOW_SIZE + 1, STRIDE):
            windows.append(group[[*feature_cols]].iloc[start : start + WINDOW_SIZE])#windows is a 3d array
    return np.array(windows, dtype=np.float32)
def windows_df(segs, *other_segs):
    #This MODIFIES the dfs given, think pass by reference
    dfs = []
    for i in (segs, *other_segs):
        dfs.append(df[df["segment_id"].isin(i)].copy())
    scaler = StandardScaler()
    feature_cols = tuple(set(dfs[0].columns) - {"segment_id"})
    scaler.fit(dfs[0][[*feature_cols]])
    #this calcualates mean and SD, later used in transform to scale things
    #The scaling/transformation does is z = x - \mu/\sigma, the z score stuff
    #It is really important you fit on training data, fitting on the other data means 
    #you gain info about something that is meant to be unknown(test and val are unseen data)

    transformed = []
    for df in dfs:
        df[[*feature_cols]] = scaler.transform(df[[*feature_cols]])
        transformed.append(extract_windows(df, feature_cols))
    return transformed
def get_loader_for_window_list(windowList):
    # Currently the shape is (N, Window_Size, Channels/Features)
    # Convert to PyTorch Conv1D shape: (Batch, Channels/Features, Window_Size)
    windowList_t = torch.tensor(np.array(windowList)).transpose(1, 2)
    return DataLoader(TensorDataset(windowList_t), batch_size=BATCH_SIZE, shuffle=True)
    # A Dataset is a way to store samples and if you need to you can store labels associated with tensors 
    # so "pos" might be associated with "good review"
    # Below we don't have any labels and just store the sample
    # A Dataset is something that represents where data is stored(it could be in 
    # some file, some nparray), the class requires only a getItem(int idx) method 
    # which should give you the (sample, label) or just sample if label is not there
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

def prepare_datasets(csv_path: str):
    np.random.seed(42)
    df = pd.read_csv(csv_path)
    
    # 1. Segment-based dataset split (70% Train, 15% Val, 15% Test)
    # Splitting by segment_id avoids temporal correlation leakage between splits
    segments = df["segment_id"].unique()
    np.random.shuffle(segments)

    n_train = int(len(segments) * 0.70)
    n_val = int(len(segments) * 0.15)

    train_segs = segments[:n_train]
    val_segs = segments[n_train : n_train + n_val]
    test_segs = segments[n_train + n_val :]

    train_df = df[df["segment_id"].isin(train_segs)].copy()
    val_df = df[df["segment_id"].isin(val_segs)].copy()
    test_df = df[df["segment_id"].isin(test_segs)].copy()
    # 2. Feature Scaling (Fit ONLY on training data to prevent leakage)
    scaler = StandardScaler()
    FEATURE_COLUMNS = tuple(set(df.columns).difference(["segment_id"]))
    scaler.fit(train_df[[*FEATURE_COLUMNS]])
    for dframe in [train_df, val_df, test_df]:
        dframe[[*FEATURE_COLUMNS]] = scaler.transform(dframe[[*FEATURE_COLUMNS]])#perform the z score transformation

    # 3. Sliding Window Extraction (Respecting Segment Boundaries)
    def extract_windows(dframe):
        windows = []
        for _, group in dframe.groupby("segment_id"):
            for start in range(0, len(group) - WINDOW_SIZE + 1, STRIDE):
                windows.append(group[[*FEATURE_COLUMNS]].iloc[start : start + WINDOW_SIZE])#windows is a 3d array
        return np.array(windows, dtype=np.float32)

    X_train = extract_windows(train_df)
    X_val = extract_windows(val_df)
    X_test = extract_windows(test_df)
    X_train_t = torch.tensor(np.array(X_train)).transpose(1, 2)
    X_val_t = torch.tensor(np.array(X_val)).transpose(1, 2)

    train_loader = DataLoader(
        TensorDataset(X_train_t), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t), batch_size=BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader, scaler, X_val_t, X_test, X_train_t

def inject_fault(raw_window: pd.DataFrame, fault_type: str, rng: np.random.Generator) -> np.ndarray:
    """Injects CAN bus sensor faults into raw unscaled window which has shape (WINDOW_SIZE, n_features)."""

    result = raw_window.copy()
    randSize = rng.choice([2,3,4,5])
    start = rng.integers(0, WINDOW_SIZE - randSize)  # Fault begins partway through window

    # Retrieve the target rows' index to use with .loc
    target_idx = result.index[start:start + randSize]#this extracts whatever the 
    # index list is, so list that stores the stuff before , on loc 
    # this list could be string labels
    # if you want to use integers no matter the index use iloc, which is loc but works only with integers

    if fault_type == Sensors.RPM.fault.value:
        sensor = Sensors.RPM
        result.loc[target_idx, sensor.name] = np.clip(
            result.loc[target_idx, sensor.name] * rng.uniform(1.45, 1.9), 
            0, 
            sensor.max_val
        )
    elif fault_type == Sensors.SPEED.fault.value:
        sensor = Sensors.SPEED
        result.loc[target_idx, sensor.name] = np.clip(
            result.loc[target_idx, sensor.name] + rng.choice((-1, 1)) * rng.uniform(55, 90), 
            0, 
            sensor.max_val
        )
    elif fault_type == Sensors.THROTTLE.fault.value:
        sensor = Sensors.THROTTLE
        result.loc[target_idx, sensor.name] = rng.choice((0, sensor.max_val))
    else:  # gear manipulation
        sensor = Sensors.GEAR
        result.loc[target_idx, sensor.name] = np.clip(
            result.loc[target_idx, sensor.name] + rng.choice((-3, -2, 2, 3)), 
            0, 
            sensor.max_val
        )
    
    return result


def generate_evaluation_dataset(
    raw_test_windows: np.ndarray, scaler, anomaly_ratio: float = 0.5, seed: int = 42
):
    """
    Generates balanced test windows with equal mix of clean data and 4 injected fault types.
    Normalizes data using the pre-fitted scaler.
    raw_test_windows has format (num_test_windows, WINDOW_SIZE, n_features)
    """
    rng = np.random.default_rng(seed)
    
    processed_windows = []
    labels = []  # 0: Normal, 1: Anomaly
    fault_tags = []

    for window in raw_test_windows:
        fault_str = ""
        is_anomaly = rng.random() < anomaly_ratio
        if is_anomaly:
            fault = rng.choice(Sensors.FAULT_TYPES)
            fault_str = fault.value
            modified_window = inject_fault(window, fault, rng)
            label = 1
        else:
            fault_str = "clean"
            modified_window = window.copy()
            label = 0

        # Scale features using fitted scaler parameters
        scaled_window = scaler.transform(np.array(modified_window))
        processed_windows.append(scaled_window)
        labels.append(label)
        fault_tags.append(fault_str)

    X_test_scaled = np.array(processed_windows, dtype=np.float32)
    # Transpose to PyTorch Conv1D layout: (Batch, Channels/Features, Window_Size)
    X_test_tensor = torch.tensor(X_test_scaled).transpose(1, 2)
    y_test = np.array(labels, dtype=int)

    return X_test_tensor, y_test, fault_tags
    