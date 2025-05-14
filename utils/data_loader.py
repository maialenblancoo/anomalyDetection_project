import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    """Carga datos desde un archivo CSV."""
    logger.info(f"Charging data from {filepath}")
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    logger.info(f"Charged data: {len(df)} registers")
    return df

def load_anomaly_labels(labels_path, windows_path):
    """Carga etiquetas de anomalías desde archivos JSON."""
    logger.info(f"Charging labels from {labels_path} and {windows_path}")
    with open(labels_path, 'r') as f:
        labels = json.load(f)

    with open(windows_path, 'r') as f:
        windows = json.load(f)

    return labels, windows

def normalize_data(data):
    """Normaliza los datos usando MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    return data_scaled, scaler

def create_sequences(data, seq_length=150, step=10):
    """Crea secuencias para el entrenamiento de autoencoders."""
    sequences = []
    for i in range(0, len(data) - seq_length, step):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

def create_labels_array(df, anomaly_timestamps, anomaly_windows, window_length=150, step=10):
    """Crea etiquetas para las secuencias."""
    labels = np.zeros(len(df))

    for window in anomaly_windows:
        start = pd.to_datetime(window[0])
        end = pd.to_datetime(window[1])
        mask = (df.index >= start) & (df.index <= end)
        labels[mask] = 1

    sequence_labels = []
    for i in range(0, len(labels) - window_length, step):
        if np.sum(labels[i:i+window_length]) > 0:
            sequence_labels.append(1)
        else:
            sequence_labels.append(0)

    return np.array(sequence_labels)

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_ratio), shuffle=False, random_state=42
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio_adjusted), shuffle=False, random_state=42
    )

    logger.info(f"Data division: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_loaders(X_train, X_val, X_test, batch_size=64):
    """Prepara los data loaders para PyTorch."""
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)

    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_prediction_sequences(data, seq_length=150, horizon=24):
    """Crea secuencias para modelos de predicción."""
    X, y = [], []
    for i in range(0, len(data) - seq_length - horizon, 1):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length:i+seq_length+horizon])
    return np.array(X), np.array(y)