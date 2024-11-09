import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class SignalDataset(Dataset):
    def __init__(self, csv_file, root_path, dwn_factor, time_length, mode="hr", signal_scaler=None, label_scaler=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_path = root_path
        self.dwn_factor = dwn_factor
        self.time_length = time_length
        self.seq_len = int(30 / dwn_factor) * time_length
        self.mode = mode
        self.signal_scaler = signal_scaler
        self.label_scaler = label_scaler

        # Load data from CSV
        self.mred, self.mblue, self.mgreen, self.hr, self.spo2 = self._load_signals()

        # Standardize the signals using the provided signal scaler
        if self.signal_scaler:
            if self.mode == 'hr':
                # For HR mode, scale only the mred channel
                self.mred = self.signal_scaler.transform(self.mred.reshape(-1, 1)).flatten()
            else:
                # For SpO2 mode, scale all three channels
                self.mred = self.signal_scaler.transform(self.mred.reshape(-1, 1)).flatten()
                self.mgreen = self.signal_scaler.transform(self.mgreen.reshape(-1, 1)).flatten()
                self.mblue = self.signal_scaler.transform(self.mblue.reshape(-1, 1)).flatten()

        # Normalize the labels using the provided label scaler
        if self.label_scaler:
            if self.mode == 'hr':
                self.hr = self.label_scaler.transform(self.hr.reshape(-1, 1)).flatten()
            else:
                self.spo2 = self.label_scaler.transform(self.spo2.reshape(-1, 1)).flatten()

        # Calculate labels based on the mode
        self.labels = self._calculate_labels()

    def _load_signals(self):
        mred, mblue, mgreen = [], [], []
        hr, spo2 = [], []

        for i in range(len(self.csv_file)):
            try:
                signals = np.load(f'{self.root_path}/MTHS/Data/signal_{self.csv_file.iloc[i]["patient_ID"]}.npy')
                labels = np.load(f'{self.root_path}/MTHS/Data/label_{self.csv_file.iloc[i]["patient_ID"]}.npy')
            except FileNotFoundError:
                continue  # Skip if file is not found

            mred.extend(signals[:, 0])
            mgreen.extend(signals[:, 1])
            mblue.extend(signals[:, 2])
            hr.extend(labels[:, 0])
            spo2.extend(labels[:, 1])

        # Downsample signals
        mred, mblue, mgreen = mred[::self.dwn_factor], mblue[::self.dwn_factor], mgreen[::self.dwn_factor]
        return np.array(mred), np.array(mblue), np.array(mgreen), np.array(hr), np.array(spo2)

    def _calculate_labels(self):
        # Calculate average labels for each sequence
        num_sequences = len(self.mred) // self.seq_len
        hr_labels = np.zeros((num_sequences,))
        spo2_labels = np.zeros((num_sequences,))

        for i in range(num_sequences):
            hr_labels[i] = np.mean(self.hr[i * self.time_length: (i + 1) * self.time_length])
            spo2_labels[i] = np.mean(self.spo2[i * self.time_length: (i + 1) * self.time_length])

        # Return the appropriate label based on mode
        return hr_labels if self.mode == 'hr' else spo2_labels

    def __len__(self):
        return len(self.mred) // self.seq_len

    def __getitem__(self, idx):
        start, end = idx * self.seq_len, (idx + 1) * self.seq_len
        ppg = np.zeros((self.seq_len, 3))

        if self.mode == 'hr':
            ppg[:, 0] = self.mred[start:end]
            ppg[:, 1] = self.mred[start:end]
            ppg[:, 2] = self.mred[start:end]
        else:
            ppg[:, 0] = self.mred[start:end]
            ppg[:, 1] = self.mblue[start:end]
            ppg[:, 2] = self.mgreen[start:end]

        label = self.labels[idx]

        ppg_tensor = torch.tensor(ppg, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return ppg_tensor, label_tensor


# Standardization process for signals
def fit_scaler_on_train_data(train_dataset, mode):
    scaler = StandardScaler()

    if mode == 'hr':
        # For "hr" mode, only use mred channel for scaling
        all_signals = train_dataset.mred.reshape(-1, 1)
    else:
        # For "spo2" mode, use all three channels (mred, mgreen, mblue) for scaling
        all_signals = np.concatenate([train_dataset.mred, train_dataset.mgreen, train_dataset.mblue]).reshape(-1, 1)

    scaler.fit(all_signals)
    return scaler


# Standardization process for labels
def fit_label_scaler_on_train_data(train_dataset, mode):
    scaler = StandardScaler()

    if mode == 'hr':
        # For "hr" mode, use hr labels
        label_data = train_dataset.hr.reshape(-1, 1)
    else:
        # For "spo2" mode, use spo2 labels
        label_data = train_dataset.spo2.reshape(-1, 1)

    scaler.fit(label_data)
    return scaler


def data_loader(train_csv, val_csv, test_csv, root_path, dwn_factor, time_length, batch_size=32, mode="hr"):
    # Create training dataset and fit signal scaler for signals
    train_dataset = SignalDataset(train_csv, root_path, dwn_factor, time_length, mode)
    signal_scaler = fit_scaler_on_train_data(train_dataset, mode)

    # Fit the label scaler based on the mode (hr or spo2)
    label_scaler = fit_label_scaler_on_train_data(train_dataset, mode)

    # Create datasets with the fitted scalers for train, validation, and test
    train_dataset = SignalDataset(train_csv, root_path, dwn_factor, time_length, mode, signal_scaler, label_scaler)
    val_dataset = SignalDataset(val_csv, root_path, dwn_factor, time_length, mode, signal_scaler, label_scaler)
    test_dataset = SignalDataset(test_csv, root_path, dwn_factor, time_length, mode, signal_scaler, label_scaler)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for ppg, label in train_loader:
        print("PPG batch shape:", ppg.shape)  # Should be [batch_size, seq_len, 3]
        print("Label batch shape:", label.shape)  # Should be [batch_size]
        break

    return train_loader, val_loader, test_loader