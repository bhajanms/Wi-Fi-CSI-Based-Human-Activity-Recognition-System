import os
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.signal import savgol_filter

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = r"C:\esp32-csi-tool\dataset3"

SEG_LEN = 500
STEP = 100

BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "stand":0,
    "walk":1,
    "sit":2,
    "fall":3
}

# --------------------------------------------------
# CSI PARSER
# --------------------------------------------------

def parse_csi_string(csi_str):

    nums = re.findall(r'-?\d+', str(csi_str))

    return np.array([int(x) for x in nums])


# --------------------------------------------------
# CSI MAGNITUDE
# --------------------------------------------------

def compute_magnitude(csi):

    I = csi[0::2]
    Q = csi[1::2]

    mag = np.sqrt(I**2 + Q**2)

    return mag


# --------------------------------------------------
# TEMPORAL GRADIENT
# --------------------------------------------------

def temporal_gradient(signal):

    grad = np.diff(signal)

    grad = np.insert(grad, 0, 0)

    return grad


# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

def preprocess(signal):

    signal = savgol_filter(signal, 7, 3)

    return signal


# --------------------------------------------------
# LOAD FILE
# --------------------------------------------------

def load_file(file):

    df = pd.read_csv(file)

    csi_list = []

    for row in df["CSI_DATA"]:

        arr = parse_csi_string(row)

        if len(arr) > 0:
            csi_list.append(arr)

    return csi_list


# --------------------------------------------------
# SEGMENT
# --------------------------------------------------

def segment(data):

    segments = []

    length = len(data)

    if length < SEG_LEN:
        return segments

    for i in range(0, length - SEG_LEN + 1, STEP):

        window = data[i:i+SEG_LEN]

        segments.append(window)

    return segments


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class CSIDataset(Dataset):

    def __init__(self, dataset_dir):

        self.samples = []
        self.labels = []

        for activity in os.listdir(dataset_dir):

            activity_path = os.path.join(dataset_dir, activity)

            if not os.path.isdir(activity_path):
                continue

            label = LABEL_MAP[activity]

            for file in os.listdir(activity_path):

                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(activity_path, file)

                raw = load_file(file_path)

                features = []

                for row in raw:

                    mag = compute_magnitude(row)

                    mag = preprocess(mag)

                    grad = temporal_gradient(mag)

                    feat = np.stack([mag, grad], axis=1)

                    feat = feat.flatten()

                    features.append(feat)

                features = np.array(features)

                windows = segment(features)

                for w in windows:

                    self.samples.append(w)

                    self.labels.append(label)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        x = torch.tensor(self.samples[idx], dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


# --------------------------------------------------
# ATTENTION
# --------------------------------------------------

class Attention(nn.Module):

    def __init__(self, hidden):

        super().__init__()

        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):

        weights = torch.softmax(self.attn(x), dim=1)

        context = torch.sum(weights * x, dim=1)

        return context


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class CSIModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(128, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3)

        self.pool = nn.MaxPool1d(2)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.attention = Attention(64)

        self.fc = nn.Linear(64, 4)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = torch.relu(self.conv1(x))

        x = self.pool(x)

        x = torch.relu(self.conv2(x))

        x = self.pool(x)

        x = x.permute(0,2,1)

        out, _ = self.gru(x)

        context = self.attention(out)

        out = self.fc(context)

        return out


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

dataset = CSIDataset(DATA_DIR)

print("Total training samples:", len(dataset))

if len(dataset) == 0:

    print("ERROR: dataset empty")

    exit()

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --------------------------------------------------
# MODEL SETUP
# --------------------------------------------------

model = CSIModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

for epoch in range(EPOCHS):

    total_loss = 0

    for x, y in loader:

        x = x.to(DEVICE)

        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

torch.save(model.state_dict(), "csi_har_model.pth")

print("Training finished")