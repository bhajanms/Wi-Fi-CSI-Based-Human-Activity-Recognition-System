import serial
import numpy as np
import pandas as pd
import re
import requests

from math import sqrt
from scipy.signal import savgol_filter, butter, filtfilt

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# CONFIG
# =========================

PORT = "COM10"
BAUD = 115200

SEG_LEN = 700

DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

MODEL_PATH = "cnn_gru_csi_model.pth"
CLASS_PATH = "cnn_gru_classes.npy"

API_URL = "http://172.20.10.8:5000/api/activity/update"

PERSON_ID = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CLASSES
# =========================

classes = np.load(CLASS_PATH, allow_pickle=True)

# =========================
# CSI PARSER
# =========================

def parse_csi(csi_string):

    nums = [int(x) for x in re.findall(r'-?\d+', str(csi_string))]

    imag = nums[::2]
    real = nums[1::2]

    amp = []

    for i, r in zip(imag, real):
        amp.append(sqrt(i*i + r*r))

    return np.array(amp)

# =========================
# HIGH PASS FILTER
# =========================

def highpass(signal):

    b, a = butter(3, 0.1, btype="highpass")

    return filtfilt(b, a, signal)

# =========================
# MODEL
# =========================

class CNN_GRU_Model(nn.Module):

    def __init__(self, num_features, num_classes):

        super().__init__()

        self.conv1 = nn.Conv1d(num_features, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

        self.drop = nn.Dropout(0.4)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.permute(0,2,1)

        out,_ = self.gru(x)

        x = out[:,-1,:]

        x = self.drop(F.relu(self.fc1(x)))

        return self.fc2(x)

# =========================
# LOAD MODEL
# =========================

model = CNN_GRU_Model(28, len(classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded!")

# =========================
# SERIAL
# =========================

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Listening for CSI packets...")

buffer = []

# =========================
# MAIN LOOP
# =========================

while True:

    try:

        line = ser.readline().decode(errors="ignore")

        if "CSI_DATA" not in line:
            continue

        feat = parse_csi(line)

        buffer.append(feat)

        if len(buffer) < SEG_LEN:
            continue

        buffer = buffer[-SEG_LEN:]

        A = pd.DataFrame(buffer)

        # smoothing
        for c in A.columns:
            if len(A[c]) >= 11:
                A[c] = savgol_filter(A[c],11,3)

        # mean removal
        A = A - A.mean(axis=0)

        # highpass
        for c in A.columns:
            A[c] = highpass(A[c])

        # drop noisy carriers
        valid = [c for c in DROP_COLS if c < A.shape[1]]
        A.drop(A.columns[valid], axis=1, inplace=True)

        # select stable carriers
        A = A.iloc[:,12:40]

        seg = A.values

        seg = (seg - seg.mean())/(seg.std()+1e-6)

        X = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():

            out = model(X)

            probs = torch.softmax(out,1)

            conf = torch.max(probs).item()

            pred = torch.argmax(out,1).item()

            activity = classes[pred]

        print(f"Predicted Activity: {activity} | Confidence: {conf:.3f}")

        # =========================
        # SEND TO SERVER
        # =========================

        data = {
            "person_id": PERSON_ID,
            "activity": str(activity),
            "confidence": float(conf)
        }

        try:
            requests.post(API_URL, json=data, timeout=2)
        except:
            print("Server not reachable")

    except Exception as e:

        print("Error:", e)