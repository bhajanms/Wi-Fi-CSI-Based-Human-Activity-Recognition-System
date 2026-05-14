import re
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import serial
from scipy.signal import savgol_filter

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

MODEL_PATH = "csi_har_model.pth"

WINDOW_SIZE = 500
PRED_INTERVAL = 10

TARGET_SUBCARRIERS = 64   # must match training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    0: "stand",
    1: "walk",
    2: "sit",
    3: "fall"
}

print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Serial:", SERIAL_PORT)

# --------------------------------------------------
# CSI PARSER
# --------------------------------------------------

def parse_csi_string(csi_str):

    try:
        nums = re.findall(r"-?\d+", csi_str)

        if not nums:
            return None

        return np.array([int(x) for x in nums])

    except Exception:
        traceback.print_exc()
        return None


# --------------------------------------------------
# CSI MAGNITUDE
# --------------------------------------------------

def compute_magnitude(csi):

    try:

        I = csi[0::2]
        Q = csi[1::2]

        min_len = min(len(I), len(Q))

        I = I[:min_len].astype(float)
        Q = Q[:min_len].astype(float)

        mag = np.sqrt(I**2 + Q**2)

        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

        return mag

    except Exception:
        traceback.print_exc()
        return None


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

    try:
        return savgol_filter(signal, 7, 3)
    except:
        return signal


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

        x = x.permute(0, 2, 1)

        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.permute(0, 2, 1)

        out, _ = self.gru(x)

        context = self.attention(out)

        out = self.fc(context)

        return out


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model = CSIModel().to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    model.eval()

    print("Model loaded successfully")

except Exception:

    traceback.print_exc()

    exit()


# --------------------------------------------------
# SERIAL CONNECTION
# --------------------------------------------------

try:

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

    print("Serial connected")

except Exception:

    traceback.print_exc()

    exit()


# --------------------------------------------------
# LIVE LOOP
# --------------------------------------------------

buffer = []

last_prediction_time = time.time()

print("\nListening for CSI data...\n")

while True:

    try:

        line = ser.readline().decode("utf-8", errors="ignore")

        if "CSI_DATA" not in line:
            continue

        csi_part = line.split("CSI_DATA")[-1]

        csi = parse_csi_string(csi_part)

        if csi is None:
            continue

        mag = compute_magnitude(csi)

        if mag is None:
            continue

        mag = preprocess(mag)

        grad = temporal_gradient(mag)

        # --------------------------------------------------
        # FIX FEATURE SIZE (64 SUBCARRIERS)
        # --------------------------------------------------

        if len(mag) < TARGET_SUBCARRIERS:

            mag = np.pad(mag, (0, TARGET_SUBCARRIERS - len(mag)))

        if len(grad) < TARGET_SUBCARRIERS:

            grad = np.pad(grad, (0, TARGET_SUBCARRIERS - len(grad)))

        mag = mag[:TARGET_SUBCARRIERS]
        grad = grad[:TARGET_SUBCARRIERS]

        feat = np.stack([mag, grad], axis=1).flatten()

        buffer.append(feat)

        if len(buffer) % 50 == 0:

            print("CSI packets collected:", len(buffer))

        if len(buffer) > 2000:

            buffer = buffer[-2000:]

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        if time.time() - last_prediction_time >= PRED_INTERVAL:

            if len(buffer) >= WINDOW_SIZE:

                try:

                    data = np.array(buffer[-WINDOW_SIZE:])

                    x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():

                        output = model(x)

                        pred = torch.argmax(output, dim=1).item()

                    activity = LABEL_MAP[pred]

                    print("\n==============================")
                    print("PREDICTED ACTIVITY:", activity)
                    print("==============================\n")

                except Exception:

                    print("Prediction error")

                    traceback.print_exc()

            else:

                print("Waiting for enough CSI packets...")

            last_prediction_time = time.time()

    except KeyboardInterrupt:

        print("\nStopping live prediction")

        ser.close()

        break

    except Exception:

        traceback.print_exc()