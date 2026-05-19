# =========================
# FORCE CPU
# =========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
import re
import time
from math import sqrt, atan2
from scipy.signal import savgol_filter
from tensorflow import keras
from collections import deque
from datetime import datetime

# =========================
# CONFIG
# =========================

SEG_LEN = 150
PREDICT_INTERVAL = 2.0
DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

BASE_PATH = r"C:\esp32-csi-tool"
LIVE_FILE = r"C:\esp32-csi-tool\active_ap\live_csi1.csv"

MODEL_PATH = os.path.join(BASE_PATH, "model_csi_new.keras")
MEAN_PATH  = os.path.join(BASE_PATH, "norm_mean.npy")
STD_PATH   = os.path.join(BASE_PATH, "norm_std.npy")
CLASS_PATH = os.path.join(BASE_PATH, "class_names.npy")

SMOOTH_WINDOW = 5

# =========================
# LOAD MODEL
# =========================

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
class_names = np.load(CLASS_PATH, allow_pickle=True)
mean = np.load(MEAN_PATH)
std  = np.load(STD_PATH)

print("✅ Model Loaded")
print("Classes:", class_names)
print("\nStarting Stable Live Prediction...\n")

# =========================
# BUFFERS
# =========================

packet_buffer = deque(maxlen=SEG_LEN)
prob_buffer = deque(maxlen=SMOOTH_WINDOW)

last_len = 0
last_prediction_time = time.time()

# =========================
# CSI PARSER
# =========================

def parse_row(val):

    m = re.search(r"\[(.*)\]", str(val))
    if not m:
        return None

    raw = [int(x) for x in re.split(r"[,\s]+", m.group(1)) if x]
    if len(raw) < 2:
        return None

    imag = raw[::2]
    real = raw[1::2]

    amp = np.sqrt(np.array(imag)**2 + np.array(real)**2)
    phase = np.arctan2(imag, real)

    return amp, phase

# =========================
# LIVE LOOP
# =========================

while True:

    if not os.path.exists(LIVE_FILE):
        time.sleep(0.2)
        continue

    try:
        df = pd.read_csv(LIVE_FILE)
    except:
        time.sleep(0.1)
        continue

    if "CSI_DATA" not in df.columns:
        time.sleep(0.1)
        continue

    current_len = len(df)

    if current_len > last_len:
        new_rows = df["CSI_DATA"].iloc[last_len:]
        last_len = current_len

        for row in new_rows:
            parsed = parse_row(row)
            if parsed is not None:
                packet_buffer.append(parsed)

    if time.time() - last_prediction_time < PREDICT_INTERVAL:
        time.sleep(0.05)
        continue

    last_prediction_time = time.time()

    if len(packet_buffer) < SEG_LEN:
        print("Collecting packets...")
        continue

    # =========================
    # BUILD WINDOW
    # =========================

    amps = np.array([p[0] for p in packet_buffer])
    phases = np.array([p[1] for p in packet_buffer])

    for c in range(amps.shape[1]):
        if amps.shape[0] >= 11:
            amps[:,c] = savgol_filter(amps[:,c], 11, 3)
            phases[:,c] = savgol_filter(phases[:,c], 11, 3)

    valid = [c for c in DROP_COLS if c < amps.shape[1]]
    amps = np.delete(amps, valid, axis=1)
    phases = np.delete(phases, valid, axis=1)

    feat = np.stack([amps, phases], axis=-1).astype("float32")
    X = feat[np.newaxis, ...]

    # 🔥 TRAINING NORMALIZATION
    X = (X - mean) / (std + 1e-6)

    # =========================
    # PREDICT
    # =========================

    probs = model.predict(X, verbose=0)[0]
    prob_buffer.append(probs)

    avg_probs = np.mean(prob_buffer, axis=0)

    final_idx = int(np.argmax(avg_probs))
    final_conf = float(avg_probs[final_idx])

    current_time = datetime.now().strftime("%H:%M:%S")

    print(f"[{current_time}] Activity:",
          class_names[final_idx],
          "| Confidence:",
          round(final_conf,3))