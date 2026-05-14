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

LIVE_FILE = r"C:\esp32-csi-tool\active_ap\live_csi1.csv"

print("Loading model...")

model = keras.models.load_model("new_csi_model.keras")
class_names = np.load("new_class_names.npy", allow_pickle=True)

print("Classes:", class_names)
print("\nStarting Live Prediction...\n")

packet_buffer = deque(maxlen=SEG_LEN)
prob_buffer = deque(maxlen=5)

last_len = 0
last_prediction_time = time.time()
last_activity = None

# =========================
# PARSER
# =========================

def parse_row(val):

    m = re.search(r"\[(.*)\]", str(val))
    if not m:
        return None

    raw = [int(x) for x in re.split(r"[,\s]+", m.group(1)) if x]

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

    df = pd.read_csv(LIVE_FILE)

    if "CSI_DATA" not in df.columns:
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

    amps = np.array([p[0] for p in packet_buffer])
    phases = np.array([p[1] for p in packet_buffer])

    for c in range(amps.shape[1]):
        if amps.shape[0] >= 11:
            amps[:,c] = savgol_filter(amps[:,c],11,3)
            phases[:,c] = savgol_filter(phases[:,c],11,3)

    # 🔥 static removal
    amps = amps - np.mean(amps, axis=0)
    phases = phases - np.mean(phases, axis=0)

    valid = [c for c in DROP_COLS if c < amps.shape[1]]
    amps = np.delete(amps, valid, axis=1)
    phases = np.delete(phases, valid, axis=1)

    feat = np.stack([amps, phases], axis=-1)
    X = feat[np.newaxis,...]

    # 🔥 per-window normalization
    X = (X - X.mean()) / (X.std()+1e-6)

    probs = model.predict(X, verbose=0)[0]
    prob_buffer.append(probs)

    avg_probs = np.mean(prob_buffer, axis=0)
    final_idx = int(np.argmax(avg_probs))
    final_conf = float(avg_probs[final_idx])

    current_activity = class_names[final_idx]
    current_time = datetime.now().strftime("%H:%M:%S")

    if current_activity != last_activity:
        print(f"[{current_time}] Activity →",
              current_activity,
              "| Confidence:",
              round(final_conf,3))
        last_activity = current_activity