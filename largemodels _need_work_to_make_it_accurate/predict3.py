import serial
import numpy as np
import pandas as pd
import re
import time
from math import sqrt, atan2
from scipy.signal import savgol_filter
from tensorflow import keras

# =========================
# CONFIG
# =========================

PORT = "COM10"
BAUD = 115200

SEG_LEN = 150
DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

PREDICT_INTERVAL = 15

# =========================
# LOAD MODEL
# =========================

model = keras.models.load_model("csi_model3_60.keras")

# Only two activities
classes = ["stand", "walk"]

print("Model Loaded")
print("Activities:", classes)

# =========================
# SERIAL CONNECTION
# =========================

ser = serial.Serial(PORT, BAUD)
print("Listening on", PORT)

amp_buffer = []
phase_buffer = []

last_prediction_time = time.time()

# =========================
# PARSE CSI
# =========================

def parse_csi(line):

    m = re.search(r"\[(.*)\]", line)
    if not m:
        return None, None

    raw = [int(x) for x in re.split(r"[,\s]+", m.group(1)) if x]

    imag = raw[::2]
    real = raw[1::2]

    amp = [sqrt(i*i + r*r) for i,r in zip(imag,real)]
    phase = [atan2(i,r) for i,r in zip(imag,real)]

    return amp, phase


# =========================
# LIVE LOOP
# =========================

while True:

    try:

        line = ser.readline().decode(errors="ignore")

        if "CSI_DATA" not in line:
            continue

        amp, phase = parse_csi(line)

        if amp is None:
            continue

        amp_buffer.append(amp)
        phase_buffer.append(phase)

        if len(amp_buffer) > SEG_LEN:
            amp_buffer.pop(0)
            phase_buffer.pop(0)

        if time.time() - last_prediction_time > PREDICT_INTERVAL:

            if len(amp_buffer) < SEG_LEN:
                continue

            A = pd.DataFrame(amp_buffer)
            P = pd.DataFrame(phase_buffer)

            # smoothing
            for c in A.columns:
                if len(A[c]) >= 11:
                    A[c] = savgol_filter(A[c],11,3)
                    P[c] = savgol_filter(P[c],11,3)

            # remove static component
            A = A - A.mean(axis=0)
            P = P - P.mean(axis=0)

            valid = [c for c in DROP_COLS if c < A.shape[1]]

            A.drop(A.columns[valid], axis=1, inplace=True)
            P.drop(P.columns[valid], axis=1, inplace=True)

            feat = np.stack([A.values, P.values], axis=-1)

            # normalize
            feat = (feat - feat.mean()) / (feat.std()+1e-6)

            feat = np.expand_dims(feat, axis=0).astype("float32")

            # =========================
            # PREDICTION
            # =========================

            pred = model.predict(feat, verbose=0)[0]

            stand_prob = pred[0]
            walk_prob = pred[1]

            label = classes[np.argmax(pred)]
            conf = np.max(pred)

            print("\n============================")
            print("Predicted Activity:", label)
            print("Stand Probability :", round(float(stand_prob),3))
            print("Walk Probability  :", round(float(walk_prob),3))
            print("Confidence        :", round(float(conf),3))
            print("============================\n")

            last_prediction_time = time.time()

    except KeyboardInterrupt:
        print("Stopped")
        break