# =========================================
# WIFI CSI LIVE PREDICTION
# FIX 1 + FIX 4 ONLY
# =========================================

import re
import pickle
import serial
import requests

import numpy as np
import pandas as pd

from collections import Counter
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# =========================================
# CONFIG
# =========================================

MODEL_DIR = r"C:\esp32-csi-tool\model4"

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

WINDOW_SIZE = 128
STRIDE = 64

# FIX 1
MOVING_AVG = 2

NUM_FEATURES = 54

CONFIDENCE_THRESHOLD = 0.60

API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# REQUIRED PACKETS
# =========================================

REQUIRED_PACKETS = (
    WINDOW_SIZE
    + MOVING_AVG
    - 1
)

# =========================================
# LOAD MODEL
# =========================================

model = load_model(
    f"{MODEL_DIR}/cnn_gru_model.keras"
)

scaler = pickle.load(
    open(
        f"{MODEL_DIR}/cnn_scaler.pkl",
        "rb"
    )
)

labels = pickle.load(
    open(
        f"{MODEL_DIR}/labels.pkl",
        "rb"
    )
)

print("Model Loaded")

# =========================================
# FILTER
# =========================================

def butter_filter(data, cutoff=0.1, order=3):

    b, a = butter(order, cutoff)

    return filtfilt(
        b,
        a,
        data,
        axis=0
    )

# =========================================
# PARSE CSI
# =========================================

def parse_csi_line(line):

    try:

        match = re.search(
            r"\[(.*)\]",
            line
        )

        if not match:
            return None

        raw = match.group(1)

        values = [
            int(x)
            for x in raw.split()
        ]

        return np.array(
            values,
            dtype=np.float32
        )

    except:

        return None

# =========================================
# AMPLITUDE
# =========================================

def convert_amplitude(arr):

    imag = arr[::2]

    real = arr[1::2]

    amp = np.sqrt(
        imag**2 + real**2
    )

    return amp

# =========================================
# SUBCARRIERS
# =========================================

def select_subcarriers(df):

    part1 = df.iloc[:, 5:32]

    part2 = df.iloc[:, 33:60]

    return pd.concat(
        [part1, part2],
        axis=1
    )

# =========================================
# MOVING AVG
# =========================================

def moving_average(df):

    return df.rolling(
        MOVING_AVG
    ).mean().dropna()

# =========================================
# FIX 4
# REDUCED SMOOTHING
# =========================================

prediction_history = []

SMOOTHING_SIZE = 2

def smooth_prediction(activity):

    prediction_history.append(activity)

    if len(prediction_history) > SMOOTHING_SIZE:

        prediction_history.pop(0)

    counter = Counter(prediction_history)

    return counter.most_common(1)[0][0]

# =========================================
# SERIAL
# =========================================

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE
)

print("Listening CSI...")

buffer = []

# =========================================
# LOOP
# =========================================

while True:

    try:

        line = ser.readline().decode(
            errors="ignore"
        )

        if "CSI_DATA" not in line:
            continue

        parsed = parse_csi_line(line)

        if parsed is None:
            continue

        amp = convert_amplitude(parsed)

        buffer.append(amp)

        print("Packets :", len(buffer))

        if len(buffer) >= REQUIRED_PACKETS:

            df = pd.DataFrame(buffer)

            df = select_subcarriers(df)

            df = moving_average(df)

            features = pd.DataFrame(
                butter_filter(
                    df.values
                )
            )

            features = features.iloc[
                :WINDOW_SIZE
            ]

            X = features.values

            if X.shape != (
                WINDOW_SIZE,
                NUM_FEATURES
            ):

                print(
                    "Invalid Shape :",
                    X.shape
                )

                buffer = buffer[STRIDE:]

                continue

            X_scaled = scaler.transform(X)

            X_final = X_scaled.reshape(
                1,
                WINDOW_SIZE,
                NUM_FEATURES
            )

            pred = model.predict(
                X_final,
                verbose=0
            )

            print("Raw Prediction :", pred)

            activity_id = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            activity = labels[
                activity_id
            ]

            if confidence < CONFIDENCE_THRESHOLD:

                print(
                    "Low Confidence"
                )

                buffer = buffer[STRIDE:]

                continue

            # FIX 4
            final_activity = smooth_prediction(
                activity
            )

            print("\n================")
            print(
                "Activity :",
                final_activity
            )
            print(
                "Confidence :",
                round(confidence, 4)
            )
            print("================\n")

            payload = {
                "activity": final_activity,
                "confidence": confidence
            }

            try:

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=5
                )

                print(
                    "Server :",
                    response.text
                )

            except Exception as e:

                print(
                    "API Failed :",
                    e
                )

            buffer = buffer[STRIDE:]

    except KeyboardInterrupt:

        print("Stopped")

        ser.close()

        break

    except Exception as e:

        print(
            "Runtime Error :",
            e
        )