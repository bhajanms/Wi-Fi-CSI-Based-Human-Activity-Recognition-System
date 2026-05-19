# =========================================
# FINAL WIFI CSI LIVE PREDICTION
# CNN-GRU
# WALK / STATIC / FALL
# FIXED WINDOW SIZE VERSION
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

MODEL_DIR = r"C:\esp32-csi-tool\model3"

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

WINDOW_SIZE = 128
STRIDE = 64
MOVING_AVG = 5

NUM_FEATURES = 54

CONFIDENCE_THRESHOLD = 0.60

# =========================================
# REQUIRED PACKETS
# =========================================

REQUIRED_PACKETS = (
    WINDOW_SIZE
    + MOVING_AVG
    - 1
)

# =========================================
# API URL
# =========================================

API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# LOAD MODEL
# =========================================

print("\nLoading Model...\n")

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

print("Model Loaded Successfully")

# =========================================
# BUTTER FILTER
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
# PARSE CSI LINE
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

    except Exception as e:

        print("Parse Error :", e)

        return None

# =========================================
# CSI -> AMPLITUDE
# =========================================

def convert_amplitude(arr):

    imag = arr[::2]

    real = arr[1::2]

    amp = np.sqrt(
        imag**2 + real**2
    )

    return amp

# =========================================
# SUBCARRIER SELECTION
# =========================================

def select_subcarriers(df):

    part1 = df.iloc[:, 5:32]

    part2 = df.iloc[:, 33:60]

    return pd.concat(
        [part1, part2],
        axis=1
    )

# =========================================
# MOVING AVERAGE
# =========================================

def moving_average(df):

    return df.rolling(
        MOVING_AVG
    ).mean().dropna()

# =========================================
# PREDICTION SMOOTHING
# =========================================

prediction_history = []

SMOOTHING_SIZE = 5

def smooth_prediction(activity):

    prediction_history.append(activity)

    if len(prediction_history) > SMOOTHING_SIZE:

        prediction_history.pop(0)

    counter = Counter(prediction_history)

    final_prediction = counter.most_common(1)[0][0]

    return final_prediction

# =========================================
# SERIAL CONNECTION
# =========================================

print("\nConnecting Serial...\n")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE
)

print("Listening CSI Data...\n")

buffer = []

# =========================================
# LIVE LOOP
# =========================================

while True:

    try:

        line = ser.readline().decode(
            errors="ignore"
        )

        # =====================================
        # ONLY CSI LINES
        # =====================================

        if "CSI_DATA" not in line:
            continue

        parsed = parse_csi_line(line)

        if parsed is None:
            continue

        # =====================================
        # CONVERT TO AMPLITUDE
        # =====================================

        amp = convert_amplitude(parsed)

        buffer.append(amp)

        print(
            "Packets Collected :",
            len(buffer)
        )

        # =====================================
        # START PREDICTION
        # =====================================

        if len(buffer) >= REQUIRED_PACKETS:

            df = pd.DataFrame(buffer)

            # =================================
            # SUBCARRIER SELECTION
            # =================================

            df = select_subcarriers(df)

            # =================================
            # MOVING AVERAGE
            # =================================

            df = moving_average(df)

            # =================================
            # BUTTER FILTER
            # =================================

            features = pd.DataFrame(
                butter_filter(
                    df.values
                )
            )

            # =================================
            # EXACT WINDOW
            # =================================

            features = features.iloc[
                :WINDOW_SIZE
            ]

            X = features.values

            # =================================
            # SHAPE CHECK
            # =================================

            print(
                "Feature Shape :",
                X.shape
            )

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

            # =================================
            # NORMALIZE
            # =================================

            X_scaled = scaler.transform(X)

            # =================================
            # RESHAPE
            # =================================

            X_final = X_scaled.reshape(
                1,
                WINDOW_SIZE,
                NUM_FEATURES
            )

            print(
                "Prediction Shape :",
                X_final.shape
            )

            # =================================
            # MODEL PREDICTION
            # =================================

            pred = model.predict(
                X_final,
                verbose=0
            )

            activity_id = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            activity = labels[
                activity_id
            ]

            # =================================
            # CONFIDENCE FILTER
            # =================================

            if confidence < CONFIDENCE_THRESHOLD:

                print(
                    "Low Confidence :",
                    round(confidence, 4)
                )

                buffer = buffer[STRIDE:]

                continue

            # =================================
            # PREDICTION SMOOTHING
            # =================================

            final_activity = smooth_prediction(
                activity
            )

            # =================================
            # PRINT RESULT
            # =================================

            print("\n========================")
            print(
                "Predicted Activity :",
                final_activity
            )
            print(
                "Confidence :",
                round(confidence, 4)
            )
            print("========================\n")

            # =================================
            # SEND TO SERVER
            # =================================

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
                    "Server Response :",
                    response.text
                )

            except Exception as e:

                print(
                    "API Failed :",
                    e
                )

            # =================================
            # SLIDING WINDOW
            # =================================

            buffer = buffer[STRIDE:]

    except KeyboardInterrupt:

        print("\nStopping Prediction...\n")

        ser.close()

        break

    except Exception as e:

        print(
            "Runtime Error :",
            e
        )