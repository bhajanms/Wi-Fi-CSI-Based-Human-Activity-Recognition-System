# =========================================
# LIVE WIFI CSI HAR PREDICTION
# ACTIVITIES: WALK / STATIC
# FINAL FULLY CORRECT STABLE VERSION
# =========================================

import serial
import re
import time
import pickle
import requests

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# =========================================
# CONFIG
# =========================================

PORT = "COM10"
BAUD = 115200

MODEL_DIR = r"C:\esp32-csi-tool\model5"

# prediction interval (seconds)
PREDICT_INTERVAL = 5

# =========================================
# WINDOW SETTINGS
# =========================================

# SAME AS TRAINING
WINDOW_SIZE = 300

# SAME AS TRAINING
MOVING_AVG = 5

# rolling average removes first 4 rows
RAW_WINDOW_SIZE = WINDOW_SIZE + MOVING_AVG - 1

# =========================================
# OPTIONAL API
# =========================================

USE_API = True

API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# LOAD MODEL + PREPROCESSORS
# =========================================

print("\nLoading Model...\n")

model = load_model(
    MODEL_DIR + r"\cnn_gru_model1.keras"
)

scaler = pickle.load(
    open(
        MODEL_DIR + r"\cnn_scaler2.pkl",
        "rb"
    )
)

pca = pickle.load(
    open(
        MODEL_DIR + r"\cnn_pca2.pkl",
        "rb"
    )
)

EXPECTED_LEN = pickle.load(
    open(
        MODEL_DIR + r"\csi_len.pkl",
        "rb"
    )
)

labels = pickle.load(
    open(
        MODEL_DIR + r"\labels2.pkl",
        "rb"
    )
)

print("Model Loaded Successfully")
print("Expected CSI Length :", EXPECTED_LEN)
print("Labels :", labels)

# =========================================
# SERIAL CONNECTION
# =========================================

print("\nOpening Serial Port...\n")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("Listening on", PORT)

# =========================================
# BUTTERWORTH FILTER
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

def parse_csi(line):

    try:

        m = re.search(
            r"\[(.*)\]",
            line
        )

        if not m:
            return None

        row = m.group(1)

        values = np.array(
            [
                float(x)
                for x in re.findall(
                    r"-?\d+",
                    row
                )
            ],
            dtype=np.float32
        )

        return values

    except:

        return None

# =========================================
# CSI → AMP + PHASE
# =========================================

def convert_amp_phase(csi):

    real = csi[::2]
    imag = csi[1::2]

    amp = np.sqrt(
        imag**2 + real**2
    )

    phase = np.unwrap(
        np.arctan2(imag, real)
    )

    return amp, phase

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
# BUFFERS
# =========================================

amp_buffer = []
phase_buffer = []

last_prediction = time.time()

# =========================================
# LIVE LOOP
# =========================================

print("\nStarting Live Prediction...\n")

while True:

    try:

        # =====================================
        # READ SERIAL
        # =====================================

        line = ser.readline().decode(
            errors="ignore"
        )

        # only CSI packets
        if "CSI_DATA" not in line:
            continue

        parsed = parse_csi(line)

        if parsed is None:
            continue

        # =====================================
        # FIXED CSI LENGTH CHECK
        # =====================================

        if len(parsed) != EXPECTED_LEN:
            continue

        # =====================================
        # AMP + PHASE
        # =====================================

        amp, phase = convert_amp_phase(parsed)

        amp_buffer.append(amp)
        phase_buffer.append(phase)

        # =====================================
        # KEEP FIXED BUFFER
        # =====================================

        if len(amp_buffer) > RAW_WINDOW_SIZE:

            amp_buffer.pop(0)
            phase_buffer.pop(0)

        # =====================================
        # WAIT UNTIL BUFFER FILLS
        # =====================================

        if len(amp_buffer) < RAW_WINDOW_SIZE:
            continue

        # =====================================
        # PREDICTION INTERVAL
        # =====================================

        if time.time() - last_prediction >= PREDICT_INTERVAL:

            # =====================================
            # DATAFRAME
            # =====================================

            amp_df = pd.DataFrame(
                amp_buffer
            )

            phase_df = pd.DataFrame(
                phase_buffer
            )

            # =====================================
            # SUBCARRIER SELECTION
            # =====================================

            amp_df = select_subcarriers(
                amp_df
            )

            phase_df = select_subcarriers(
                phase_df
            )

            # =====================================
            # MOVING AVERAGE
            # =====================================

            amp_df = moving_average(
                amp_df
            )

            phase_df = moving_average(
                phase_df
            )

            # =====================================
            # EXACT WINDOW CHECK
            # =====================================

            if len(amp_df) != WINDOW_SIZE:
                continue

            # =====================================
            # STATIC REMOVAL
            # =====================================

            amp_df = amp_df - amp_df.mean(axis=0)

            phase_df = phase_df - phase_df.mean(axis=0)

            # =====================================
            # FEATURE CONCAT
            # =====================================

            features = pd.concat(
                [amp_df, phase_df],
                axis=1
            )

            # =====================================
            # BUTTER FILTER
            # =====================================

            filtered = butter_filter(
                features.values
            )

            # =====================================
            # SAFE VALUES
            # =====================================

            filtered = np.nan_to_num(
                filtered
            )

            # =====================================
            # SCALE
            # =====================================

            scaled = scaler.transform(
                filtered
            )

            # =====================================
            # PCA
            # =====================================

            pca_features = pca.transform(
                scaled
            )

            # =====================================
            # RESHAPE
            # =====================================

            X_live = np.expand_dims(
                pca_features,
                axis=0
            ).astype(np.float32)

            # =====================================
            # SAFE NAN CHECK
            # =====================================

            if np.isnan(X_live).any():
                continue

            # =====================================
            # MODEL PREDICTION
            # =====================================

            pred = model.predict(
                X_live,
                verbose=0
            )[0]

            class_idx = int(
                np.argmax(pred)
            )

            activity = labels[
                int(class_idx)
            ]

            confidence = float(
                pred[class_idx]
            )

            walk_prob = float(pred[0])

            static_prob = float(pred[1])

            # =====================================
            # PRINT RESULTS
            # =====================================

            print("\n================================")

            print(
                "Predicted Activity :",
                activity
            )

            print(
                "Confidence         :",
                round(confidence, 3)
            )

            print("\nProbabilities")

            print(
                "walk   :",
                round(walk_prob, 3)
            )

            print(
                "static :",
                round(static_prob, 3)
            )

            print("================================")

            # =====================================
            # OPTIONAL API CALL
            # =====================================

            if USE_API:

                try:

                    payload = {

                        "activity": activity,

                        "confidence": confidence,

                        "walk_probability": walk_prob,

                        "static_probability": static_prob
                    }

                    response = requests.post(
                        API_URL,
                        json=payload,
                        timeout=2
                    )

                    if response.ok:

                        print(
                            "API Status :",
                            response.status_code
                        )

                    else:

                        print(
                            "API Failed :",
                            response.status_code
                        )

                except Exception as api_error:

                    print(
                        "API Error :",
                        api_error
                    )

            # =====================================
            # UPDATE TIMER
            # =====================================

            last_prediction = time.time()

    except KeyboardInterrupt:

        print("\nStopped Live Prediction")

        break

    except Exception as e:

        print(
            "\nRuntime Error :",
            e
        )