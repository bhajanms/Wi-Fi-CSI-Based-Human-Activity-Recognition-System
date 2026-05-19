# =========================================
# WIFI CSI HAR LIVE PREDICTION (FINAL STABLE)
# =========================================

import re
import pickle
import serial
import numpy as np
import pandas as pd
import requests

from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# =========================================
# CONFIG
# =========================================

MODEL_DIR = r"C:\esp32-csi-tool\model3"

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

# Must be larger than model timestep
WINDOW_SIZE = 320

# Sliding window
STRIDE = 150

# Moving average smoothing
MOVING_AVG = 5

# Model expected sequence length
TARGET_LEN = 300

# Flask API
API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# LOAD MODEL + PREPROCESSORS
# =========================================

print("\nLoading model...")

model = load_model(
    f"{MODEL_DIR}/cnn_gru_modelattention.keras"
)

scaler = pickle.load(
    open(f"{MODEL_DIR}/cnn_scaler2.pkl", "rb")
)

pca = pickle.load(
    open(f"{MODEL_DIR}/cnn_pca2.pkl", "rb")
)

labels = pickle.load(
    open(f"{MODEL_DIR}/labels2.pkl", "rb")
)

print("Model loaded successfully")

print("\nModel Input Shape:")
print(model.input_shape)

# =========================================
# BUTTERWORTH FILTER
# =========================================

def butter_filter(data, cutoff=0.1, order=3):

    b, a = butter(order, cutoff)

    return filtfilt(b, a, data, axis=0)

# =========================================
# CSI PARSER
# =========================================

def parse_csi(line):

    try:

        match = re.search(r"\[(.*)\]", line)

        if match is None:
            return None

        csi_str = match.group(1)

        nums = re.findall(r"-?\d+", csi_str)

        csi = np.array(
            [int(x) for x in nums],
            dtype=np.float32
        )

        # Must contain I/Q pairs
        if len(csi) < 128:
            return None

        print("Parsed CSI length:", len(csi))

        return csi

    except Exception as e:

        print("CSI Parse Error:", e)

        return None

# =========================================
# AMP + PHASE EXTRACTION
# =========================================

def convert_amp_phase(arr):

    imag = arr[::2]
    real = arr[1::2]

    amp = np.sqrt(imag**2 + real**2)

    phase = np.unwrap(
        np.arctan2(imag, real)
    )

    return amp, phase

# =========================================
# SUBCARRIER SELECTION
# =========================================

def select_subcarriers(df):

    try:

        # Remove null/DC carriers
        part1 = df.iloc[:, 5:32]
        part2 = df.iloc[:, 33:60]

        final_df = pd.concat(
            [part1, part2],
            axis=1
        )

        return final_df

    except Exception as e:

        print("Subcarrier Selection Error:", e)

        return df

# =========================================
# MOVING AVERAGE
# =========================================

def moving_average(df):

    return df.rolling(
        MOVING_AVG
    ).mean().dropna()

# =========================================
# FIXED SEQUENCE LENGTH
# =========================================

def fix_sequence_length(data, target_len):

    current_len = data.shape[0]

    # PAD
    if current_len < target_len:

        pad_amount = target_len - current_len

        padding = np.zeros(
            (pad_amount, data.shape[1]),
            dtype=np.float32
        )

        data = np.vstack([data, padding])

        print(
            f"Padded: {current_len} -> {target_len}"
        )

    # TRIM
    elif current_len > target_len:

        data = data[:target_len]

        print(
            f"Trimmed: {current_len} -> {target_len}"
        )

    return data

# =========================================
# SERIAL CONNECTION
# =========================================

print("\nConnecting Serial...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE
)

print("Serial Connected")
print("\nListening for CSI packets...\n")

# =========================================
# BUFFER
# =========================================

buffer = []

# =========================================
# LIVE LOOP
# =========================================

while True:

    try:

        line = ser.readline().decode(
            errors="ignore"
        )

        line = line.strip()

        if "CSI_DATA" not in line:
            continue

        print("\nRAW:", line[:120])

        # =====================================
        # PARSE CSI
        # =====================================

        parsed = parse_csi(line)

        if parsed is None:
            continue

        # =====================================
        # AMP + PHASE
        # =====================================

        amp, phase = convert_amp_phase(parsed)

        combined = np.concatenate(
            [amp, phase]
        )

        buffer.append(combined)

        print("Buffer Size:", len(buffer))

        # =====================================
        # WINDOW READY
        # =====================================

        if len(buffer) >= WINDOW_SIZE:

            print("\n================================")
            print("WINDOW READY -> PREPROCESSING")
            print("================================")

            # =================================
            # DATAFRAME
            # =================================

            df = pd.DataFrame(buffer)

            total_cols = len(df.columns)

            amp_df = df.iloc[:, :total_cols // 2]
            phase_df = df.iloc[:, total_cols // 2:]

            # =================================
            # SUBCARRIER SELECTION
            # =================================

            amp_df = select_subcarriers(amp_df)
            phase_df = select_subcarriers(phase_df)

            # =================================
            # MOVING AVERAGE
            # =================================

            amp_df = moving_average(amp_df)
            phase_df = moving_average(phase_df)

            # =================================
            # COMBINE FEATURES
            # =================================

            features = pd.concat(
                [amp_df, phase_df],
                axis=1
            )

            print(
                "Feature Shape Before Filter:",
                features.shape
            )

            # =================================
            # BUTTER FILTER
            # =================================

            try:

                filtered = butter_filter(
                    features.values
                )

            except Exception as e:

                print("Filter Error:", e)

                buffer = buffer[STRIDE:]

                continue

            X = filtered

            print(
                "Feature Shape After Filter:",
                X.shape
            )

            # =================================
            # SCALER
            # =================================

            try:

                X_scaled = scaler.transform(X)

            except Exception as e:

                print("Scaler Error:", e)

                buffer = buffer[STRIDE:]

                continue

            # =================================
            # PCA
            # =================================

            try:

                X_pca = pca.transform(X_scaled)

            except Exception as e:

                print("PCA Error:", e)

                buffer = buffer[STRIDE:]

                continue

            print(
                "PCA Shape:",
                X_pca.shape
            )

            # =================================
            # FIX SEQUENCE LENGTH
            # =================================

            X_pca = fix_sequence_length(
                X_pca,
                TARGET_LEN
            )

            # =================================
            # FINAL MODEL INPUT
            # =================================

            X_final = X_pca.reshape(
                1,
                TARGET_LEN,
                X_pca.shape[1]
            )

            print(
                "Final Input Shape:",
                X_final.shape
            )

            # =================================
            # PREDICTION
            # =================================

            try:

                pred = model.predict(
                    X_final,
                    verbose=0
                )

            except Exception as e:

                print("Prediction Error:", e)

                buffer = buffer[STRIDE:]

                continue

            # =================================
            # RESULT
            # =================================

            activity_id = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            activity = labels[activity_id]

            print("\n================================")
            print("PREDICTED ACTIVITY:", activity)
            print("CONFIDENCE:", confidence)
            print("================================\n")

            # =================================
            # SEND TO API
            # =================================

            payload = {
                "activity": activity,
                "confidence": confidence
            }

            try:

                print("Sending To API...")

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=3
                )

                print(
                    "API Response:",
                    response.text
                )

            except Exception as e:

                print("API Error:", e)

            # =================================
            # SLIDING WINDOW
            # =================================

            buffer = buffer[STRIDE:]

            print(
                f"Sliding Window Applied -> Remaining Buffer: {len(buffer)}"
            )

    except KeyboardInterrupt:

        print("\nStopping Prediction...")

        ser.close()

        break

    except Exception as e:

        print("\nRuntime Error:", e)

        continue