# =========================================
# WIFI CSI HAR LIVE PREDICTION
# FINAL ERROR-FREE VERSION
# MATCHED EXACTLY WITH TRAINING
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

# MUST MATCH TRAINING PIPELINE
WINDOW_SIZE = 304
STRIDE = 100
MOVING_AVG = 5

# AFTER MOVING AVERAGE:
# 304 - 5 + 1 = 300
FINAL_TIMESTEPS = 300

API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# LOAD MODEL + PREPROCESSORS
# =========================================

print("\nLoading Model...")

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

print("Model Loaded Successfully")

print("\nModel Input Shape:")
print(model.input_shape)

print("\nLabels:")
print(labels)

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
            [float(x) for x in nums],
            dtype=np.float32
        )

        # Skip invalid packets
        if len(csi) < 128:
            return None

        return csi

    except Exception as e:

        print("CSI Parse Error:", e)

        return None

# =========================================
# AMP + PHASE
# EXACTLY MATCHES TRAINING
# =========================================

def convert_amp_phase(arr):

    real = arr[::2]
    imag = arr[1::2]

    amp = np.sqrt(
        real**2 + imag**2
    )

    phase = np.unwrap(
        np.arctan2(imag, real)
    )

    return amp, phase

# =========================================
# SUBCARRIER SELECTION
# =========================================

def select_subcarriers(df):

    try:

        part1 = df.iloc[:, 5:32]
        part2 = df.iloc[:, 33:60]

        return pd.concat(
            [part1, part2],
            axis=1
        )

    except Exception as e:

        print("Subcarrier Error:", e)

        return df

# =========================================
# MOVING AVERAGE
# =========================================

def moving_average(df):

    return df.rolling(
        MOVING_AVG
    ).mean().dropna()

# =========================================
# SERIAL CONNECTION
# =========================================

print("\nConnecting Serial...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

print("Serial Connected")
print("\nListening For CSI Packets...\n")

# =========================================
# BUFFER
# =========================================

buffer = []

# =========================================
# LIVE LOOP
# =========================================

while True:

    try:

        # =====================================
        # READ SERIAL
        # =====================================

        line = ser.readline().decode(
            errors="ignore"
        ).strip()

        if "CSI_DATA" not in line:
            continue

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

        print(
            f"Buffer Size: {len(buffer)}",
            end="\r"
        )

        # =====================================
        # WINDOW READY
        # =====================================

        if len(buffer) >= WINDOW_SIZE:

            print("\n\n================================")
            print("WINDOW READY -> PREPROCESSING")
            print("================================")

            # =================================
            # CREATE DATAFRAME
            # =================================

            df = pd.DataFrame(buffer)

            total_cols = len(df.columns)

            amp_df = df.iloc[
                :,
                :total_cols // 2
            ]

            phase_df = df.iloc[
                :,
                total_cols // 2:
            ]

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
                "Feature Shape:",
                features.shape
            )

            # =================================
            # CHECK FINAL TIMESTEPS
            # =================================

            if features.shape[0] != FINAL_TIMESTEPS:

                print(
                    "Invalid timestep length:",
                    features.shape[0]
                )

                buffer = buffer[STRIDE:]

                continue

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

            print(
                "Filtered Shape:",
                filtered.shape
            )

            # =================================
            # SCALER
            # =================================

            try:

                X_scaled = scaler.transform(
                    filtered
                )

            except Exception as e:

                print("Scaler Error:", e)

                buffer = buffer[STRIDE:]

                continue

            # =================================
            # PCA
            # =================================

            try:

                X_pca = pca.transform(
                    X_scaled
                )

            except Exception as e:

                print("PCA Error:", e)

                buffer = buffer[STRIDE:]

                continue

            print(
                "PCA Shape:",
                X_pca.shape
            )

            # =================================
            # FINAL INPUT
            # =================================

            X_final = X_pca.reshape(
                1,
                FINAL_TIMESTEPS,
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

            print("\nRaw Prediction:", pred)

            activity_id = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            activity = labels[activity_id]

            # =================================
            # RESULT
            # =================================

            print("\n================================")
            print("PREDICTED ACTIVITY :", activity)
            print("CONFIDENCE          :", confidence)
            print("================================")

            # =================================
            # SEND TO API
            # =================================

            payload = {
                "activity": activity,
                "confidence": confidence
            }

            try:

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
                f"\nSliding Window Applied "
                f"-> Remaining Buffer: {len(buffer)}"
            )

    except KeyboardInterrupt:

        print("\n\nStopping Prediction...")

        ser.close()

        break

    except Exception as e:

        print("\nRuntime Error:", e)