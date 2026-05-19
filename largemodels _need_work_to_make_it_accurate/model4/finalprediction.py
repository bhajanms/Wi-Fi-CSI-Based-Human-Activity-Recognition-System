# =========================================
# WIFI CSI HAR LIVE PREDICTION
# FINAL HYBRID VERSION
# WALK / STATIC / FALL
# =========================================

import re
import pickle
import serial
import numpy as np
import pandas as pd
import requests

from collections import Counter
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model

# =========================================
# CONFIG
# =========================================

MODEL_DIR = r"C:\esp32-csi-tool\model4"

SERIAL_PORT = "COM10"
BAUD_RATE = 115200

# MUST MATCH TRAINING
WINDOW_SIZE = 132
STRIDE = 64
MOVING_AVG = 5

# 132 - 5 + 1 = 128
FINAL_TIMESTEPS = 128

API_URL = "http://172.20.10.8:5000/api/activity/update"

# SMOOTHING
SMOOTHING_WINDOW = 3

# =========================================
# LOAD MODEL
# =========================================

print("\nLoading Model...")

model = load_model(
    f"{MODEL_DIR}/cnn_gru_attention.keras"
)

scaler = pickle.load(
    open(f"{MODEL_DIR}/cnn_scaler.pkl", "rb")
)

pca = pickle.load(
    open(f"{MODEL_DIR}/cnn_pca.pkl", "rb")
)

labels = pickle.load(
    open(f"{MODEL_DIR}/labels.pkl", "rb")
)

print("Model Loaded Successfully")

print("\nModel Input Shape:")
print(model.input_shape)

print("\nLabels:")
print(labels)

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
# CSI PARSER
# =========================================

def parse_csi(line):

    try:

        match = re.search(
            r"\[(.*)\]",
            line
        )

        if match is None:
            return None

        csi_str = match.group(1)

        nums = re.findall(
            r"-?\d+",
            csi_str
        )

        csi = np.array(
            [float(x) for x in nums],
            dtype=np.float32
        )

        if len(csi) < 128:
            return None

        return csi

    except Exception as e:

        print("CSI Parse Error:", e)

        return None

# =========================================
# AMP + PHASE
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
# BUFFERS
# =========================================

buffer = []

prediction_history = []

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
            # DATAFRAME
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
            # SUBCARRIERS
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
            # TIMESTEP CHECK
            # =================================

            if features.shape[0] != FINAL_TIMESTEPS:

                print(
                    "Invalid Timesteps:",
                    features.shape[0]
                )

                buffer = buffer[STRIDE:]

                continue

            # =================================
            # FILTER
            # =================================

            try:

                filtered = butter_filter(
                    features.values
                )

            except Exception as e:

                print("Filter Error:", e)

                buffer = buffer[STRIDE:]

                continue

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
            # MODEL PREDICTION
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
            # RAW MODEL PROBABILITIES
            # =================================

            walk_prob = float(pred[0][0])
            static_prob = float(pred[0][1])
            fall_prob = float(pred[0][2])

            print("\nModel Probabilities:")
            print(f"Walk   : {walk_prob:.4f}")
            print(f"Static : {static_prob:.4f}")
            print(f"Fall   : {fall_prob:.4f}")

            # =================================
            # CSI MOTION FEATURES
            # =================================

            motion_std = np.std(filtered)

            motion_mean = np.mean(
                np.abs(filtered)
            )

            motion_energy = np.mean(
                filtered ** 2
            )

            print("\nCSI Motion Features:")
            print(f"STD    : {motion_std:.4f}")
            print(f"Mean   : {motion_mean:.4f}")
            print(f"Energy : {motion_energy:.4f}")

            # =================================
            # HYBRID ACTIVITY DETECTION
            # =================================

            # STATIC
            if motion_std < 0.80 and motion_energy < 2.0:

                activity = "static"
                confidence = 0.90

            # FALL
            elif motion_std > 4.5 and fall_prob > 0.30:

                activity = "fall"
                confidence = max(
                    fall_prob,
                    0.85
                )

            # WALK
            else:

                activity = "walk"
                confidence = max(
                    walk_prob,
                    0.80
                )

            # =================================
            # SMOOTHING
            # =================================

            prediction_history.append(activity)

            if len(prediction_history) > SMOOTHING_WINDOW:

                prediction_history.pop(0)

            final_activity = Counter(
                prediction_history
            ).most_common(1)[0][0]

            # =================================
            # RESULT
            # =================================

            print("\n================================")
            print("FINAL ACTIVITY :", final_activity)
            print("CONFIDENCE     :", confidence)
            print("================================")

            # =================================
            # API
            # =================================

            payload = {
                "activity": final_activity,
                "confidence": float(confidence)
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

        print("\nStopping Prediction...")

        ser.close()

        break

    except Exception as e:

        print("\nRuntime Error:", e)