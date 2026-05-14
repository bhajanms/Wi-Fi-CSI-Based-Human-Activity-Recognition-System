# =========================================
# REALTIME CSI PREDICTION (WALK / STAND)
# READ CSI FROM SERIAL PORT
# PREDICT EVERY 500 PACKETS
# =========================================

import serial
import numpy as np
import pandas as pd
import pickle
import requests
from scipy.signal import butter, filtfilt
import tensorflow as tf

# =========================================
# CONFIG
# =========================================

PORT = "COM10"
BAUD = 115200

PACKET_WINDOW = 500
WINDOW_SIZE = 200
MOVING_AVG = 5

MODEL_DIR = r"C:\esp32-csi-tool\model3"

API_URL = "http://172.20.10.8:5000/api/activity/update"

# =========================================
# LOAD TRAINED MODELS
# =========================================

print("Loading models...")

model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn_gru_model.h5")

scaler = pickle.load(open(f"{MODEL_DIR}/cnn_scaler.pkl", "rb"))

labels = pickle.load(open(f"{MODEL_DIR}/labels.pkl", "rb"))

print("Models loaded successfully")


# =========================================
# BUTTERWORTH FILTER
# =========================================

def butter_filter(data, cutoff=0.1, order=3):

    b, a = butter(order, cutoff)

    return filtfilt(b, a, data, axis=0)


# =========================================
# CSI PARSER
# =========================================

def parse_csi(raw):

    try:

        raw = raw.strip()

        if "[" in raw:
            raw = raw.split("[")[-1]

        if "]" in raw:
            raw = raw.split("]")[0]

        values = [int(x) for x in raw.split()]

        return np.array(values, dtype=float)

    except:

        return None


# =========================================
# AMP + PHASE EXTRACTION
# =========================================

def convert_csi_to_amp_phase(data):

    amps = []
    phases = []

    for row in data:

        imag = row[::2]
        real = row[1::2]

        amp = np.sqrt(imag**2 + real**2)

        phase = np.unwrap(np.arctan2(imag, real))

        amps.append(amp)
        phases.append(phase)

    return np.array(amps), np.array(phases)


# =========================================
# SUBCARRIER SELECTION
# =========================================

def select_subcarriers(data):

    part1 = data[:, 5:32]
    part2 = data[:, 33:60]

    return np.concatenate([part1, part2], axis=1)


# =========================================
# MOVING AVERAGE
# =========================================

def moving_average(data):

    df = pd.DataFrame(data)

    df = df.rolling(MOVING_AVG).mean().dropna()

    return df.values


# =========================================
# SERIAL CONNECTION
# =========================================

print("Connecting to serial port...")

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Listening on", PORT)

buffer = []


# =========================================
# MAIN LOOP
# =========================================

while True:

    try:

        line = ser.readline().decode(errors="ignore").strip()

        if "CSI_DATA" not in line:
            continue

        parts = line.split(",")

        csi_raw = parts[-1]

        parsed = parse_csi(csi_raw)

        if parsed is None:
            continue

        buffer.append(parsed)

        print("Packets:", len(buffer), end="\r")

        # =========================================
        # PREDICT EVERY 500 PACKETS
        # =========================================

        if len(buffer) >= PACKET_WINDOW:

            data = np.array(buffer[-PACKET_WINDOW:])

            amp, phase = convert_csi_to_amp_phase(data)

            amp = select_subcarriers(amp)
            phase = select_subcarriers(phase)

            features = np.concatenate([amp, phase], axis=1)

            features = moving_average(features)

            features = butter_filter(features)

            window = features[:WINDOW_SIZE]

            if window.shape[0] < WINDOW_SIZE:
                continue

            window = scaler.transform(window)

            window = window.reshape(1, window.shape[0], window.shape[1])

            pred = model.predict(window, verbose=0)

            cls = np.argmax(pred)

            activity = labels[cls]

            confidence = float(np.max(pred))

            print("\nPrediction:", activity)
            print("Confidence:", confidence)

            # =========================================
            # SEND DATA TO BACKEND API
            # =========================================

            try:

                payload = {
                    "activity": activity,
                    "confidence": confidence
                }

                response = requests.post(API_URL, json=payload, timeout=2)

                print("API Response:", response.text)

            except Exception as api_error:

                print("API Error:", api_error)

            # sliding window
            buffer = buffer[-WINDOW_SIZE:]

    except Exception as e:

        print("Error:", e)