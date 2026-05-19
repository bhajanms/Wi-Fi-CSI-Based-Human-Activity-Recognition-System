import serial
import numpy as np
import re
from math import sqrt, atan2
import tensorflow as tf
import joblib
import requests

# =========================
# CONFIG
# =========================

PORT = "COM10"
BAUD = 115200

PACKET_WINDOW = 500
SEG_LEN = 50
STEP = 10

REMOVE_SUB = [2,3,4,5,32,59,60,61,62,63]

SERVER_URL = "http://192.168.1.22:5000/api/activity/latest"

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("Models/cnn_bilstm_csi_model.keras")
scaler = joblib.load("Models/scaler.pkl")
le = joblib.load("Models/labels.pkl")

print("Model Loaded")
print("Classes:", le.classes_)

# =========================
# CSI PARSER
# =========================

def parse_csi(line):

    nums = re.findall(r'-?\d+', line)

    if len(nums) < 20:
        return None

    nums = list(map(int, nums))

    real = nums[::2]
    imag = nums[1::2]

    amp = []
    phase = []

    for r, i in zip(real, imag):
        amp.append(sqrt(r*r + i*i))
        phase.append(atan2(i, r))

    amp = np.array(amp)
    phase = np.array(phase)

    feat = np.concatenate([amp, phase])

    try:
        feat = np.delete(feat, REMOVE_SUB)
    except:
        return None

    return feat

# =========================
# SERIAL CONNECTION
# =========================

ser = serial.Serial(PORT, BAUD, timeout=1)

print("Listening to ESP32 CSI stream...\n")

buffer = []
expected_len = None
packet_count = 0

# =========================
# MAIN LOOP
# =========================

while True:

    line = ser.readline().decode(errors="ignore")

    if "CSI_DATA" not in line:
        continue

    try:

        feat = parse_csi(line)

        if feat is None:
            continue

        if expected_len is None:
            expected_len = len(feat)

        if len(feat) != expected_len:
            continue

        buffer.append(feat)
        packet_count += 1

        print(f"Packet {packet_count}")

        # wait until 500 packets collected
        if packet_count < PACKET_WINDOW:
            continue

        print("\n500 packets collected → Running prediction\n")

        data = np.array(buffer)

        all_predictions = []

        # =========================
        # SLIDING WINDOW
        # =========================

        for i in range(0, PACKET_WINDOW - SEG_LEN, STEP):

            segment = data[i:i+SEG_LEN]

            if segment.shape[0] != SEG_LEN:
                continue

            samples = segment.shape[0]
            features = segment.shape[1]

            X = segment.reshape(-1, features)

            X = scaler.transform(X)

            X = X.reshape(1, samples, features)

            pred = model.predict(X, verbose=0)

            all_predictions.append(pred[0])

        if len(all_predictions) == 0:
            print("No valid predictions")
            buffer = []
            packet_count = 0
            continue

        # =========================
        # MEAN AVERAGE PREDICTION
        # =========================

        avg_prediction = np.mean(all_predictions, axis=0)

        cls = np.argmax(avg_prediction)

        final_activity = le.inverse_transform([cls])[0]

        confidence = float(np.max(avg_prediction))

        print("Average Probabilities:", avg_prediction)
        print("Final Activity:", final_activity)
        print("Confidence:", round(confidence,3))

        # =========================
        # SEND TO SERVER
        # =========================

        payload = {
            "activity": final_activity,
            "confidence": confidence
        }

        try:
            response = requests.post(SERVER_URL, json=payload)
            print("Server Response:", response.text)
        except Exception as e:
            print("Server error:", e)

        # =========================
        # RESET BUFFER
        # =========================

        buffer = []
        packet_count = 0

        print("\nWaiting for next 500 packets...\n")

    except Exception as e:

        print("Processing error:", e)