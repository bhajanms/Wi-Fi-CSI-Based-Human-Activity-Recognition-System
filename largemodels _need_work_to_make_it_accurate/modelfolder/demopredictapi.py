import numpy as np
import os
import re
from math import sqrt, atan2

from scipy.signal import butter, filtfilt

import tensorflow as tf
import joblib
import requests


# ===============================
# CONFIG
# ===============================

MODEL_PATH = "cnn_bilstm_csi_model.keras"
SCALER_PATH = "scaler.pkl"
LABEL_PATH = "labels.pkl"

DATA_FOLDER = r"C:\esp32-csi-tool\demodataset"

SEG_LEN = 50
STEP = 10

REMOVE_SUB = [2,3,4,5,32,59,60,61,62,63]

API_URL = "http://172.20.10.8/api/activity/update"


# ===============================
# LOAD MODEL
# ===============================

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_PATH)

print("Model Loaded")
print("Classes:", le.classes_)


# ===============================
# LOWPASS FILTER
# ===============================

def butter_lowpass(data, cutoff=0.1, fs=1.0, order=3):

    b, a = butter(order, cutoff/(0.5*fs), btype='low')

    return filtfilt(b, a, data)


# ===============================
# PARSE CSI
# ===============================

def parse_csi(line):

    nums = re.findall(r'-?\d+', line)
    nums = list(map(int, nums))

    real = nums[::2]
    imag = nums[1::2]

    amp = []
    phase = []

    for r,i in zip(real,imag):
        amp.append(sqrt(r*r + i*i))
        phase.append(atan2(i,r))

    amp = np.array(amp)
    phase = np.array(phase)

    return np.concatenate([amp,phase])


# ===============================
# LOAD FILE
# ===============================

def load_file(path):

    rows = []
    expected_len = None

    with open(path,'r',errors="ignore") as f:

        for line in f:

            if "CSI_DATA" not in line:
                continue

            try:

                feat = parse_csi(line)

                feat = np.delete(feat, REMOVE_SUB)

                if expected_len is None:
                    expected_len = len(feat)

                if len(feat) != expected_len:
                    continue

                rows.append(feat)

            except:
                continue

    if len(rows) == 0:
        return np.empty((0,0))

    data = np.array(rows)

    for i in range(data.shape[1]):
        data[:,i] = butter_lowpass(data[:,i])

    return data


# ===============================
# SEGMENT
# ===============================

def segment(data):

    segments = []

    for i in range(0, len(data) - SEG_LEN + 1, STEP):

        seg = data[i:i+SEG_LEN]
        segments.append(seg)

    return segments


# ===============================
# PREDICT FILE
# ===============================

def predict_file(file_path):

    data = load_file(file_path)

    if data.shape[0] < SEG_LEN:
        return None, 0

    segs = segment(data)
    segs = np.array(segs)

    samples = segs.shape[0]
    timesteps = segs.shape[1]
    features = segs.shape[2]

    # normalize
    X = segs.reshape(-1,features)
    X = scaler.transform(X)
    X = X.reshape(samples,timesteps,features)

    preds = model.predict(X, verbose=0)

    labels = np.argmax(preds, axis=1)

    counts = np.bincount(labels)

    final_label = np.argmax(counts)

    confidence = np.max(counts) / len(labels)

    activity = le.inverse_transform([final_label])[0]

    return activity, confidence


# ===============================
# SEND TO SERVER
# ===============================

def send_activity(activity, confidence):

    try:

        data = {
            "activity": activity,
            "confidence": float(confidence)
        }

        response = requests.post(API_URL, json=data)

        print("Server Response:", response.text)

    except Exception as e:

        print("Server connection error:", e)


# ===============================
# PREDICT FOLDER
# ===============================

for folder in os.listdir(DATA_FOLDER):

    folder_path = os.path.join(DATA_FOLDER, folder)

    if not os.path.isdir(folder_path):
        continue

    print("\n========", folder, "========")

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        activity, confidence = predict_file(file_path)

        if activity is None:
            continue

        print(file, "->", activity, "(", round(confidence,2), ")")

        # send to API
        send_activity(activity, confidence)