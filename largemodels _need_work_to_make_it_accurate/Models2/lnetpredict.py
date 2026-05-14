import serial
import re
import numpy as np
from math import sqrt
from collections import Counter

from scipy.signal import savgol_filter
from tensorflow import keras
import joblib

# =========================
# CONFIG
# =========================

PORT = "COM10"
BAUD = 115200

TOTAL_PACKETS = 700
SEG_LEN = 200
STEP = 100

# =========================
# LOAD MODEL + PREPROCESS
# =========================

print("Loading model...")

model = keras.models.load_model("lnet.keras")
labels = np.load("cnnlnet_labels.npy")

pca = joblib.load("pca_model.pkl")

mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

print("Model loaded")

# =========================
# SERIAL
# =========================

ser = serial.Serial(PORT, BAUD, timeout=1)

packets = []

print("Listening for CSI packets...")

# =========================
# CSI PARSER
# =========================

def parse_csi(line):

    m = re.search(r"\[(.*?)\]", line)
    if not m:
        return None

    raw = [int(x) for x in re.split(r"[,\s]+", m.group(1)) if x]

    real = raw[::2]
    imag = raw[1::2]

    amp = []

    for r,i in zip(real,imag):
        amp.append(sqrt(r*r + i*i))

    amp = np.array(amp)

    # remove null carriers
    amp = amp[6:58]

    return amp

# =========================
# MAIN LOOP
# =========================

while True:

    line = ser.readline().decode(errors="ignore")

    if "CSI_DATA" not in line:
        continue

    amp = parse_csi(line)

    if amp is None:
        continue

    packets.append(amp)

    print("Packets:",len(packets),"/",TOTAL_PACKETS)

    if len(packets) < TOTAL_PACKETS:
        continue

    print("\nProcessing activity...\n")

    data = np.array(packets)

    # smoothing
    data = savgol_filter(data,7,2,axis=0)

    # PCA
    data = pca.transform(data)

    predictions = []

    for i in range(0,TOTAL_PACKETS-SEG_LEN,STEP):

        segment = data[i:i+SEG_LEN]

        segment = (segment-mean)/(std+1e-6)

        segment = segment.reshape(1,SEG_LEN,segment.shape[1])

        pred = model.predict(segment,verbose=0)

        activity = labels[np.argmax(pred)]

        predictions.append(activity)

    final_activity = Counter(predictions).most_common(1)[0][0]

    print("Final Activity:",final_activity,"\n")

    packets = []