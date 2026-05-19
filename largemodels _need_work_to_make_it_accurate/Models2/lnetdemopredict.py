import numpy as np
import pandas as pd
import re
from math import sqrt
from scipy.signal import savgol_filter
import joblib
from tensorflow import keras

# =========================
# CONFIG
# =========================

CSI_FILE = r"C:\esp32-csi-tool\datasets1\walk\walk3.csv"

SEG_LEN = 200
STEP = 100

# =========================
# LOAD MODEL + FILES
# =========================

model = keras.models.load_model("lnet.keras")

pca = joblib.load("pca_model.pkl")

labels = np.load("cnnlnet_labels.npy",allow_pickle=True)

mean = np.load("norm_mean.npy")
std = np.load("norm_std.npy")

print("Classes:",labels)


# =========================
# CSI PARSER
# =========================

def parse_csi(val):

    m = re.search(r"\[(.*?)\]", str(val))
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
# LOAD CSI FILE
# =========================

df = pd.read_csv(CSI_FILE)

rows = []

for val in df["CSI_DATA"]:

    amp = parse_csi(val)

    if amp is not None:
        rows.append(amp)

data = np.array(rows)

print("Packets:",len(data))


# =========================
# SMOOTHING
# =========================

data = savgol_filter(data,7,2,axis=0)


# =========================
# SEGMENTATION
# =========================

segments = []

for i in range(0,len(data)-SEG_LEN,STEP):

    seg = data[i:i+SEG_LEN]
    segments.append(seg)

segments = np.array(segments)

print("Segments:",len(segments))


# =========================
# PCA TRANSFORM
# =========================

X = segments.reshape(-1,segments.shape[2])

X = pca.transform(X)

X = X.reshape(segments.shape[0],SEG_LEN,20)


# =========================
# NORMALIZATION
# =========================

X = (X-mean)/(std+1e-6)


# =========================
# PREDICTION
# =========================

pred = model.predict(X)

mean_prob = pred.mean(axis=0)

idx = np.argmax(mean_prob)

activity = labels[idx]

confidence = mean_prob[idx]

print("\nPrediction:",activity)
print("Confidence:",round(float(confidence),3))