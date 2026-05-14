# =========================
# IMPORTS
# =========================

import numpy as np
import pandas as pd
import os
import re
from math import sqrt, atan2
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG (FASTER SETTINGS)
# =========================

DATASET_PATH = r"C:\esp32-csi-tool\datasets"

SEG_LEN = 120          # reduced from 150
STEP = 100             # less overlap = fewer samples
DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

EPOCHS = 25            # reduced epochs
BATCH_SIZE = 32        # bigger batch = faster

print("Starting Training...\n")

# =========================
# LOAD DATA
# =========================

features = []
labels = []

for activity in os.listdir(DATASET_PATH):

    act_path = os.path.join(DATASET_PATH, activity)
    if not os.path.isdir(act_path):
        continue

    for file in os.listdir(act_path):

        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(act_path, file)
        df = pd.read_csv(full_path)

        if "CSI_DATA" not in df.columns:
            continue

        amp_rows = []
        phase_rows = []

        for val in df["CSI_DATA"]:

            m = re.search(r"\[(.*)\]", str(val))
            if not m:
                continue

            raw = [int(x) for x in re.split(r"[,\s]+", m.group(1)) if x]

            imag = raw[::2]
            real = raw[1::2]

            amp = [sqrt(i*i + r*r) for i,r in zip(imag,real)]
            phase = [atan2(i,r) for i,r in zip(imag,real)]

            amp_rows.append(amp)
            phase_rows.append(phase)

        if len(amp_rows) < SEG_LEN:
            continue

        A = pd.DataFrame(amp_rows)
        P = pd.DataFrame(phase_rows)

        # smoothing
        for c in A.columns:
            if len(A[c]) >= 11:
                A[c] = savgol_filter(A[c],11,3)
                P[c] = savgol_filter(P[c],11,3)

        # 🔥 remove static background
        A = A - A.mean(axis=0)
        P = P - P.mean(axis=0)

        # drop unused carriers
        valid = [c for c in DROP_COLS if c < A.shape[1]]
        A.drop(A.columns[valid], axis=1, inplace=True)
        P.drop(P.columns[valid], axis=1, inplace=True)

        feat = np.stack([A.values, P.values], axis=-1)

        for start in range(0, len(feat)-SEG_LEN+1, STEP):

            segment = feat[start:start+SEG_LEN]

            # 🔥 per-segment normalization
            segment = (segment - segment.mean()) / (segment.std()+1e-6)

            features.append(segment)
            labels.append(activity)

print("Total segments:", len(features))

X = np.array(features).astype("float32")
y = np.array(labels)

# =========================
# ENCODE LABELS
# =========================

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.3, stratify=y_enc, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# =========================
# MODEL (FASTER CNN + SMALL LSTM)
# =========================

model = keras.Sequential([

    keras.layers.Conv2D(32,3,padding="same",activation="relu",
                        input_shape=(SEG_LEN, X.shape[2],2)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64,3,padding="same",activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Reshape((SEG_LEN//4, -1)),

    keras.layers.LSTM(32),     # smaller LSTM = faster

    keras.layers.Dense(32,activation="relu"),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(len(le.classes_),activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN
# =========================

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test,y_test),
    shuffle=True
)

# =========================
# SAVE MODEL
# =========================

model.save("fast_csi_model.keras")
np.save("fast_class_names.npy", le.classes_)

loss, acc = model.evaluate(X_test,y_test)
print("\nFinal Test Accuracy:", acc)