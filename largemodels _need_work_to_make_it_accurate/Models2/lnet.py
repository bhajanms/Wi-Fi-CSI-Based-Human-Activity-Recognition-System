import os
import re
import numpy as np
import pandas as pd
from math import sqrt

from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers

import joblib

# =========================
# CONFIG
# =========================

DATASET_PATH = r"C:\esp32-csi-tool\datasets1"

SEG_LEN = 200
STEP = 100

EPOCHS = 50
BATCH = 16

MODEL_NAME = "lnet.keras"

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
# LOAD DATASET
# =========================

features = []
labels = []

print("Reading dataset...")

for activity in os.listdir(DATASET_PATH):

    act_path = os.path.join(DATASET_PATH,activity)

    if not os.path.isdir(act_path):
        continue

    print("Processing:",activity)

    for file in os.listdir(act_path):

        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(act_path,file))

        if "CSI_DATA" not in df.columns:
            continue

        rows = []

        for val in df["CSI_DATA"]:

            amp = parse_csi(val)

            if amp is not None:
                rows.append(amp)

        if len(rows) < SEG_LEN:
            continue

        data = np.array(rows)

        # smoothing
        data = savgol_filter(data,7,2,axis=0)

        # segmentation
        for i in range(0,len(data)-SEG_LEN,STEP):

            seg = data[i:i+SEG_LEN]

            features.append(seg)
            labels.append(activity)


X = np.array(features)
y = np.array(labels)

print("Dataset shape:",X.shape)

# =========================
# PCA
# =========================

print("Applying PCA")

X_r = X.reshape(-1,X.shape[2])

pca = PCA(n_components=20)

X_pca = pca.fit_transform(X_r)

joblib.dump(pca,"pca_model.pkl")

X = X_pca.reshape(X.shape[0],SEG_LEN,20)

# =========================
# NORMALIZATION
# =========================

mean = X.mean()
std = X.std()

np.save("norm_mean.npy",mean)
np.save("norm_std.npy",std)

X = (X-mean)/(std+1e-6)

# =========================
# LABEL ENCODING
# =========================

le = LabelEncoder()

y_enc = le.fit_transform(y)

np.save("cnnlnet_labels.npy",le.classes_)

print("Classes:",le.classes_)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train,X_test,y_train,y_test = train_test_split(
    X,y_enc,test_size=0.2,stratify=y_enc,random_state=42
)

# =========================
# CLASS WEIGHTS
# =========================

weights = compute_class_weight(
    "balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

# =========================
# CNN + BiGRU MODEL
# =========================

inputs = keras.Input(shape=(SEG_LEN,X.shape[2]))

x = layers.Conv1D(64,5,padding="same",activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Conv1D(128,5,padding="same",activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2)(x)

x = layers.Bidirectional(layers.GRU(128,return_sequences=True))(x)
x = layers.Bidirectional(layers.GRU(64))(x)

x = layers.Dense(128,activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(len(le.classes_),activation="softmax")(x)

model = keras.Model(inputs,outputs)

model.compile(
    optimizer=keras.optimizers.Adam(0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_split=0.2,
    class_weight=class_weights
)

model.save(MODEL_NAME)

print("Training complete")