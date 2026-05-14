
# WIFI CSI HAR TRAINING PIPELINE
# ACTIVITIES: WALK / STATIC
# FINAL STABLE VERSION


import os
import re
import numpy as np
import pandas as pd
import pickle

from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    GRU,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import EarlyStopping


# CONFIG


DATASET_DIR = r"C:\esp32-csi-tool\dataset4"
MODEL_DIR = r"C:\esp32-csi-tool\model5"

WINDOW_SIZE = 300
STRIDE = 300
MOVING_AVG = 5

# FIXED CSI PACKET LENGTH
EXPECTED_LEN = 128

os.makedirs(MODEL_DIR, exist_ok=True)


# BUTTERWORTH FILTER


def butter_filter(data, cutoff=0.1, order=3):

    b, a = butter(order, cutoff)

    return filtfilt(b, a, data, axis=0)


# PARSE CSI


def parse_csi(row):

    try:

        row = row.strip()

        if row.startswith("["):
            row = row[1:]

        if row.endswith("]"):
            row = row[:-1]

        values = np.array(
            [float(x) for x in re.findall(r'-?\d+', row)],
            dtype=np.float32
        )

        return values

    except:

        return None


# LOAD CSI FILES


def load_csi_files():

    walk_files = []
    static_files = []

    walk_dir = os.path.join(DATASET_DIR, "walk")
    static_dir = os.path.join(DATASET_DIR, "static")

    
    # WALK
    

    for file in os.listdir(walk_dir):

        if file.endswith(".csv"):

            path = os.path.join(walk_dir, file)

            print("Loading WALK:", path)

            df = pd.read_csv(path)

            csi_rows = []

            for row in df["CSI_DATA"].dropna():

                parsed = parse_csi(str(row))

                # FIXED CSI SIZE
                if parsed is not None and len(parsed) == EXPECTED_LEN:

                    csi_rows.append(parsed)

            if len(csi_rows) > WINDOW_SIZE:

                walk_files.append(
                    pd.DataFrame(csi_rows)
                )

    
    # STATIC
    

    for file in os.listdir(static_dir):

        if file.endswith(".csv"):

            path = os.path.join(static_dir, file)

            print("Loading STATIC:", path)

            df = pd.read_csv(path)

            csi_rows = []

            for row in df["CSI_DATA"].dropna():

                parsed = parse_csi(str(row))

                # FIXED CSI SIZE
                if parsed is not None and len(parsed) == EXPECTED_LEN:

                    csi_rows.append(parsed)

            if len(csi_rows) > WINDOW_SIZE:

                static_files.append(
                    pd.DataFrame(csi_rows)
                )

    return walk_files, static_files


# CSI → AMP + PHASE


def convert_csi_to_amp_phase(df):

    amps = []
    phases = []

    for row in df.values:

        real = row[::2]
        imag = row[1::2]

        amp = np.sqrt(
            imag**2 + real**2
        )

        phase = np.unwrap(
            np.arctan2(imag, real)
        )

        amps.append(amp)
        phases.append(phase)

    return pd.DataFrame(amps), pd.DataFrame(phases)


# SUBCARRIER SELECTION


def select_subcarriers(df):

    part1 = df.iloc[:, 5:32]
    part2 = df.iloc[:, 33:60]

    return pd.concat(
        [part1, part2],
        axis=1
    )


# MOVING AVERAGE

def moving_average(df):

    return df.rolling(
        MOVING_AVG
    ).mean().dropna()


# WINDOW SEGMENTATION


def create_windows(df, label):

    X = []
    y = []

    for i in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STRIDE
    ):

        window = df.iloc[
            i:i + WINDOW_SIZE
        ].values

        if window.shape[0] == WINDOW_SIZE:

            X.append(window)
            y.append(label)

    return X, y


# LOAD FILES


print("\nLoading dataset...\n")

walk_files, static_files = load_csi_files()


# SPLIT FILES FIRST


walk_train, walk_test = train_test_split(
    walk_files,
    test_size=0.2,
    random_state=42
)

static_train, static_test = train_test_split(
    static_files,
    test_size=0.2,
    random_state=42
)


# DATA CONTAINERS


X_train = []
y_train = []

X_test = []
y_test = []


# PROCESS FILES


def process_files(files, label, X_store, y_store):

    for df in files:

        # AMP + PHASE
        amp, phase = convert_csi_to_amp_phase(df)

        # SUBCARRIER SELECTION
        amp = select_subcarriers(amp)
        phase = select_subcarriers(phase)

        # MOVING AVERAGE
        amp = moving_average(amp)
        phase = moving_average(phase)

        # STATIC REMOVAL
        amp = amp - amp.mean(axis=0)
        phase = phase - phase.mean(axis=0)

        # FEATURE CONCAT
        features = pd.concat(
            [amp, phase],
            axis=1
        )

        # BUTTERWORTH FILTER
        filtered = butter_filter(
            features.values
        )

        # SAFE FEATURES
        features = pd.DataFrame(
            np.nan_to_num(filtered)
        )

        # CREATE WINDOWS
        X_tmp, y_tmp = create_windows(
            features,
            label
        )

        X_store.extend(X_tmp)
        y_store.extend(y_tmp)


# PROCESS TRAIN DATA


process_files(
    walk_train,
    0,
    X_train,
    y_train
)

process_files(
    static_train,
    1,
    X_train,
    y_train
)


# PROCESS TEST DATA


process_files(
    walk_test,
    0,
    X_test,
    y_test
)

process_files(
    static_test,
    1,
    X_test,
    y_test
)


# NUMPY CONVERSION


X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print("\nTrain Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)


# SHUFFLE


X_train, y_train = shuffle(
    X_train,
    y_train,
    random_state=42
)

X_test, y_test = shuffle(
    X_test,
    y_test,
    random_state=42
)


# NORMALIZATION


scaler = StandardScaler()

samples_train = X_train.shape[0]
timesteps = X_train.shape[1]
features = X_train.shape[2]

# TRAIN
X_train_reshaped = X_train.reshape(
    -1,
    features
)

X_train_scaled = scaler.fit_transform(
    np.nan_to_num(X_train_reshaped)
)

# TEST+
X_test_reshaped = X_test.reshape(
    -1,
    features
)

X_test_scaled = scaler.transform(
    np.nan_to_num(X_test_reshaped)
)


# PCA


pca = PCA(
    n_components=0.95
)

# TRAIN PCA
X_train_pca = pca.fit_transform(
    X_train_scaled
)

# TEST PCA
X_test_pca = pca.transform(
    X_test_scaled
)

new_features = X_train_pca.shape[1]

# RESHAPE TRAIN
X_train = X_train_pca.reshape(
    samples_train,
    timesteps,
    new_features
)

# RESHAPE TEST
samples_test = X_test.shape[0]

X_test = X_test_pca.reshape(
    samples_test,
    timesteps,
    new_features
)


# SAVE PREPROCESSORS


pickle.dump(
    scaler,
    open(
        os.path.join(
            MODEL_DIR,
            "cnn_scaler2.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    pca,
    open(
        os.path.join(
            MODEL_DIR,
            "cnn_pca2.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    EXPECTED_LEN,
    open(
        os.path.join(
            MODEL_DIR,
            "csi_len.pkl"
        ),
        "wb"
    )
)


# CLASS WEIGHTS


weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(
    enumerate(weights)
)


# MODEL


print("\nTraining CNN-GRU...\n")

model = Sequential()


# CNN BLOCK 1


model.add(
    Conv1D(
        64,
        5,
        activation="relu",
        input_shape=X_train.shape[1:]
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling1D(2)
)


# CNN BLOCK 2


model.add(
    Conv1D(
        128,
        3,
        activation="relu"
    )
)

model.add(
    BatchNormalization()
)

model.add(
    MaxPooling1D(2)
)


# CNN BLOCK 3


model.add(
    Conv1D(
        256,
        3,
        activation="relu"
    )
)

model.add(
    MaxPooling1D(2)
)


# GRU


model.add(
    GRU(64)
)


# DENSE


model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(
    Dropout(0.5)
)


# OUTPUT


model.add(
    Dense(
        2,
        activation="softmax"
    )
)


# COMPILE


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# EARLY STOPPING


early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.summary()


# TRAIN


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early],
    shuffle=True
)


# EVALUATION


y_pred = np.argmax(
    model.predict(X_test),
    axis=1
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# SAVE MODEL


model.save(
    os.path.join(
        MODEL_DIR,
        "cnn_gru_model1.keras"
    )
)

labels = {
    0: "walk",
    1: "static"
}

pickle.dump(
    labels,
    open(
        os.path.join(
            MODEL_DIR,
            "labels2.pkl"
        ),
        "wb"
    )
)

print("\nTraining complete.")