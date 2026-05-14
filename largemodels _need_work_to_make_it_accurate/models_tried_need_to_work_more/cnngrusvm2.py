# =========================================
# FINAL WIFI CSI CNN-GRU TRAINING
# WALK / STATIC / FALL
# =========================================

import os
import re
import pickle
import numpy as np
import pandas as pd

from collections import Counter
from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, resample
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    Dense,
    GRU
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =========================================
# CONFIG
# =========================================

DATASET_DIR = r"C:\esp32-csi-tool\dataset3"

MODEL_DIR = r"C:\esp32-csi-tool\model3"

WINDOW_SIZE = 128
STRIDE = 64
MOVING_AVG = 5

NUM_FEATURES = 54

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================
# BUTTER FILTER
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
# PARSE RAW CSI
# =========================================

def parse_csi_line(line):

    try:

        match = re.search(
            r"\[(.*)\]",
            line
        )

        if not match:
            return None

        raw = match.group(1)

        values = [
            int(x)
            for x in raw.split()
        ]

        return np.array(
            values,
            dtype=np.float32
        )

    except:

        return None

# =========================================
# LOAD FILES
# =========================================

def load_class_files(class_name):

    data = []

    folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    print(f"\nLoading {class_name}\n")

    for file in os.listdir(folder):

        if not file.endswith(".csv"):
            continue

        path = os.path.join(
            folder,
            file
        )

        rows = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                if "CSI_DATA" not in line:
                    continue

                parsed = parse_csi_line(line)

                if parsed is not None:

                    rows.append(parsed)

        if len(rows) > 300:

            data.append(
                pd.DataFrame(rows)
            )

            print(
                file,
                "Packets :",
                len(rows)
            )

    return data

# =========================================
# AMPLITUDE ONLY
# =========================================

def convert_amplitude(df):

    amps = []

    for row in df.values:

        imag = row[::2]

        real = row[1::2]

        amp = np.sqrt(
            imag**2 + real**2
        )

        amps.append(amp)

    return pd.DataFrame(amps)

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
# WINDOW CREATION
# =========================================

def create_windows(df, label):

    X = []
    y = []

    for i in range(
        0,
        len(df) - WINDOW_SIZE,
        STRIDE
    ):

        window = df.iloc[
            i:i+WINDOW_SIZE
        ].values

        if window.shape == (
            WINDOW_SIZE,
            NUM_FEATURES
        ):

            X.append(window)

            y.append(label)

    return X, y

# =========================================
# PROCESS FILES
# =========================================

X = []
y = []

def process_files(files, label):

    global X, y

    for df in files:

        amp = convert_amplitude(df)

        amp = select_subcarriers(amp)

        amp = moving_average(amp)

        filtered = pd.DataFrame(
            butter_filter(
                amp.values
            )
        )

        X_tmp, y_tmp = create_windows(
            filtered,
            label
        )

        X.extend(X_tmp)

        y.extend(y_tmp)

# =========================================
# LOAD DATASET
# =========================================

walk_files = load_class_files("walk")

static_files = load_class_files("static")

fall_files = load_class_files("fall")

process_files(walk_files, 0)

process_files(static_files, 1)

process_files(fall_files, 2)

X = np.array(X)

y = np.array(y)

print("\nBefore Balance")

print(Counter(y))

# =========================================
# BALANCE DATASET
# =========================================

X_walk = X[y == 0]

X_static = X[y == 1]

X_fall = X[y == 2]

min_samples = min(
    len(X_walk),
    len(X_static),
    len(X_fall)
)

X_walk = resample(
    X_walk,
    replace=False,
    n_samples=min_samples,
    random_state=42
)

X_static = resample(
    X_static,
    replace=False,
    n_samples=min_samples,
    random_state=42
)

X_fall = resample(
    X_fall,
    replace=False,
    n_samples=min_samples,
    random_state=42
)

X = np.concatenate([
    X_walk,
    X_static,
    X_fall
])

y = np.concatenate([
    np.zeros(min_samples),
    np.ones(min_samples),
    np.full(min_samples, 2)
])

print("\nAfter Balance")

print(Counter(y))

# =========================================
# SHUFFLE
# =========================================

X, y = shuffle(
    X,
    y,
    random_state=42
)

# =========================================
# NORMALIZATION
# =========================================

scaler = StandardScaler()

X_reshaped = X.reshape(
    -1,
    X.shape[-1]
)

X_scaled = scaler.fit_transform(
    X_reshaped
)

X = X_scaled.reshape(
    X.shape[0],
    X.shape[1],
    X.shape[2]
)

# =========================================
# SAVE SCALER
# =========================================

pickle.dump(
    scaler,
    open(
        os.path.join(
            MODEL_DIR,
            "cnn_scaler.pkl"
        ),
        "wb"
    )
)

# =========================================
# SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain Shape :", X_train.shape)

print("Test Shape :", X_test.shape)

# =========================================
# CNN-GRU MODEL
# =========================================

model = Sequential()

model.add(
    Conv1D(
        64,
        5,
        activation="relu",
        input_shape=(
            WINDOW_SIZE,
            NUM_FEATURES
        )
    )
)

model.add(BatchNormalization())

model.add(MaxPooling1D(2))

model.add(Dropout(0.2))

model.add(
    Conv1D(
        128,
        3,
        activation="relu"
    )
)

model.add(BatchNormalization())

model.add(MaxPooling1D(2))

model.add(Dropout(0.2))

model.add(
    GRU(
        128,
        return_sequences=True
    )
)

model.add(Dropout(0.3))

model.add(
    GRU(64)
)

model.add(Dropout(0.3))

model.add(
    Dense(
        128,
        activation="relu"
    )
)

model.add(Dropout(0.5))

model.add(
    Dense(
        3,
        activation="softmax"
    )
)

# =========================================
# COMPILE
# =========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================
# CALLBACKS
# =========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# =========================================
# TRAIN
# =========================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(
        X_test,
        y_test
    ),
    callbacks=[
        early_stop,
        reduce_lr
    ]
)

# =========================================
# EVALUATION
# =========================================

pred = model.predict(X_test)

y_pred = np.argmax(
    pred,
    axis=1
)

print("\nConfusion Matrix")

print(confusion_matrix(
    y_test,
    y_pred
))

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred
))

# =========================================
# DEBUG
# =========================================

print("\nModel Input Shape")

print(model.input_shape)

# =========================================
# SAVE MODEL
# =========================================

model.save(
    os.path.join(
        MODEL_DIR,
        "cnn_gru_model.keras"
    )
)

labels = {
    0: "walk",
    1: "static",
    2: "fall"
}

pickle.dump(
    labels,
    open(
        os.path.join(
            MODEL_DIR,
            "labels.pkl"
        ),
        "wb"
    )
)

print("\nTraining Completed Successfully")