# =========================================
# IMPROVED WIFI CSI HAR TRAINING PIPELINE
# BETTER WALK / STATIC / FALL DETECTION
# =========================================

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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GRU,
    Dense,
    Dropout,
    BatchNormalization,
    Attention,
    GlobalAveragePooling1D,
    Bidirectional
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =========================================
# CONFIG
# =========================================

DATASET_DIR = r"C:\esp32-csi-tool\dataset3"
MODEL_DIR = r"C:\esp32-csi-tool\model4"

os.makedirs(MODEL_DIR, exist_ok=True)

# SHORTER WINDOWS = BETTER FALL DETECTION
WINDOW_SIZE = 128
STRIDE = 64

MOVING_AVG = 5

# =========================================
# FILTER
# =========================================

def butter_filter(data, cutoff=0.1, order=3):

    b, a = butter(order, cutoff)

    return filtfilt(b, a, data, axis=0)

# =========================================
# CSI PARSER
# =========================================

def parse_csi(row):

    try:

        row = row.strip()

        if row.startswith("["):
            row = row[1:]

        if row.endswith("]"):
            row = row[:-1]

        nums = re.findall(r"-?\d+", row)

        return np.array(
            [float(x) for x in nums],
            dtype=np.float32
        )

    except:

        return None

# =========================================
# LOAD FILES
# =========================================

def load_activity(folder):

    activity_files = []

    folder_path = os.path.join(
        DATASET_DIR,
        folder
    )

    for file in os.listdir(folder_path):

        if file.endswith(".csv"):

            path = os.path.join(
                folder_path,
                file
            )

            print(f"Loading {folder}:", file)

            df = pd.read_csv(path)

            rows = []

            for row in df["CSI_DATA"].dropna():

                parsed = parse_csi(str(row))

                if parsed is not None:

                    rows.append(parsed)

            if len(rows) > 0:

                activity_files.append(
                    pd.DataFrame(rows)
                )

    return activity_files

# =========================================
# AMP + PHASE
# =========================================

def convert_amp_phase(df):

    amps = []
    phases = []

    for row in df.values:

        real = row[::2]
        imag = row[1::2]

        amp = np.sqrt(
            real**2 + imag**2
        )

        phase = np.unwrap(
            np.arctan2(imag, real)
        )

        amps.append(amp)
        phases.append(phase)

    return (
        pd.DataFrame(amps),
        pd.DataFrame(phases)
    )

# =========================================
# SUBCARRIERS
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

        if window.shape[0] == WINDOW_SIZE:

            X.append(window)
            y.append(label)

    return X, y

# =========================================
# BUILD DATASET
# =========================================

print("\nLoading Dataset...\n")

walk_files = load_activity("walk")
static_files = load_activity("static")
fall_files = load_activity("fall")

X = []
y = []

def process_files(files, label):

    global X, y

    for df in files:

        amp, phase = convert_amp_phase(df)

        amp = select_subcarriers(amp)
        phase = select_subcarriers(phase)

        amp = moving_average(amp)
        phase = moving_average(phase)

        features = pd.concat(
            [amp, phase],
            axis=1
        )

        # FILTER
        features = pd.DataFrame(
            butter_filter(features.values)
        )

        X_tmp, y_tmp = create_windows(
            features,
            label
        )

        X.extend(X_tmp)
        y.extend(y_tmp)

# LABELS
# 0 = WALK
# 1 = STATIC
# 2 = FALL

process_files(walk_files, 0)
process_files(static_files, 1)
process_files(fall_files, 2)

X = np.array(X)
y = np.array(y)

print("\nDataset Shape:", X.shape)

# =========================================
# SHUFFLE
# =========================================

X, y = shuffle(
    X,
    y,
    random_state=42
)

# =========================================
# CLASS DISTRIBUTION
# =========================================

print("\nClass Distribution:")

print(pd.Series(y).value_counts())

# =========================================
# NORMALIZATION
# =========================================

samples = X.shape[0]
timesteps = X.shape[1]
features = X.shape[2]

scaler = StandardScaler()

X_reshaped = X.reshape(
    -1,
    features
)

X_scaled = scaler.fit_transform(
    X_reshaped
)

# =========================================
# PCA
# =========================================

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(
    X_scaled
)

new_features = X_pca.shape[1]

X = X_pca.reshape(
    samples,
    timesteps,
    new_features
)

print("\nPCA Features:", new_features)

# SAVE PREPROCESSORS

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

pickle.dump(
    pca,
    open(
        os.path.join(
            MODEL_DIR,
            "cnn_pca.pkl"
        ),
        "wb"
    )
)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================
# CLASS WEIGHTS
# =========================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights = dict(
    enumerate(weights)
)

print("\nClass Weights:")
print(class_weights)

# =========================================
# MODEL
# =========================================

print("\nTraining Model...\n")

inputs = Input(
    shape=X_train.shape[1:]
)

# CNN BLOCK 1
x = Conv1D(
    64,
    5,
    activation="relu",
    padding="same"
)(inputs)

x = BatchNormalization()(x)

x = MaxPooling1D(2)(x)

# CNN BLOCK 2
x = Conv1D(
    128,
    3,
    activation="relu",
    padding="same"
)(x)

x = BatchNormalization()(x)

x = MaxPooling1D(2)(x)

# BIDIRECTIONAL GRU
x = Bidirectional(
    GRU(
        64,
        return_sequences=True
    )
)(x)

# ATTENTION
attention = Attention()([x, x])

# GLOBAL POOLING
x = GlobalAveragePooling1D()(attention)

# DENSE
x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(0.4)(x)

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(0.3)(x)

outputs = Dense(
    3,
    activation="softmax"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================================
# CALLBACKS
# =========================================

early = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3
)

# =========================================
# TRAIN
# =========================================

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early, reduce_lr]
)

# =========================================
# EVALUATION
# =========================================

pred = model.predict(X_test)

y_pred = np.argmax(
    pred,
    axis=1
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================
# SAVE MODEL
# =========================================

model.save(
    os.path.join(
        MODEL_DIR,
        "cnn_gru_attention.keras"
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

print("\nTraining Complete.")