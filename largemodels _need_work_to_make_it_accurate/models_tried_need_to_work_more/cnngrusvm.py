# =========================================
# WIFI CSI HAR TRAINING PIPELINE
# ACTIVITIES: WALK / STAND
# =========================================

import os
import numpy as np
import pandas as pd
import pickle
from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization

# =========================================
# CONFIG
# =========================================

DATASET_DIR = r"C:\esp32-csi-tool\dataset3"
MODEL_DIR = r"C:\esp32-csi-tool\model3"

WINDOW_SIZE = 200
STRIDE = 100
MOVING_AVG = 5

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================
# BUTTERWORTH FILTER
# =========================================

def butter_filter(data, cutoff=0.1, order=3):
    b, a = butter(order, cutoff)
    return filtfilt(b, a, data, axis=0)


# =========================================
# PARSE CSI STRING
# =========================================

def parse_csi(row):

    try:
        row = row.strip()

        if row.startswith("["):
            row = row[1:]

        if row.endswith("]"):
            row = row[:-1]

        return np.array([int(x) for x in row.split()], dtype=float)

    except:
        return None


# =========================================
# LOAD DATASET
# =========================================

def load_csi_files():

    walk_files = []
    stand_files = []

    walk_dir = os.path.join(DATASET_DIR, "walk")
    stand_dir = os.path.join(DATASET_DIR, "stand")

    # WALK DATA
    if os.path.exists(walk_dir):

        for file in os.listdir(walk_dir):

            if file.endswith(".csv"):

                path = os.path.join(walk_dir, file)

                print("Loading WALK:", path)

                df = pd.read_csv(path)

                csi_rows = []

                for row in df["CSI_DATA"].dropna():

                    parsed = parse_csi(str(row))

                    if parsed is not None:
                        csi_rows.append(parsed)

                if len(csi_rows) > 0:
                    walk_files.append(pd.DataFrame(csi_rows))

    # STAND DATA
    if os.path.exists(stand_dir):

        for file in os.listdir(stand_dir):

            if file.endswith(".csv"):

                path = os.path.join(stand_dir, file)

                print("Loading STAND:", path)

                df = pd.read_csv(path)

                csi_rows = []

                for row in df["CSI_DATA"].dropna():

                    parsed = parse_csi(str(row))

                    if parsed is not None:
                        csi_rows.append(parsed)

                if len(csi_rows) > 0:
                    stand_files.append(pd.DataFrame(csi_rows))

    return walk_files, stand_files


# =========================================
# CSI → AMPLITUDE + PHASE
# =========================================

def convert_csi_to_amp_phase(df):

    amplitudes = []
    phases = []

    for row in df.values:

        imag = row[::2]
        real = row[1::2]

        amp = np.sqrt(imag**2 + real**2)

        phase = np.unwrap(np.arctan2(imag, real))

        amplitudes.append(amp)
        phases.append(phase)

    return pd.DataFrame(amplitudes), pd.DataFrame(phases)


# =========================================
# SELECT SUBCARRIERS
# =========================================

def select_subcarriers(df):

    if df.shape[1] < 64:
        return df

    part1 = df.iloc[:,5:32]
    part2 = df.iloc[:,33:60]

    return pd.concat([part1, part2], axis=1)


# =========================================
# MOVING AVERAGE
# =========================================

def moving_average(df):

    return df.rolling(MOVING_AVG).mean().dropna()


# =========================================
# WINDOW SEGMENTATION
# =========================================

def create_windows(df, label):

    X = []
    y = []

    for i in range(0, len(df) - WINDOW_SIZE, STRIDE):

        window = df.iloc[i:i+WINDOW_SIZE].values

        if window.shape[0] == WINDOW_SIZE:

            X.append(window)
            y.append(label)

    return X, y


# =========================================
# DATASET BUILDING
# =========================================

print("\nLoading dataset...\n")

walk_files, stand_files = load_csi_files()

X = []
y = []

def process_files(files, label):

    global X, y

    for df in files:

        amp, phase = convert_csi_to_amp_phase(df)

        amp = select_subcarriers(amp)
        phase = select_subcarriers(phase)

        amp = moving_average(amp)
        phase = moving_average(phase)

        features = pd.concat([amp, phase], axis=1)

        features = pd.DataFrame(
            butter_filter(features.values)
        )

        X_tmp, y_tmp = create_windows(features, label)

        X.extend(X_tmp)
        y.extend(y_tmp)


process_files(walk_files, 0)
process_files(stand_files, 1)

X = np.array(X)
y = np.array(y)

print("\nDataset shape:", X.shape)

if len(X) == 0:
    print("\nERROR: No training data created.")
    exit()

X, y = shuffle(X, y, random_state=42)

# =========================================
# NORMALIZATION
# =========================================

scaler = StandardScaler()

X_reshaped = X.reshape(-1, X.shape[-1])

X_scaled = scaler.fit_transform(X_reshaped)

X = X_scaled.reshape(X.shape)

pickle.dump(
    scaler,
    open(os.path.join(MODEL_DIR, "cnn_scaler.pkl"), "wb")
)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================
# CNN-GRU MODEL
# =========================================

print("\nTraining CNN-GRU model...\n")

model = Sequential()

model.add(
    Conv1D(64, 3, activation="relu",
           input_shape=X_train.shape[1:])
)

model.add(BatchNormalization())
model.add(MaxPooling1D(2))

model.add(Conv1D(128, 3, activation="relu"))
model.add(MaxPooling1D(2))

model.add(GRU(128))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save(os.path.join(MODEL_DIR, "cnn_gru_model.h5"))

print("CNN-GRU model saved")

# =========================================
# SVM MODEL
# =========================================

print("\nTraining SVM model...\n")

X_flat = X.reshape(X.shape[0], -1)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_flat, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=40)),
    ("svc", SVC(kernel="rbf", class_weight="balanced"))
])

svm_pipeline.fit(X_train_svm, y_train_svm)

y_pred = svm_pipeline.predict(X_test_svm)

print("\nSVM Accuracy:", svm_pipeline.score(X_test_svm, y_test_svm))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_svm, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_svm, y_pred))

pickle.dump(
    svm_pipeline,
    open(os.path.join(MODEL_DIR, "svm_model.pkl"), "wb")
)

# =========================================
# SAVE LABELS
# =========================================

labels = {
    0: "walk",
    1: "stand"
}

pickle.dump(
    labels,
    open(os.path.join(MODEL_DIR, "labels.pkl"), "wb")
)

print("\nTraining complete.")