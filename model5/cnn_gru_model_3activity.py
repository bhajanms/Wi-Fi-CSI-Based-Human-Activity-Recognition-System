# =========================================================
# ADVANCED WIFI CSI HAR TRAINING PIPELINE
# ACTIVITIES:
# WALK / STATIC / FALL
#
# FINAL ENHANCED VERSION
# =========================================================

import os
import re
import pickle

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

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

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# =========================================================
# CONFIG
# =========================================================

CSI_DATASET_PATH = r"C:\esp32-csi-tool\dataset5"

CSI_MODEL_SAVE_PATH = r"C:\esp32-csi-tool\model5"

os.makedirs(
    CSI_MODEL_SAVE_PATH,
    exist_ok=True
)

# =========================================================
# WINDOW CONFIG
# =========================================================

CSI_WINDOW_LENGTH = 150

CSI_WINDOW_STRIDE = 50

CSI_MOVING_AVERAGE = 5

CSI_EXPECTED_PACKET_LENGTH = 128

# =========================================================
# PCA CONFIG
# =========================================================

CSI_PCA_COMPONENTS = 15

# =========================================================
# FLOAT SAFETY
# =========================================================

np.seterr(all='ignore')

# =========================================================
# FILTER
# =========================================================

def apply_butterworth_filter(
    data,
    cutoff=0.1,
    order=3
):

    b, a = butter(order, cutoff)

    return filtfilt(
        b,
        a,
        data,
        axis=0
    )

# =========================================================
# PARSE CSI
# =========================================================

def parse_csi_packet(row):

    try:

        row = row.strip()

        if row.startswith("["):
            row = row[1:]

        if row.endswith("]"):
            row = row[:-1]

        values = np.array(
            [
                float(x)
                for x in re.findall(
                    r"-?\d+",
                    row
                )
            ],
            dtype=np.float32
        )

        return values

    except:

        return None

# =========================================================
# LOAD FILES
# =========================================================

def load_activity_files(activity_name):

    loaded_files = []

    activity_folder = os.path.join(
        CSI_DATASET_PATH,
        activity_name
    )

    if not os.path.exists(activity_folder):

        print(
            f"Missing Folder: {activity_folder}"
        )

        return loaded_files

    for file in os.listdir(activity_folder):

        if not file.endswith(".csv"):
            continue

        full_path = os.path.join(
            activity_folder,
            file
        )

        print(
            f"Loading {activity_name.upper()} :",
            full_path
        )

        try:

            df = pd.read_csv(full_path)

        except:

            continue

        if "CSI_DATA" not in df.columns:
            continue

        parsed_packets = []

        for row in df["CSI_DATA"].dropna():

            parsed = parse_csi_packet(
                str(row)
            )

            if (
                parsed is not None
                and len(parsed)
                == CSI_EXPECTED_PACKET_LENGTH
            ):

                parsed_packets.append(parsed)

        if len(parsed_packets) > CSI_WINDOW_LENGTH:

            loaded_files.append(
                pd.DataFrame(parsed_packets)
            )

    return loaded_files

# =========================================================
# CSI → AMP + PHASE
# =========================================================

def convert_to_amplitude_phase(df):

    amplitude_rows = []

    phase_rows = []

    for row in df.values:

        # ESP32 CSI FORMAT
        # imag, real, imag, real

        imag = row[::2]

        real = row[1::2]

        amplitude = np.sqrt(
            imag**2 + real**2
        )

        phase = np.unwrap(
            np.arctan2(imag, real)
        )

        amplitude_rows.append(
            amplitude
        )

        phase_rows.append(
            phase
        )

    return (
        pd.DataFrame(amplitude_rows),
        pd.DataFrame(phase_rows)
    )

# =========================================================
# SUBCARRIER SELECTION
# =========================================================

def keep_valid_subcarriers(df):

    left = df.iloc[:, 5:32]

    right = df.iloc[:, 33:60]

    return pd.concat(
        [left, right],
        axis=1
    )

# =========================================================
# MOVING AVERAGE
# =========================================================

def smooth_signal(df):

    return df.rolling(
        CSI_MOVING_AVERAGE
    ).mean().dropna()

# =========================================================
# CREATE WINDOWS
# =========================================================

def build_windows(dataframe, label):

    feature_windows = []

    labels = []

    for start in range(
        0,
        len(dataframe)
        - CSI_WINDOW_LENGTH + 1,
        CSI_WINDOW_STRIDE
    ):

        segment = dataframe.iloc[
            start:
            start + CSI_WINDOW_LENGTH
        ].values

        if (
            segment.shape[0]
            == CSI_WINDOW_LENGTH
        ):

            feature_windows.append(segment)

            labels.append(label)

    return feature_windows, labels

# =========================================================
# PROCESS FILES
# =========================================================

def process_activity_files(
    file_list,
    activity_label,
    X_store,
    y_store
):

    for df in file_list:

        # =====================================
        # AMP + PHASE
        # =====================================

        amp_df, phase_df = (
            convert_to_amplitude_phase(df)
        )

        # =====================================
        # SUBCARRIER SELECTION
        # =====================================

        amp_df = keep_valid_subcarriers(
            amp_df
        )

        phase_df = keep_valid_subcarriers(
            phase_df
        )

        # =====================================
        # MOVING AVERAGE
        # =====================================

        amp_df = smooth_signal(
            amp_df
        )

        phase_df = smooth_signal(
            phase_df
        )

        # =====================================
        # STATIC REMOVAL
        # =====================================

        amp_df = (
            amp_df
            - amp_df.mean(axis=0)
        )

        phase_df = (
            phase_df
            - phase_df.mean(axis=0)
        )

        # =====================================
        # FEATURE CONCAT
        # =====================================

        combined_features = pd.concat(
            [amp_df, phase_df],
            axis=1
        )

        # =====================================
        # MOTION ENERGY FEATURE
        # =====================================

        motion_energy = np.var(
            combined_features.values,
            axis=1
        ).reshape(-1, 1)

        combined_features = np.concatenate(
            [
                combined_features.values,
                motion_energy
            ],
            axis=1
        )

        # =====================================
        # FILTER
        # =====================================

        filtered_features = (
            apply_butterworth_filter(
                combined_features
            )
        )

        filtered_features = np.nan_to_num(
            filtered_features
        )

        filtered_df = pd.DataFrame(
            filtered_features
        )

        # =====================================
        # WINDOWS
        # =====================================

        X_temp, y_temp = build_windows(
            filtered_df,
            activity_label
        )

        X_store.extend(X_temp)

        y_store.extend(y_temp)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading Dataset...\n")

walk_file_list = load_activity_files(
    "walk"
)

static_file_list = load_activity_files(
    "static"
)

fall_file_list = load_activity_files(
    "fall"
)

print("\nWalk Files   :", len(walk_file_list))

print("Static Files :", len(static_file_list))

print("Fall Files   :", len(fall_file_list))

# =========================================================
# FILE SPLIT
# =========================================================

walk_train_files, walk_test_files = (
    train_test_split(
        walk_file_list,
        test_size=0.2,
        random_state=42
    )
)

static_train_files, static_test_files = (
    train_test_split(
        static_file_list,
        test_size=0.2,
        random_state=42
    )
)

fall_train_files, fall_test_files = (
    train_test_split(
        fall_file_list,
        test_size=0.2,
        random_state=42
    )
)

# =========================================================
# STORAGE
# =========================================================

X_train_data = []
y_train_data = []

X_test_data = []
y_test_data = []

# =========================================================
# PROCESS TRAIN
# =========================================================

process_activity_files(
    walk_train_files,
    0,
    X_train_data,
    y_train_data
)

process_activity_files(
    static_train_files,
    1,
    X_train_data,
    y_train_data
)

process_activity_files(
    fall_train_files,
    2,
    X_train_data,
    y_train_data
)

# =========================================================
# PROCESS TEST
# =========================================================

process_activity_files(
    walk_test_files,
    0,
    X_test_data,
    y_test_data
)

process_activity_files(
    static_test_files,
    1,
    X_test_data,
    y_test_data
)

process_activity_files(
    fall_test_files,
    2,
    X_test_data,
    y_test_data
)

# =========================================================
# NUMPY
# =========================================================

X_train_data = np.array(
    X_train_data,
    dtype=np.float32
)

y_train_data = np.array(
    y_train_data
)

X_test_data = np.array(
    X_test_data,
    dtype=np.float32
)

y_test_data = np.array(
    y_test_data
)

print("\nTrain Shape :", X_train_data.shape)

print("Test Shape  :", X_test_data.shape)

# =========================================================
# SHUFFLE
# =========================================================

X_train_data, y_train_data = shuffle(
    X_train_data,
    y_train_data,
    random_state=42
)

X_test_data, y_test_data = shuffle(
    X_test_data,
    y_test_data,
    random_state=42
)

# =========================================================
# NORMALIZATION
# =========================================================

feature_scaler_v2 = StandardScaler()

train_samples = X_train_data.shape[0]

time_steps = X_train_data.shape[1]

feature_count = X_train_data.shape[2]

test_samples = X_test_data.shape[0]

# TRAIN
X_train_reshaped = X_train_data.reshape(
    train_samples * time_steps,
    feature_count
)

X_train_scaled = (
    feature_scaler_v2.fit_transform(
        np.nan_to_num(
            X_train_reshaped
        )
    )
)

# TEST
X_test_reshaped = X_test_data.reshape(
    test_samples * time_steps,
    feature_count
)

X_test_scaled = (
    feature_scaler_v2.transform(
        np.nan_to_num(
            X_test_reshaped
        )
    )
)

# =========================================================
# PCA
# =========================================================

pca_transformer_v2 = PCA(
    n_components=CSI_PCA_COMPONENTS
)

X_train_pca = (
    pca_transformer_v2.fit_transform(
        X_train_scaled
    )
)

X_test_pca = (
    pca_transformer_v2.transform(
        X_test_scaled
    )
)

reduced_feature_count = (
    X_train_pca.shape[1]
)

# RESHAPE TRAIN
X_train_data = X_train_pca.reshape(
    train_samples,
    time_steps,
    reduced_feature_count
)

# RESHAPE TEST
X_test_data = X_test_pca.reshape(
    test_samples,
    time_steps,
    reduced_feature_count
)

print(
    "\nFinal Train Shape :",
    X_train_data.shape
)

print(
    "Final Test Shape  :",
    X_test_data.shape
)

# =========================================================
# SAVE PREPROCESSORS
# =========================================================

pickle.dump(
    feature_scaler_v2,
    open(
        os.path.join(
            CSI_MODEL_SAVE_PATH,
            "advanced_scaler_v2.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    pca_transformer_v2,
    open(
        os.path.join(
            CSI_MODEL_SAVE_PATH,
            "advanced_pca_v2.pkl"
        ),
        "wb"
    )
)

pickle.dump(
    CSI_EXPECTED_PACKET_LENGTH,
    open(
        os.path.join(
            CSI_MODEL_SAVE_PATH,
            "advanced_csi_length_v2.pkl"
        ),
        "wb"
    )
)

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(
        y_train_data
    ),
    y=y_train_data
)

balanced_class_weights = dict(
    enumerate(class_weight_values)
)

print(
    "\nClass Weights :",
    balanced_class_weights
)

# =========================================================
# MODEL
# =========================================================

print("\nBuilding CNN-GRU...\n")

advanced_csi_model_v2 = Sequential()

# =====================================================
# CNN BLOCK 1
# =====================================================

advanced_csi_model_v2.add(
    Conv1D(
        64,
        5,
        activation="relu",
        input_shape=(
            X_train_data.shape[1],
            X_train_data.shape[2]
        )
    )
)

advanced_csi_model_v2.add(
    BatchNormalization()
)

advanced_csi_model_v2.add(
    MaxPooling1D(2)
)

# =====================================================
# CNN BLOCK 2
# =====================================================

advanced_csi_model_v2.add(
    Conv1D(
        128,
        3,
        activation="relu"
    )
)

advanced_csi_model_v2.add(
    BatchNormalization()
)

advanced_csi_model_v2.add(
    MaxPooling1D(2)
)

# =====================================================
# CNN BLOCK 3
# =====================================================

advanced_csi_model_v2.add(
    Conv1D(
        256,
        3,
        activation="relu"
    )
)

advanced_csi_model_v2.add(
    MaxPooling1D(2)
)

# =====================================================
# GRU
# =====================================================

advanced_csi_model_v2.add(
    GRU(64)
)

# =====================================================
# DENSE
# =====================================================

advanced_csi_model_v2.add(
    Dense(
        64,
        activation="relu"
    )
)

advanced_csi_model_v2.add(
    Dropout(0.5)
)

# =====================================================
# OUTPUT
# =====================================================

advanced_csi_model_v2.add(
    Dense(
        3,
        activation="softmax"
    )
)

# =========================================================
# COMPILE
# =========================================================

advanced_csi_model_v2.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================================================
# CALLBACKS
# =========================================================

early_stop_callback_v2 = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr_callback_v2 = (
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
)

advanced_csi_model_v2.summary()

# =========================================================
# TRAIN
# =========================================================

training_history_v2 = (
    advanced_csi_model_v2.fit(
        X_train_data,
        y_train_data,
        epochs=30,
        batch_size=32,
        validation_data=(
            X_test_data,
            y_test_data
        ),
        class_weight=(
            balanced_class_weights
        ),
        callbacks=[
            early_stop_callback_v2,
            reduce_lr_callback_v2
        ],
        shuffle=True
    )
)

# =========================================================
# EVALUATION
# =========================================================

predictions_v2 = np.argmax(
    advanced_csi_model_v2.predict(
        X_test_data
    ),
    axis=1
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_data,
        predictions_v2
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test_data,
        predictions_v2,
        target_names=[
            "walk",
            "static",
            "fall"
        ]
    )
)

# =========================================================
# SAVE MODEL
# =========================================================

advanced_csi_model_v2.save(
    os.path.join(
        CSI_MODEL_SAVE_PATH,
        "advanced_cnn_gru_fall_v2.keras"
    )
)

activity_labels_v2 = {

    0: "walk",

    1: "static",

    2: "fall"
}

pickle.dump(
    activity_labels_v2,
    open(
        os.path.join(
            CSI_MODEL_SAVE_PATH,
            "advanced_labels_v2.pkl"
        ),
        "wb"
    )
)

print("\nTraining Complete.")