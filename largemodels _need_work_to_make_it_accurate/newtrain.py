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
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG
# =========================

DATASET_PATH = r"C:\esp32-csi-tool\datasets"
SEG_LEN = 150
STEP = SEG_LEN // 2
DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

# =========================
# LABEL MAP — NO LIMB MERGE
# =========================

def map_label(lbl):
    if lbl == "GU":
        return None   # skip only GU
    return lbl        # keep all others separate

# =========================
# LOAD CSV FILES
# =========================

activity_dfs = []
file_labels = []

for activity in os.listdir(DATASET_PATH):

    mapped = map_label(activity)
    if mapped is None:
        continue

    act_path = os.path.join(DATASET_PATH, activity)
    if not os.path.isdir(act_path):
        continue

    for file in os.listdir(act_path):

        if not file.endswith(".csv"):
            continue

        full = os.path.join(act_path, file)

        try:
            df = pd.read_csv(full)
        except:
            continue

        if "CSI_DATA" not in df.columns:
            continue

        activity_dfs.append(df)
        file_labels.append(mapped)

print("Loaded CSI files:", len(activity_dfs))

# =========================
# CSI → AMP + PHASE
# =========================

feat_arrays = []
feat_labels = []

for idx, df in enumerate(activity_dfs):

    amp_rows = []
    phase_rows = []

    for val in df["CSI_DATA"]:

        m = re.findall(r"\[(.*)\]", str(val))
        if not m:
            continue

        raw = [int(x) for x in re.split(r"[,\s]+", m[0].strip()) if x]

        imag = raw[::2]
        real = raw[1::2]

        amp = [sqrt(i*i + r*r) for i,r in zip(imag,real)]
        phase = [atan2(i,r) for i,r in zip(imag,real)]

        amp_rows.append(amp)
        phase_rows.append(phase)

    if len(amp_rows) < SEG_LEN:
        continue

    amp_df = pd.DataFrame(amp_rows)
    phase_df = pd.DataFrame(phase_rows)

    for c in amp_df.columns:
        if len(amp_df[c]) >= 11:
            amp_df[c] = savgol_filter(amp_df[c], 11, 3)
            phase_df[c] = savgol_filter(phase_df[c], 11, 3)

    valid = [c for c in DROP_COLS if c < amp_df.shape[1]]
    amp_df.drop(amp_df.columns[valid], axis=1, inplace=True)
    phase_df.drop(phase_df.columns[valid], axis=1, inplace=True)

    feat = np.stack([amp_df.values, phase_df.values], axis=-1)

    feat_arrays.append(feat)
    feat_labels.append(file_labels[idx])

print("Feature arrays built:", len(feat_arrays))

# =========================
# SEGMENT WITH OVERLAP
# =========================

segments = []
seg_labels = []

for i, arr in enumerate(feat_arrays):

    for start in range(0, len(arr)-SEG_LEN+1, STEP):
        segments.append(arr[start:start+SEG_LEN])
        seg_labels.append(feat_labels[i])

X = np.array(segments)
y = np.array(seg_labels)

print("Segment tensor shape:", X.shape)
print("\nClass counts:")
print(pd.Series(y).value_counts())

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# NORMALIZE
# =========================

mean = X_train.mean(axis=(0,1,2))
std  = X_train.std(axis=(0,1,2)) + 1e-6

X_train = (X_train - mean)/std
X_test  = (X_test - mean)/std

np.save("norm_mean.npy", mean)
np.save("norm_std.npy", std)

# =========================
# LABEL ENCODE
# =========================

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

np.save("class_names.npy", le.classes_)
print("Classes:", le.classes_)

# =========================
# CLASS WEIGHTS
# =========================

cw = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)

cw_dict = dict(enumerate(cw))

# =========================
# CNN MODEL
# =========================

model = keras.Sequential([

    keras.Input(shape=(SEG_LEN, X.shape[2], 2)),

    keras.layers.Conv2D(32,3,padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(64,3,padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2),

    keras.layers.Conv2D(128,3,padding='same',activation='relu'),
    keras.layers.BatchNormalization(),

    keras.layers.GlobalAveragePooling2D(),

    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Dense(len(le.classes_),activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN
# =========================

model.fit(
    X_train,
    y_train_enc,
    epochs=60,
    batch_size=16,
    validation_split=0.2,
    shuffle=True,
    class_weight=cw_dict
)

model.save("model_csi_new.keras")

# =========================
# TEST
# =========================

loss, acc = model.evaluate(X_test, y_test_enc)
print("Final Test Accuracy:", acc)

