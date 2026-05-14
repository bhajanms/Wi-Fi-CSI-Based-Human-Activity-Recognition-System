import numpy as np
import os
import re
from math import sqrt, atan2

from scipy.signal import butter, filtfilt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import joblib


# ===============================
# CONFIG
# ===============================

DATASET_PATH = r"C:\esp32-csi-tool\datasets"

SEG_LEN = 50
STEP = 10

REMOVE_SUB = [2,3,4,5,32,59,60,61,62,63]


# ===============================
# LOWPASS FILTER
# ===============================

def butter_lowpass(data, cutoff=0.1, fs=1.0, order=3):

    b, a = butter(order, cutoff/(0.5*fs), btype='low')

    return filtfilt(b, a, data)


# ===============================
# PARSE CSI
# ===============================

def parse_csi(line):

    nums = re.findall(r'-?\d+', line)
    nums = list(map(int, nums))

    real = nums[::2]
    imag = nums[1::2]

    amp = []
    phase = []

    for r,i in zip(real,imag):

        amp.append(sqrt(r*r + i*i))
        phase.append(atan2(i,r))

    amp = np.array(amp)
    phase = np.array(phase)

    return np.concatenate([amp,phase])


# ===============================
# LOAD FILE
# ===============================

def load_file(path):

    rows = []
    expected_len = None

    with open(path,'r',errors="ignore") as f:

        for line in f:

            if "CSI_DATA" not in line:
                continue

            try:

                feat = parse_csi(line)

                feat = np.delete(feat, REMOVE_SUB)

                if expected_len is None:
                    expected_len = len(feat)

                if len(feat) != expected_len:
                    continue

                rows.append(feat)

            except:
                continue

    if len(rows) == 0:
        return np.empty((0,0))

    data = np.array(rows)

    # apply filtering
    for i in range(data.shape[1]):
        data[:,i] = butter_lowpass(data[:,i])

    return data


# ===============================
# SEGMENT DATA
# ===============================

def segment(data):

    segments = []

    for i in range(0, len(data) - SEG_LEN + 1, STEP):

        seg = data[i:i+SEG_LEN]
        segments.append(seg)

    return segments


# ===============================
# LOAD DATASET
# ===============================

X = []
y = []

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH,label)

    if not os.path.isdir(folder):
        continue

    print("Processing:",label)

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        data = load_file(path)

        if data.shape[0] < SEG_LEN:
            continue

        segs = segment(data)

        for s in segs:

            X.append(s)
            y.append(label)


X = np.array(X)
y = np.array(y)

print("Dataset shape:",X.shape)


# ===============================
# SHUFFLE DATASET
# ===============================

X, y = shuffle(X, y, random_state=42)


# ===============================
# NORMALIZE
# ===============================

samples = X.shape[0]
timesteps = X.shape[1]
features = X.shape[2]

X = X.reshape(-1,features)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = X.reshape(samples,timesteps,features)

joblib.dump(scaler,"scaler.pkl")


# ===============================
# LABEL ENCODING
# ===============================

le = LabelEncoder()

y = le.fit_transform(y)

joblib.dump(le,"labels.pkl")

print("Classes:",le.classes_)


# ===============================
# TRAIN TEST SPLIT
# ===============================

X_train,X_test,y_train,y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42,
    stratify=y

)


# ===============================
# MODEL
# ===============================

model = Sequential()

model.add(Conv1D(64,3,activation='relu',input_shape=(timesteps,features)))
model.add(MaxPooling1D(2))

model.add(Conv1D(128,3,activation='relu'))
model.add(MaxPooling1D(2))

model.add(Bidirectional(LSTM(64)))

model.add(Dropout(0.5))

model.add(Dense(64,activation='relu'))

model.add(Dense(len(le.classes_),activation='softmax'))


model.compile(

    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']

)

model.summary()


# ===============================
# TRAIN
# ===============================

early = EarlyStopping(

    monitor="val_loss",
    patience=8,
    restore_best_weights=True

)

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test,y_test),

    epochs=40,
    batch_size=16,

    callbacks=[early]

)


# ===============================
# TEST
# ===============================

loss,acc = model.evaluate(X_test,y_test)

print("Final Accuracy:",acc)


# ===============================
# SAVE
# ===============================

model.save("cnn_bilstm_csi_model.keras")

print("Model saved successfully")