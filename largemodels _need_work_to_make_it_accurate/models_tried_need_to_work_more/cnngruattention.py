
# WIFI CSI HAR TRAINING PIPELINE
# ACTIVITIES: WALK / STATIC / FALL


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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization



# CONFIG


DATASET_DIR = r"C:\esp32-csi-tool\dataset3"
MODEL_DIR = r"C:\esp32-csi-tool\model3"

WINDOW_SIZE = 300
STRIDE = 100
MOVING_AVG = 5

os.makedirs(MODEL_DIR, exist_ok=True)


# BUTTERWORTH FILTER


def butter_filter(data, cutoff=0.1, order=3):

    b,a = butter(order, cutoff)

    return filtfilt(b,a,data,axis=0)



# PARSE CSI

def parse_csi(row):

    try:

        row = row.strip()

        if row.startswith("["):
            row = row[1:]

        if row.endswith("]"):
            row = row[:-1]

        import re

        return np.array(
            [float(x) for x in re.findall(r'-?\d+', row)],
            dtype=float
    )

    except:

        return None



# LOAD DATASET


def load_csi_files():

    walk_files=[]
    static_files=[]
    fall_files=[]
    walk_dir=os.path.join(DATASET_DIR,"walk")
    static_dir=os.path.join(DATASET_DIR,"static")
    fall_dir=os.path.join(DATASET_DIR,"fall")

    # WALK
    for file in os.listdir(walk_dir):

        if file.endswith(".csv"):

            path=os.path.join(walk_dir,file)

            print("Loading WALK:",path)

            df=pd.read_csv(path)

            csi_rows=[]

            for row in df["CSI_DATA"].dropna():

                parsed=parse_csi(str(row))

                if parsed is not None:

                    csi_rows.append(parsed)

            if len(csi_rows)>0:

                walk_files.append(pd.DataFrame(csi_rows))

    # STAND
    for file in os.listdir(static_dir):

        if file.endswith(".csv"):

            path=os.path.join(static_dir,file)

            print("Loading STATIC:",path)

            df=pd.read_csv(path)

            csi_rows=[]

            for row in df["CSI_DATA"].dropna():

                parsed=parse_csi(str(row))

                if parsed is not None:

                    csi_rows.append(parsed)

            if len(csi_rows)>0:

                static_files.append(pd.DataFrame(csi_rows))

    # FALL
    for file in os.listdir(fall_dir):

        if file.endswith(".csv"):

            path=os.path.join(fall_dir,file)

            print("Loading FALL:",path)

            df=pd.read_csv(path)

            csi_rows=[]

            for row in df["CSI_DATA"].dropna():

                parsed=parse_csi(str(row))

                if parsed is not None:

                    csi_rows.append(parsed)

            if len(csi_rows)>0:

                fall_files.append(pd.DataFrame(csi_rows))

    return walk_files,static_files,fall_files



# CSI → AMP + PHASE


def convert_csi_to_amp_phase(df):

    amps=[]
    phases=[]

    for row in df.values:

        real=row[::2]
        imag=row[1::2]

        amp=np.sqrt(imag**2+real**2)

        phase=np.unwrap(np.arctan2(imag,real))

        amps.append(amp)
        phases.append(phase)

    return pd.DataFrame(amps),pd.DataFrame(phases)



# SUBCARRIER SELECTION


def select_subcarriers(df):

    part1=df.iloc[:,5:32]
    part2=df.iloc[:,33:60]

    return pd.concat([part1,part2],axis=1)



# MOVING AVERAGE


def moving_average(df):

    return df.rolling(MOVING_AVG).mean().dropna()



# WINDOW SEGMENTATION


def create_windows(df,label):

    X=[]
    y=[]

    for i in range(0,len(df)-WINDOW_SIZE,STRIDE):

        window=df.iloc[i:i+WINDOW_SIZE].values

        if window.shape[0]==WINDOW_SIZE:

            X.append(window)
            y.append(label)

    return X,y


# BUILD DATASET


print("\nLoading dataset...\n")

walk_files,static_files,fall_files=load_csi_files()

X=[]
y=[]

def process_files(files,label):

    global X,y

    for df in files:

        amp,phase=convert_csi_to_amp_phase(df)

        amp=select_subcarriers(amp)
        phase=select_subcarriers(phase)

        amp=moving_average(amp)
        phase=moving_average(phase)

        features=pd.concat([amp,phase],axis=1)

        features=pd.DataFrame(
            butter_filter(features.values)
        )

        X_tmp,y_tmp=create_windows(features,label)

        X.extend(X_tmp)
        y.extend(y_tmp)


process_files(walk_files,0)
process_files(static_files,1)
process_files(fall_files,2)
X=np.array(X)
y=np.array(y)

print("\nDataset shape:",X.shape)

X,y=shuffle(X,y,random_state=42)

# =========================================
# NORMALIZATION
# =========================================

scaler=StandardScaler()

X_reshaped=X.reshape(-1,X.shape[-1])

X_scaled=scaler.fit_transform(X_reshaped)


# PCA DENOISING


pca=PCA(n_components=0.95)

samples = X.shape[0]
timesteps = X.shape[1]
features = X.shape[2]

X_reshaped = X.reshape(-1,features)

X_scaled = scaler.fit_transform(X_reshaped)

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)

new_features = X_pca.shape[1]

X = X_pca.reshape(samples,timesteps,new_features)

pickle.dump(scaler,open(os.path.join(MODEL_DIR,"cnn_scaler2.pkl"),"wb"))
pickle.dump(pca,open(os.path.join(MODEL_DIR,"cnn_pca2.pkl"),"wb"))


# TRAIN TEST SPLIT


X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# CLASS WEIGHTS


weights=compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

class_weights=dict(enumerate(weights))


# CNN-GRU MODEL


print("\nTraining CNN-GRU-Attention...\n")

inputs = Input(shape=X_train.shape[1:])

# CNN BLOCK 1
x = Conv1D(64,5,activation="relu")(inputs)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

# CNN BLOCK 2
x = Conv1D(128,3,activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling1D(2)(x)

# CNN BLOCK 3
x = Conv1D(256,3,activation="relu")(x)
x = MaxPooling1D(2)(x)

# GRU
x = GRU(128,return_sequences=True)(x)

# ATTENTION
attention = Attention()([x,x])

# GLOBAL POOLING
x = GlobalAveragePooling1D()(attention)

# DENSE
x = Dense(64,activation="relu")(x)

x = Dropout(0.5)(x)

outputs = Dense(3,activation="softmax")(x)

model = Model(inputs,outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
model.summary()

model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test,y_test),
    class_weight=class_weights,
    callbacks=[early]
)


# EVALUATION


y_pred=np.argmax(model.predict(X_test),axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))


# SAVE MODEL


model.save(os.path.join(MODEL_DIR,"cnn_gru_modelattention.keras"))

labels={
    0:"walk",
    1:"static",
    2:"fall"
}

pickle.dump(labels,open(os.path.join(MODEL_DIR,"labels2.pkl"),"wb"))

print("\nTraining complete.")