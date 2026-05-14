import numpy as np
import pandas as pd
import glob
import re
import os

from math import sqrt
from scipy.signal import savgol_filter, butter, filtfilt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =========================
# CONFIG
# =========================

DATASET_PATH = "C:\esp32-csi-tool\datasets1"

SEG_LEN = 700
STEP = 200

DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

EPOCHS = 50
BATCH = 16
LR = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CSI PARSER
# =========================

def parse_csi(csi_string):

    nums = [int(x) for x in re.findall(r'-?\d+', str(csi_string))]

    imag = nums[::2]
    real = nums[1::2]

    amp = []

    for i,r in zip(imag,real):

        amp.append(sqrt(i*i + r*r))

    return np.array(amp)


# =========================
# HIGH PASS FILTER
# =========================

def highpass(signal):

    b,a = butter(3,0.1,btype="highpass")

    return filtfilt(b,a,signal)


# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

packets = []
labels = []

activities = os.listdir(DATASET_PATH)

for act in activities:

    files = glob.glob(f"{DATASET_PATH}/{act}/*.csv")

    for f in files:

        df = pd.read_csv(f)

        for row in df["CSI_DATA"]:

            feat = parse_csi(row)

            packets.append(feat)
            labels.append(act)

# find most common length
lengths = [len(p) for p in packets]

from collections import Counter
target_len = Counter(lengths).most_common(1)[0][0]

print("Most common CSI length:", target_len)

# filter packets
filtered_packets = []
filtered_labels = []

for p,l in zip(packets,labels):

    if len(p) == target_len:
        filtered_packets.append(p)
        filtered_labels.append(l)

packets = np.array(filtered_packets)
labels = np.array(filtered_labels)

print("Filtered packets:", packets.shape)

# =========================
# PREPROCESSING
# =========================

print("Preprocessing...")

A = pd.DataFrame(packets)

# smoothing
for c in A.columns:

    if len(A[c]) >= 11:

        A[c] = savgol_filter(A[c],11,3)

# mean removal
A = A - A.mean(axis=0)

# highpass
for c in A.columns:

    A[c] = highpass(A[c])

# drop noisy carriers
valid = [c for c in DROP_COLS if c < A.shape[1]]

A.drop(A.columns[valid],axis=1,inplace=True)

# select stable carriers
A = A.iloc[:,12:40]

print("Features after selection:", A.shape)

# =========================
# SEGMENTATION
# =========================

print("Creating segments...")

segments = []
seg_labels = []

labels = np.array(labels)

for i in range(0,len(A)-SEG_LEN,STEP):

    seg = A.iloc[i:i+SEG_LEN].values

    seg = (seg - seg.mean())/(seg.std()+1e-6)

    segments.append(seg)

    seg_labels.append(labels[i])

X = np.array(segments)

y = np.array(seg_labels)

print("Segment shape:",X.shape)

# =========================
# LABEL ENCODING
# =========================

le = LabelEncoder()

y = le.fit_transform(y)

classes = le.classes_

np.save("cnn_gru_classes.npy",classes)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

X_train = torch.tensor(X_train,dtype=torch.float32).to(device)
X_test = torch.tensor(X_test,dtype=torch.float32).to(device)

y_train = torch.tensor(y_train).to(device)
y_test = torch.tensor(y_test).to(device)

# =========================
# MODEL
# =========================

class CNN_GRU_Model(nn.Module):

    def __init__(self,num_features,num_classes):

        super().__init__()

        self.conv1 = nn.Conv1d(num_features,64,5,padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64,128,5,padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )

        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,num_classes)

        self.drop = nn.Dropout(0.4)

    def forward(self,x):

        x = x.permute(0,2,1)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = x.permute(0,2,1)

        out,_ = self.gru(x)

        x = out[:,-1,:]

        x = self.drop(F.relu(self.fc1(x)))

        return self.fc2(x)


model = CNN_GRU_Model(28,len(classes)).to(device)

# =========================
# TRAINING
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=LR)

print("Training started...")

for epoch in range(EPOCHS):

    model.train()

    out = model(X_train)

    loss = criterion(out,y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.no_grad():

        test_out = model(X_test)

        pred = torch.argmax(test_out,1)

        acc = (pred==y_test).float().mean()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss {loss.item():.4f} | Test Acc {acc:.4f}")

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(),"cnn_gru_csi_model.pth")

print("Model saved!")