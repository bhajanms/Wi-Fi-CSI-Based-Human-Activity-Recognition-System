import serial
import numpy as np
import pandas as pd
import re
from math import sqrt, atan2
from scipy.signal import savgol_filter, butter, filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# =========================
# CONFIG
# =========================

PORT = "COM10"
BAUD = 115200

SEG_LEN = 700
STEP = 200

DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

MODEL_PATH = "cnn_gru_csi_model.pth"
CLASS_PATH = "cnn_gru_classes.npy"

SERVER_URL = "http://172.20.10.8:5000/api/activity/update"

PERSON_ID = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# HIGH PASS FILTER
# =========================

def highpass(signal):
    b,a = butter(3,0.1,btype="highpass")
    return filtfilt(b,a,signal)

# =========================
# MODEL
# =========================

class CNN_GRU_Model(nn.Module):

    def __init__(self,num_features,num_classes):

        super().__init__()

        self.conv1=nn.Conv1d(num_features,64,3,padding=1)
        self.bn1=nn.BatchNorm1d(64)

        self.conv2=nn.Conv1d(64,128,3,padding=1)
        self.bn2=nn.BatchNorm1d(128)

        self.conv3=nn.Conv1d(128,128,3,padding=1)
        self.bn3=nn.BatchNorm1d(128)

        self.pool=nn.MaxPool1d(2)

        self.gru=nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,num_classes)

        self.dropout=nn.Dropout(0.4)

    def forward(self,x):

        x=x.permute(0,2,1)

        x=F.relu(self.bn1(self.conv1(x)))
        x=self.pool(x)

        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)

        x=F.relu(self.bn3(self.conv3(x)))

        x=x.permute(0,2,1)

        out,_=self.gru(x)

        x=out[:,-1,:]

        x=F.relu(self.fc1(x))
        x=self.dropout(x)

        x=self.fc2(x)

        return x

# =========================
# LOAD MODEL
# =========================

classes = np.load(CLASS_PATH, allow_pickle=True)

model = CNN_GRU_Model(28,len(classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.eval()

print("Model Loaded")

# =========================
# SERIAL START
# =========================

ser = serial.Serial(PORT,BAUD,timeout=1)

buffer=[]
packet_count=0

print("Live Prediction Started")

# =========================
# MAIN LOOP
# =========================

while True:

    line = ser.readline().decode(errors="ignore")

    if "CSI_DATA" not in line:
        continue

    m=re.search(r"\[(.*?)\]",line)

    if not m:
        continue

    raw=[int(x) for x in re.split(r"[,\s]+",m.group(1)) if x]

    imag=raw[::2]
    real=raw[1::2]

    amp=[]
    phase=[]

    for i,r in zip(imag,real):

        amp.append(sqrt(i*i + r*r))
        phase.append(atan2(i,r))

    combined = amp + phase

    buffer.append(combined)

    packet_count+=1

    print("Packet:",packet_count)

    if len(buffer) < SEG_LEN:
        continue

    print("700 packets collected → predicting")

    A=pd.DataFrame(buffer)

    # smoothing
    for c in A.columns:
        if len(A[c])>=11:
            A[c]=savgol_filter(A[c],11,3)

    # remove mean
    A=A-A.mean(axis=0)

    # highpass
    for c in A.columns:
        A[c]=highpass(A[c])

    # drop noisy subcarriers
    valid=[c for c in DROP_COLS if c<A.shape[1]]
    A.drop(A.columns[valid],axis=1,inplace=True)

    # keep same subcarriers as training
    A=A.iloc[:,12:40]

    segment=A.values

    segment=(segment-segment.mean())/(segment.std()+1e-6)

    X=np.expand_dims(segment,axis=0).astype("float32")
    X=torch.tensor(X).to(device)

    # =========================
    # PREDICT
    # =========================

    with torch.no_grad():

        out=model(X)

        prob=torch.softmax(out,1)

        conf,pred=torch.max(prob,1)

    activity=classes[pred.item()]
    confidence=float(conf.item())

    print("Activity:",activity)
    print("Confidence:",round(confidence,3))
    print("--------------------------------")

    # =========================
    # SEND TO SERVER
    # =========================

    data = {
        "person_id": PERSON_ID,
        "activity": activity,
        "confidence": confidence
    }

    try:
        requests.post(SERVER_URL, json=data)
    except:
        print("Server not reachable")

    # =========================
    # SLIDING WINDOW
    # =========================

    buffer = buffer[STEP:]