import numpy as np
import pandas as pd
import os
import re
from math import sqrt
from scipy.signal import savgol_filter, butter, filtfilt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIG
# =========================

DATASET_PATH = r"C:\esp32-csi-tool\datasets1"

SEG_LEN = 200
STEP = 100
BATCH_SIZE = 32
EPOCHS = 40

DROP_COLS = [2,3,4,5,32,59,60,61,62,63]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# FILTER
# =========================

def highpass(signal):

    b,a = butter(3,0.1,btype="highpass")

    return filtfilt(b,a,signal)

# =========================
# LOAD CSI DATA
# =========================

features=[]
labels=[]

for activity in os.listdir(DATASET_PATH):

    path=os.path.join(DATASET_PATH,activity)

    if not os.path.isdir(path):
        continue

    print("Loading:",activity)

    for file in os.listdir(path):

        if not file.endswith(".csv"):
            continue

        df=pd.read_csv(os.path.join(path,file))

        if "CSI_DATA" not in df.columns:
            continue

        amp_rows=[]

        for val in df["CSI_DATA"]:

            m=re.search(r"\[(.*?)\]",str(val))

            if not m:
                continue

            raw=[int(x) for x in re.split(r"[,\s]+",m.group(1)) if x]

            imag=raw[::2]
            real=raw[1::2]

            amp=[sqrt(i*i+r*r) for i,r in zip(imag,real)]

            amp_rows.append(amp)

        if len(amp_rows)<SEG_LEN:
            continue

        A=pd.DataFrame(amp_rows)

        for c in A.columns:

            if len(A[c])>=11:

                A[c]=savgol_filter(A[c],11,3)

        A=A-A.mean(axis=0)

        for c in A.columns:

            A[c]=highpass(A[c])

        valid=[c for c in DROP_COLS if c<A.shape[1]]

        A.drop(A.columns[valid],axis=1,inplace=True)

        A=A.iloc[:,10:36]

        feat=A.values

        for start in range(0,len(feat)-SEG_LEN+1,STEP):

            segment=feat[start:start+SEG_LEN]

            segment=(segment-segment.mean())/(segment.std()+1e-6)

            features.append(segment)
            labels.append(activity)

X=np.array(features).astype("float32")
y=np.array(labels)

print("Dataset shape:",X.shape)

# =========================
# LABEL ENCODING
# =========================

le=LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.25,stratify=y,random_state=42
)

# =========================
# DATASET CLASS
# =========================

class CSIDataset(Dataset):

    def __init__(self,X,y):

        self.X=torch.tensor(X)
        self.y=torch.tensor(y)

    def __len__(self):

        return len(self.X)

    def __getitem__(self,idx):

        return self.X[idx],self.y[idx]

train_loader=DataLoader(CSIDataset(X_train,y_train),batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(CSIDataset(X_test,y_test),batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================

class CNN_GRU_Model(nn.Module):

    def __init__(self,num_features=26,num_classes=3):

        super().__init__()

        self.conv1=nn.Conv1d(num_features,64,3,padding=1)
        self.bn1=nn.BatchNorm1d(64)

        self.conv2=nn.Conv1d(64,128,3,padding=1)
        self.bn2=nn.BatchNorm1d(128)

        self.pool=nn.MaxPool1d(2)

        self.gru=nn.GRU(128,128,num_layers=2,batch_first=True)

        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,num_classes)

        self.dropout=nn.Dropout(0.3)

    def forward(self,x):

        x=x.permute(0,2,1)

        x=F.relu(self.bn1(self.conv1(x)))
        x=self.pool(x)

        x=F.relu(self.bn2(self.conv2(x)))
        x=self.pool(x)

        x=x.permute(0,2,1)

        out,_=self.gru(x)

        x=out[:,-1,:]

        x=F.relu(self.fc1(x))
        x=self.dropout(x)

        x=self.fc2(x)

        return x

model=CNN_GRU_Model(num_features=X.shape[2],num_classes=len(le.classes_)).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)

# =========================
# TRAIN LOOP
# =========================

for epoch in range(EPOCHS):

    model.train()

    total_loss=0

    for Xb,yb in train_loader:

        Xb=Xb.to(device)
        yb=yb.to(device)

        optimizer.zero_grad()

        out=model(Xb)

        loss=criterion(out,yb)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss:",total_loss)

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(),"cnn_gru_csi_model.pth")

np.save("cnn_gru_classes.npy",le.classes_)

print("Model saved")