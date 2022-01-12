#%%
import numpy as np
import torch, os, random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from time import time
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
#
from torch.optim import SGD, Adam
from sklearn.metrics import confusion_matrix
from ipdb import set_trace as st
from torch.utils.data import random_split
#

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", help="input training data path")
args = parser.parse_args()

start = time()
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

# if torch.cuda.is_available() == True:
#     device = "cuda"
#     gpus = 1
#     print("Use GPU!")
# else:
#     device = "cpu"
#     gpus = 0
device = "cpu"
gpus = 0

norm = True
# testdata = np.load("../project_dataset/testdata.npy")
testdata = np.load(args.data)

if norm:
    scalerA = StandardScaler()
    scalerA.fit(testdata[:,0,:])
    tmp = scalerA.transform(testdata[:,0,:])
    testdata[:,0,:] = tmp
    scalerB = StandardScaler()
    scalerB.fit(testdata[:,1,:])
    tmp = scalerB.transform(testdata[:,1,:])
    testdata[:,1,:] = tmp

# data loader
# define dataset here
class RawDataset(Dataset):
    def __init__(self, traindata, trainlabel):
        self.traindata = torch.from_numpy(np.array(traindata).astype(np.float32))
        if trainlabel is not None:
            self.trainlabel = torch.from_numpy(np.array(trainlabel).astype(np.int64))
        else:
            self.trainlabel = None
    def __len__(self):
        return len(self.traindata)
    def __getitem__(self,idx):
        sample = self.traindata[idx]
        if self.trainlabel is not None:
            target = self.trainlabel[idx]
            return sample, target
        else:
            return sample

testset = RawDataset(testdata,None) 
testloader = DataLoader(testset,batch_size=1,shuffle=False)

class MyDSPNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        #
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 20, kernel_size=13, stride=7),
            nn.ReLU(),
            nn.Conv1d(20, 40, kernel_size=11, stride=7),
            nn.ReLU(),
            nn.Conv1d(40, 80, kernel_size=9, stride=5),
            nn.ReLU(),
            nn.Conv1d(80, 160, kernel_size=7, stride=5),
            )
        self.clf = nn.Linear(1920, 3) # classifier
        #
        
    def forward(self, x): #input x : 16000 points
        #
        x = self.encoder(x)
        x = x.view(x.size(0),-1) #flatten 
        output = self.clf(x) #go through fully connected layer
        #
        return output #output: 0 or 1 or 2
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)#1e-2)
        return optimizer
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x) #self = MyDSPNet
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss

# model
model = MyDSPNet()
model.load_state_dict(torch.load("./weight.pt"))
model.eval()

model = model.to(device) # move model to gpu
with open("result.csv","w") as f:
    f.write("id,category\n")
    for i, x in enumerate(testloader):#test data assign to x
        x = x.to(device)# move test data to gpu
        output = model(x)#output is a 3-dim vector
        pred = output.argmax(dim=1, keepdim=True)
        f.write("%d,%d\n"%(i,pred.item()))
