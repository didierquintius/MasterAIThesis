# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:44:51 2020

@author: didie
"""
import os, pickle, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.io import loadmat
#%%
EEG_data, labels, source = pickle.load(open("../Data/EEG/data_1_1_200.pkl", "rb"))
#%%

class NeuralNet(nn.Module):
    
    def __init__(self, input_size, neurons):
        super().__init__()
        self.fc0 = nn.Linear(input_size, neurons[0])
        self.fc1 = nn.Linear(neurons[0], neurons[1])
        self.out = nn.Linear(neurons[-1], 1)
    
    def forward(self,x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
#%%
Net = NeuralNet(108,[5, 3])
optimizer = optim.Adam(Net.parameters(), lr = 0.0005)
loss_function = nn.MSELoss()
#%%
def split_data(X, y, validation_perc = 0.1, test_perc = 0.1):
    
    elecs, time_steps, trials = X.shape
    
    train_ind = int((1 - validation_perc - test_perc) * trials)
    val_ind = int((1 - test_perc) * trials)
    
    X_train = torch.Tensor(X[:,:,:train_ind].copy()).transpose(0,1).reshape(train_ind * time_steps,elecs)
    X_val = torch.Tensor(X[:,:,train_ind:val_ind].copy()).transpose(0,1).reshape((val_ind - train_ind) * time_steps,elecs)
    X_test = torch.Tensor(X[:,:,val_ind:trials].copy()).transpose(0,1).reshape((trials - val_ind) * time_steps,elecs)
    
    y_train = torch.Tensor(y[:,:,:train_ind].T.copy()).transpose(0,1).reshape(train_ind * time_steps,3)
    y_val = torch.Tensor(y[:,:,train_ind:val_ind].T.copy()).transpose(0,1).reshape((val_ind - train_ind) * time_steps,3)
    y_test = torch.Tensor(y[:,:,val_ind:trials].T.copy()).transpose(0,1).reshape((trials - val_ind) * time_steps,3)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
def do_val_check(net, X, y, lag, plot = False):
    loss = 0
    with torch.no_grad():
        real = y
        pred = net(X)
        loss = loss_function(pred, real)
    
    if plot:
        for i in range(0, y.shape[0],int(y.shape[0] / 10)):
            for j in range(1):
                plt.plot(range(100),real[i:(i+ 100),j],"r", 
                         range(100),pred[i:(i+ 100),j],"b")            
                
    return np.float(loss)
#%%
X_train, X_val, X_test, y_train, y_val, y_test = split_data(EEG_data, source)  
 #%% 
x = 500
xi = torch.Tensor(EEG_data[:,:,0:x]).transpose(0,1).reshape(x * 1000,108)
yi = torch.Tensor(source[:,:,0:x]).transpose(0,1).reshape(x *1000,3)
#%%
val_counter = 0
min_val = np.inf
for j in tqdm(range(1000)):
    val_check = do_val_check(Net, X_val, y_val[:,0], 1)
    if val_check > min_val:
        val_counter+=1
        if val_counter > 5: break
    else:
        min_val = val_check
        val_counter = 0
    for i in range(0, y_train.shape[1], 50):
        Net.zero_grad()
        outputs = Net(X_train[i:(i+50),:])
        loss = loss_function(outputs, y_train[i:(i+50), 0])
        loss.backward()
        optimizer.step()
print(np.float(loss))
#%%
print(do_val_check(Net, X_test, y_test, 1,True))
        
