# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:24:43 2020

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
#from scipy.io import loadmat


#%%
def format_data(snr, cnr, noise, val = 0.1, test = 0.1):
    EEG_data, labels, source = pickle.load(open("../Data/EEG/data_"+str(snr)+"_"+str(cnr)+"_"+str(noise)+".pkl", "rb"))
    elecs, time, trials = EEG_data.shape
    source = source.reshape((3,time * trials))
    EEG_data = EEG_data.reshape((108, time * trials))
    
    elecs, time, trials
    EEG_data = EEG_data.reshape((elecs, time * trials))
    source = source.reshape((3, time * trials))
    new_labels = np.zeros((labels.shape[0], time * trials))
    for trial in range(trials):
        new_labels[:, trial * time: ((trial + 1) * time)] = np.tile(labels[:,trial], (time, 1)).transpose()
    
    return split_data(EEG_data, source, new_labels, val, test)
    
def split_data(X, y, label, validation_perc = 0.1, test_perc = 0.1):
    
    elecs, trials = X.shape
    
    train_ind = int((1 - validation_perc - test_perc) * trials)
    val_ind = int((1 - test_perc) * trials)
    
    X_train = torch.Tensor(X[:,:train_ind].copy()).view(elecs, train_ind).transpose(0,1)
    X_val = torch.Tensor(X[:,train_ind:val_ind].copy()).view(elecs, val_ind - train_ind).transpose(0,1)
    X_test = torch.Tensor(X[:,val_ind:trials].copy()).view(elecs, trials - val_ind).transpose(0,1)
    
    y_train = torch.Tensor(y[:,:train_ind].copy()).view(3, train_ind).transpose(0,1)
    y_val = torch.Tensor(y[:,train_ind:val_ind].copy()).view(3, val_ind - train_ind).transpose(0,1)
    y_test = torch.Tensor(y[:,val_ind:trials].copy()).view(3, trials - val_ind).transpose(0,1)
    
    label_train = torch.Tensor(label[:,:train_ind].copy()).view(label.shape[0], train_ind).transpose(0,1)
    label_val = torch.Tensor(label[:,train_ind:val_ind].copy()).view(label.shape[0], val_ind - train_ind).transpose(0,1)
    label_test = torch.Tensor(label[:,val_ind:trials].copy()).view(label.shape[0], trials - val_ind).transpose(0,1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_train, label_val, label_test

def train_net(net, X, y, loss_function, optimizer, batch_size = 20, epochs = 1):
    for epoch in range(epochs):
        for i in tqdm(range(0, X.shape[0], batch_size)):
            
            batch_X = X[i : (i + batch_size), :]
            batch_y = y[i : (i + batch_size), :]
            
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return net

def eval_net(net, X, y, loss_function, plot_inter = 5,plot = False):
    X = X
    y = y
    pred = net(X)
    if plot:
        for j in range(0, X.shape[0], int(X.shape[0] / plot_inter)):
            for k in range(3):
                plt.plot(range(200), y[j:(j+200),k],  "r-", 
                         range(200), pred[j:(j+200),k].detach().numpy(), "b-")
                plt.show()
    loss = loss_function(pred, y)
    return loss

class Net(nn.Module):
    def __init__(self,input_size, l1, l2, out):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = nn.Linear(input_size,l1)
        self.fc2 = nn.Linear(l1,l2)
        self.fcout = nn.Linear(l2, out)
        
    def forward(self, x):
        x = F.logsigmoid(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        x = self.fcout(x)
        return x
    
class Net_class(nn.Module):
    def __init__(self,input_size, l1, l2, l3, out):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = nn.Linear(input_size, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fcout = nn.Linear(l3, out)
        
    def forward(self, x):
        x = F.logsigmoid(self.fc1(x))
        x = F.logsigmoid(self.fc2(x))
        x = F.logsigmoid(self.fc3(x))
        x = F.softmax(self.fcout(x), dim = 1)
        return x
    

#%%
X_train, X_val, X_test, y_train, y_val, y_test, label_train, label_val, label_test = format_data(0.7, 0.7, 500)
#%%
net = Net(108, 75, 25, 3)
optimizer = optim.Adam(net.parameters(), lr = 0.00001)
loss_function = nn.MSELoss()
#%%
net = train_net(net, X_train, y_train, loss_function, optimizer, epochs = 3)
print()
print(eval_net(net,X_test, y_test, loss_function, plot = True))
