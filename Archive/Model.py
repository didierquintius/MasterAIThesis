# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:47:09 2020

@author: didie
"""
import os, pickle, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
#%%
mask = pickle.load(open("../Data/mask_matrix.pkl", "rb"))
mask = torch.Tensor(mask)

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

def train_net_RNN(net, X, y, X_val, y_val, loss_function, optimizer, lag, batch_size = 20,
              epochs = 1, val = True, limit = 50, skips = 4):

    prev = torch.Tensor(np.zeros((batch_size, lag)))
    for epoch in range(epochs):
        for i in tqdm(range(0, X.shape[0], batch_size)):                           
            
            net.zero_grad()
            batch_X = torch.cat((X[i,:], prev), dim = 1)
            outputs = net(batch_X)
            loss = loss_function(outputs, y[i,:])
            loss.backward(retain_graph = True)
            optimizer.step()
            prev[:-3] = prev[3:]
            prev[0, -3:] = outputs
    
    return net
def train_net(net, X, y, X_val, y_val, loss_function, optimizer, lag, batch_size = 20,
              epochs = 1, val = True, limit = 50, skips = 4):
    min_val = np.Inf
    vals = np.zeros(int(X.shape[0]/batch_size/4 * epochs) + 1)
    counter = 0
    j = 0
    for epoch in range(epochs):
        for i in tqdm(range(0, X.shape[0], batch_size)):
            if val & (i % (batch_size * skips) == 0):
                val_check = eval_net(net, X_val, y_val,loss_function)                
                vals[j] = val_check
                j += 1
                if val_check > min_val:
                    counter += 1
                    if counter > limit:
                        print(val_check)
                        break                        
                else:
                    min_val = val_check
                    final_net = net
                    counter = 0
                
            batch_X = X[i : (i + batch_size), :]
            batch_y = y[i : (i + batch_size), :]
            
            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
    if val: net = final_net
    return net, vals

def eval_net_RNN(net, X, y, loss_function, lag, plot_inter = 5, plot = False):
    X = X
    y = y
    prev = np.ones((1,lag))
    loss = 0
    for i in range(X.shape[0]):
        batch_X = torch.cat((X[i, :], prev))
        pred = net(batch_X)
        loss += loss_function(y[i, :], pred)
        prev[:-3] = prev[3:]
        prev[0, -3:] = pred
    if plot:
        for j in range(0, X.shape[0], int(X.shape[0] / plot_inter)):
            for k in range(3):
                plt.plot(range(200), y[j:(j+200),k],  "r-", 
                         range(200), pred[j:(j+200),k].detach().numpy(), "b-")
                plt.show()
    return loss / X.shape[0]

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

def zero_grad(self, grad_input, grad_output):
    grad_input_2 = grad_input[2] * self.mask
    return (grad_input[0], grad_input[1], grad_input_2)

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super(MaskedLinear, self).__init__(in_features, out_features)
        self.weight = torch.nn.Parameter(self.weight * mask.transpose(0,1))  # to zero it out first
        self.mask = mask
        self.handle = self.register_backward_hook(zero_grad)
        
class Net(nn.Module):
    def __init__(self, input_size, l1, l2, out, mask):
        super().__init__() # just run the init of parent class (nn.Module)
        self.fc1 = MaskedLinear(input_size,l1, mask)
        self.fc2 = nn.Linear(l1,l2)
        self.fcout = nn.Linear(l2, out)
        
    def forward(self, x):
        x = F.logsigmoid(self.fc1(x))      
        x = F.logsigmoid(self.fc2(x))
        x = self.fcout(x)
        return x
#%%
X_train, X_val, X_test, y_train, y_val, y_test, label_train, label_val, label_test = format_data(0.7, 0.7, 500)
#%%
net = Net(108, 50, 10, 3, mask)
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()
#%%
net, vals = train_net(net, X_train, y_train, X_val, y_val, loss_function,
                      optimizer, lag = 9, epochs = 3, limit = 80, batch_size = 50,
                      val = True, skips = 10)
print()
print(eval_net(net,X_test, y_test, loss_function, plot = True))
#%%
plt.figure()
plt.yscale("log")
plt.plot(vals[vals > 0])
plt.show()
#%%
def format_source(y1, lag):
    X = np.zeros((y1.shape[0] - lag - 1, 3 * lag))
    y = np.zeros((y1.shape[0] - lag - 1, 3 * lag))
    y = y1[lag:]
    for i in range(lag):
        X[:, i:(i + 3)] = y1[i:-lag + i - 1,:]
    return X, y