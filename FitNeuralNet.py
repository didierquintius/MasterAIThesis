# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:35:38 2020

@author: didie
"""

import torch, os
from copy import copy
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork import NeuralNet

def calc_loss(net, X, y, loss_function, plot = False, n_plots = 1):
    with torch.no_grad():
        real = y
        pred = net(X)
        loss = loss_function(pred, real)
    
    if plot:
        for i in range(1000, y.shape[0],int((y.shape[0] - 1000) / n_plots)):
            for j in range(3):
                plt.figure()
                plt.plot(range(100),real[i:(i+ 100),j],"r", 
                         range(100),pred[i:(i+ 100),j],"b")            
                
    return np.float(loss)

def do_val_check(Net, X, y, min_val, val_counter, loss, loss_function, max_val_amount,
                 min_Net):
    
    loss = loss + [calc_loss(Net, X["val"], y["val"], loss_function)]
    
    if loss[-1] > min_val:
        val_counter +=1
        STOP = val_counter > max_val_amount
        Net = copy(min_Net)
    else:
        min_Net = copy(Net)
        min_val = loss[-1]
        val_counter = 0
        STOP = False
    
    return min_val, val_counter, loss, STOP, Net, min_Net
    
    
#%%
def FitNeuralNet(Net, X, y, learning_rate, EPOCHS, chunk, max_val_amount, val_freq):
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    val_counter = 0
    min_val = np.inf
    min_Net = copy(Net)
    losses = []
    for epoch in range(EPOCHS):                
        for i in range(0, y["train"].shape[0], chunk):
            Net.zero_grad()
            outputs = Net(X["train"][i:(i + chunk),:])
            loss = loss_function(outputs, y["train"][i:(i + chunk), :])
            loss.backward()
            optimizer.step()
            
            if (i % (val_freq * chunk)) == 0:
                min_val, val_counter, losses, STOP, Net, min_Net = do_val_check(Net, X, y, min_val, 
                                                            val_counter, losses,
                                                            loss_function, max_val_amount,
                                                            min_Net)
            if STOP: break
        if STOP:break
    
    result = {"Net": Net,
              "validation_losses": losses,
              "train_performance": np.float(loss),
              "test_performance": np.float(loss_function(Net(X["test"]), y["test"])),
              "val_performance": np.float(loss_function(Net(X["val"]), y["val"]))}
    return result

def fit_NeuralNets(X, y, no_brain_areas, architecture = [20], learning_rate = 5e-4, EPOCHS = 20, chunk = 50,
                   max_val_amount = 40, val_freq = 5):
    NeuralNets = {}
    for brain_area in range(no_brain_areas):
        Net = NeuralNet(X[str(brain_area)]["train"].shape[1], architecture,
                        output = y[str(brain_area)]["train"].shape[1])
        
        result = FitNeuralNet(Net, X[str(brain_area)], y[str(brain_area)], learning_rate, EPOCHS, chunk, 
                              max_val_amount, val_freq)
            
        print(result["test_performance"])
        NeuralNets[brain_area] = result["Net"]
    return NeuralNets