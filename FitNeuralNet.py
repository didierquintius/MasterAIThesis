# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:35:38 2020

@author: didie
"""

import torch, os, random
from copy import copy
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork import NeuralNet, CNN

def calc_loss(Net, X, y, loss_function):
    with torch.no_grad():
        loss = loss_function(Net(X), y)
    return np.float(loss)

class ValidationChecker():
    def __init__(self,max_val_amount, loss_function, X, y):
        self.loss_function = loss_function
        self.val_counter = 0
        self.max_val_amount = max_val_amount
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        self.X_val = X["val"]
        self.y_val = y["val"]
        
    def check(self, Net):
        
        self.val_losses += [calc_loss(Net, self.X_val, self.y_val, self.loss_function)]
        if self.val_losses[-1] > self.min_val:
            self.val_counter +=1
            if self.val_counter > self.max_val_amount:
                return copy(self.min_Net), True
        else:
            self.min_Net = copy(Net)
            self.min_val = self.val_losses[-1]
            self.val_counter = 0
        
        return Net, False
    
def createChunks(listi, n):
    return [listi[i:(i + n)] for i in range(0, len(listi), n)]
        
#%%
def result_dict(Net, X, y, val_losses, loss_function):
    
    result = {"Net": Net,
          "validation_losses": val_losses,
          "train_performance": np.float(loss_function(Net(X["train"]), y["train"])),
          "test_performance": np.float(loss_function(Net(X["test"]), y["test"])),
          "val_performance": np.float(loss_function(Net(X["val"]), y["val"]))}
    return result
    
def FitNeuralNet(Net, X, y, learning_rate, EPOCHS, chunk_size, max_val_amount, val_freq):
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    Validation = ValidationChecker(max_val_amount, loss_function, X, y)
    for epoch in range(EPOCHS):
        chunks = createChunks(range(y["train"].shape[0]), chunk_size)               
        for i, chunk in enumerate(chunks):
            chunk_X = X["train"][chunk,:]
            chunk_y =  y["train"][chunk, :]
            updateNet(Net, chunk_X, chunk_y, loss_function, optimizer)
            
            if (i % val_freq) == 0: 
                Net, STOP = Validation.check(Net)                
                if STOP: result_dict(Net, X, y, Validation.val_losses, loss_function)

    return result_dict(Net, X, y, Validation.val_losses, loss_function)

def createAndSplitData(EEG_data, source_matrix, timesteps, NeuralNets, 
                       brainarea, train_perc = 0.7, val_perc = 0.1):
    
    predictions = NeuralNets[brainarea](EEG_data).view((-1, timesteps))
    active = np.where(source_matrix[:, brainarea] == 1)[0]
    idle   = np.where(source_matrix[:, brainarea] == 0)[0]
    percs = {"train":(0, train_perc),"val":(train_perc, val_perc),"test": (val_perc, 1)}
    activity = {"active": active, "idle": idle}
    data_dict = {"X":{},"y":{}}
    original = {"X": predictions, "y": source_matrix[:, brainarea]}
    for data_name, data in data_dict.items():
        for section, perc in percs.items():
            for state_name, state in activity.items():
                data[section] = {}
                index = np.arange(int(perc[0] * state.shape[0]),int(perc[1] * state.shape[0]))
                if data_name == "y": data[section][state_name] = original[data_name][state][index]
                else: data[section][state_name] = original[data_name][state,:][index]
            
    X, y  = data_dict.values()
    no_blocks = int(source_matrix.shape[0] / 3 - 1)
    blocks = createChunks(np.arange(len(idle)), no_blocks)
    return X, y, blocks, activity
#%%
def fitBalancedNeuralNet(Net, NeuralNets, EEG_data, source_matrix, brain_area, timesteps,
                         learning_rate, EPOCHS, chunk_size, max_val_amount, val_freq):
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    X, y, blocks, activity =  createAndSplitData(EEG_data, source_matrix, timesteps,
                                                 NeuralNets, brain_area)
    Validation = ValidationChecker(max_val_amount, loss_function, X, y)
    for epoch in range(EPOCHS):
        for block in tqdm(blocks):
            indexes = np.append(activity["active"], activity["idle"][block])
            random.shuffle(indexes)
            block_X = X["train"][indexes, :]
            block_y = y["train"][indexes]
            chunks = createChunks(np.arange(block_y.shape[0]), chunk_size)
               
            for i, chunk in enumerate(chunks):
                chunk_X = block_X[chunk, :].view(-1, 1, block_X.shape[1])
                chunk_y = torch.Tensor(block_y[chunk].view(-1,1))
                
                updateNet(Net, chunk_X, chunk_y, loss_function, optimizer)
                
                if (i % val_freq) == 0: 
                    Net, STOP = Validation.check(Net)                
                    if STOP: result_dict(Net, X, y, Validation.val_losses, loss_function)

    return result_dict(Net, X, y, Validation.val_losses, loss_function)
#%%
def fit_NeuralNets(X, y, no_brain_areas, architecture = [20], learning_rate = 5e-4, EPOCHS = 20, chunk = 50,
                   max_val_amount = 40, val_freq = 5):
    NeuralNets = {}
    for brain_area in range(no_brain_areas):
        Net = NeuralNet(X[str(brain_area)]["train"].shape[1], architecture,
                        output = y[str(brain_area)]["train"].shape[1])
        
        result = FitNeuralNet(Net, X[str(brain_area)], y[str(brain_area)], 
                              learning_rate, EPOCHS, chunk, max_val_amount, val_freq)
            
        print(result["test_performance"])
        NeuralNets[brain_area] = result["Net"]
        
    return NeuralNets

def updateNet(Net, X, y, loss_function, optimizer):
    outputs = Net(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()
    return loss
#%%

def fitBalancedNeuralNets(NeuralNets, EEG_data, source_matrix, timesteps,
                          architecture = ([10, 15, 5], [3, 3], [2,2]),
                          learning_rate = 5e-4, EPOCHS = 5, chunk = 50,
                          max_val_amount = 40, val_freq = 5, chunk_size = 100):
    
    brainareas = len(NeuralNets)
    results = {}
    for brainarea in range(brainareas):
        Net = CNN(timesteps, architecture[0], architecture[1], architecture[2])
        
        result = fitBalancedNeuralNet(Net, NeuralNets, EEG_data, source_matrix, brainarea,
                                      timesteps, learning_rate, EPOCHS, 
                                      chunk_size, max_val_amount, val_freq)
        results[brainarea] = result
            
    return results
                
                