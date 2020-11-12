# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:35:38 2020

@author: didie
"""

import torch, os, random
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork_functions import NeuralNet, CNN

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
            #Net = deepcopy(self.min_Net)
            if self.val_counter > self.max_val_amount:
                return Net, True
        else:
            #self.min_Net = deepcopy(Net)
            self.min_val = self.val_losses[-1]
            self.val_counter = 0
        
        return Net, False
    
def createChunks(listi, n):
    return [listi[i:(i + n)] for i in range(0, len(listi), n)]
        

def result_dict(Net, X, y, val_losses, loss_function):
    
    result = {"Net": Net,
          "validation_losses": val_losses,
          "test_prediction":Net(X["test"]).detach().transpose(0,1)[0],
          "train_performance": np.float(loss_function(Net(X["train"]), y["train"])),
          "test_performance": np.float(loss_function(Net(X["test"]), y["test"])),
          "val_performance": np.float(loss_function(Net(X["val"]), y["val"]))}
    return result

def updateNet(Net, X, y, loss_function, optimizer):
    outputs = Net(X)
    loss = loss_function(outputs, y)
    loss.backward()
    optimizer.step()
    return loss
    
def FitNeuralNet(Net, X, y, learning_rate, EPOCHS, chunk_size, max_val_amount, val_freq):
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    Validation = ValidationChecker(max_val_amount, loss_function, X, y)
    chunks = createChunks(range(y["train"].shape[0]), chunk_size)    
    for epoch in range(EPOCHS):           
        for i, chunk in enumerate(chunks):
            chunk_X = X["train"][chunk,:]
            chunk_y =  y["train"][chunk, :]
            updateNet(Net, chunk_X, chunk_y, loss_function, optimizer)
            
            if (i % val_freq) == 0: 
                Net, STOP = Validation.check(Net)                
                if STOP: result_dict(Net, X, y, Validation.val_losses, loss_function)

    return result_dict(Net, X, y, Validation.val_losses, loss_function)

def fit_NeuralNets(X, y, architecture = [20], learning_rate = 5e-4, EPOCHS = 20, chunk = 50,
                   max_val_amount = 40, val_freq = 5):
    NeuralNets = {}
    results = {}
    brain_areas = len(X)
    
    electrodes = X["0"]["train"].shape[1]
    active_areas = y["0"]["train"].shape[1]
    
    for brain_area in range(brain_areas):
        Net = NeuralNet(electrodes, architecture, output = active_areas)
        
        result = FitNeuralNet(Net, X[str(brain_area)], y[str(brain_area)], 
                              learning_rate, EPOCHS, chunk, max_val_amount, val_freq)
            
        NeuralNets[brain_area] = result["Net"]
        del result["Net"]
        results[brain_area]  = result
        
    return NeuralNets, results

def createAndSplitData(EEG_data, source_matrix, timesteps, NeuralNets, 
                       brainarea, train_perc = 0.7, val_perc = 0.1):
    
    predictions = NeuralNets[brainarea](EEG_data).view((-1, timesteps))

    data_dict = {"X":{},"y":{}}
    original = {"X": predictions, "y": source_matrix[:, brainarea]}
    
    random.seed(0)
    active_trials = np.where(original["y"] == 1)[0].tolist()    
    train_active = random.sample(active_trials, int(train_perc * len(active_trials)))
    remaining_trials = list(set(active_trials) -set(train_active))
    val_active = random.sample(remaining_trials, int(val_perc * len(active_trials)))
    test_active = list(set(remaining_trials) - set(val_active))
    
    idle_trials = np.where(original["y"] == 0)[0].tolist()
    val_idle = random.sample(idle_trials, len(val_active))
    remaining_trials = list(set(idle_trials) -set(val_idle))
    test_idle = random.sample(remaining_trials, len(test_active))
    train_idle = list(set(remaining_trials) - set(test_idle))
    
    sections = {"val":val_active + val_idle,  "test":test_active + test_idle}
    for data_name, data in data_dict.items():
        for section, index in sections.items():
            if data_name == "y": data[section] = original[data_name][index].view(-1,1)
            else: data[section] = original[data_name][index,:].view(-1, 1, timesteps)
            
    data_dict["X"]["train"] = {}
    data_dict["X"]["train"]["active"] = predictions[train_active,:].view(-1, 1, timesteps)
    data_dict["X"]["train"]["idle"] = predictions[train_idle,:].view(-1, 1, timesteps)

    data_dict["y"]["train"] = {}
    data_dict["y"]["train"]["active"] = original["y"][train_active]
    data_dict["y"]["train"]["idle"] = original["y"][train_idle]
    
    X, y = data_dict.values()
    no_blocks = int(source_matrix.shape[0] / 3 - 1)
    blocks = createChunks(np.arange(X["train"]["idle"].shape[0]), no_blocks)
    return X, y, blocks
#%%
def class_results(Net,Validation, X, y, loss_function):
    pred = Net(X["test"]).detach().transpose(0,1)[0]
    y_test = y["test"].view(-1)
    result = {"Net": Net,
              "MSE_test": np.float(loss_function(Net(X["test"]), y["test"])),
              "TP_test": sum((pred > 0.5) & (y_test == 1)).true_divide(sum(y_test == 1)),
              "TN_test": sum((pred < 0.5) & (y_test == 0)).true_divide(sum(y_test == 0)),
              "val_losses": Validation.val_losses,
              "prediction": pred,
              "X_test": X["test"],
              "y_test": y_test,
              "STOP":"Validation"}
    
    return result
        
def fitBalancedNeuralNet(Net, NeuralNets, EEG_data, source_matrix, brain_area, timesteps,
                         learning_rate, EPOCHS, chunk_size, max_val_amount, val_freq):
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    X, y, blocks =  createAndSplitData(EEG_data, source_matrix, timesteps,
                                                 NeuralNets, brain_area)
    Validation = ValidationChecker(max_val_amount, loss_function, X, y)
    for epoch in range(EPOCHS):
        for block in blocks:
            block_X = torch.cat((X["train"]["active"], X["train"]["idle"][block,:]), 0)
            block_y = torch.cat((y["train"]["active"], y["train"]["idle"][block]), 0)
            shuffeled_indexes = np.arange(block_X.shape[0])
            random.shuffle(shuffeled_indexes)
            block_X = block_X[shuffeled_indexes,:,:]
            block_y = block_y[shuffeled_indexes]
            chunks = createChunks(np.arange(block_y.shape[0]), chunk_size)
               
            for i, chunk in enumerate(chunks):
                chunk_X = block_X[chunk, :]
                chunk_y = torch.Tensor(block_y[chunk].view(-1, 1))
                outputs = Net(chunk_X)
                loss = loss_function(outputs, chunk_y)
                loss.backward()
                optimizer.step()
                
                if (i % val_freq) == 0: 
                    Net, STOP = Validation.check(Net)  
                    if STOP: 
                        result = class_results(Net,Validation, X, y, loss_function)
                        return result
                                        
    result = class_results(Net,Validation, X, y, loss_function)
    return result
#%%

def fitBalancedNeuralNets(NeuralNets, EEG_data, source_matrix, timesteps,
                          architecture = ([10, 15, 5], [3, 3], [2,2]),
                          learning_rate = 5e-4, EPOCHS = 20, chunk = 50,
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
                
                