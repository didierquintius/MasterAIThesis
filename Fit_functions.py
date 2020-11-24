# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:48:20 2020

@author: didie
"""
import torch
import numpy as np
import pandas as pd
import torch, os, random
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork_functions import NeuralNet, CNN
from DataSplit_functions import prepareProjectionData

def calc_loss(Net, X, y, loss_function):
    with torch.no_grad():
        loss = loss_function(Net(X), y)
    return np.float(loss)

class Validation():
    def __init__(self,max_val_amount, loss_function, X, y):
        self.loss_function = loss_function
        self.val_counter = 0
        self.max_val_amount = max_val_amount
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        self.X_val = X
        self.y_val = y
        
    def update(self, Net):
        
        self.val_losses += [calc_loss(Net, self.X_val, self.y_val, self.loss_function)]
        if self.val_losses[-1] > self.min_val:
            self.val_counter +=1
            if self.val_counter > self.max_val_amount:
                return self.min_Net, "Max val increment"
        else:
            self.min_Net = deepcopy(Net)
            self.min_val = self.val_losses[-1]
            self.val_counter = 0
        
        return Net, ""
    
    def reset(self):
        self.val_counter = 0
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        
def createBatches(listi, n):
    return [listi[i:(i + n)] for i in range(0, len(listi), n)]

def updateNet(Net, X, y, batch, loss_function, optimizer):
    batch_X = X[batch, :]
    batch_y = y[batch, :]
    outputs = Net(batch_X)
    loss = loss_function(outputs, batch_y)
    loss.backward()
    optimizer.step()
    return loss

def fitProjectionModel(EEG, sources,noisy_sources, activity, brain_area, architecture = [20], learning_rate = 5e-4, EPOCHS = 20,
                       batch_size = 50, max_val_amount = 40, val_freq = 5):
    
    X_train, y_train, X_val, y_val = prepareProjectionData(EEG, sources, noisy_sources, activity, brain_area)

    measurements , electrodes = X_train.shape
    
    Net = NeuralNet(electrodes, architecture, 1)
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    
    Validator = Validation(max_val_amount, loss_function, X_val, y_val)
    batches = createBatches(np.arange(measurements), batch_size)
    
    for epoch in range(EPOCHS):
        for i, batch in enumerate(batches):      
            updateNet(Net, X_train, y_train, batch, loss_function, optimizer)
            
            if (i % val_freq) == 0: 
                Net, STOP = Validator.update(Net)                
                if STOP != "": 
                    train_performance = np.float(loss_function(Net(X_train), y_train))
                    return Net, train_performance, Validator.val_losses, STOP
                
    train_performance = np.float(loss_function(Net(X_train), y_train))              
    return Net, train_performance, Validator.val_losses, "Max Epochs" 

def fitProjectionModels(EEG, sources,noisy_sources, activity, no_brain_areas, architecture = [20], learning_rate = 5e-4, EPOCHS = 20,
                       batch_size = 50, max_val_amount = 40, val_freq = 5):
    NeuralNets = {}
    train_performances = []
    val_losses = {}
    STOPs = [] 
    
    for brain_area in range(no_brain_areas):
        NeuralNets[brain_area], train_performance, val_losses[brain_area], STOP = fitProjectionModel(EEG, sources, noisy_sources, activity, brain_area, architecture, learning_rate , EPOCHS, batch_size , max_val_amount, val_freq)
        train_performances += [train_performance]
        STOPs += [STOP]

    train_performance = np.array(train_performances)
    mean_train = train_performance.mean()
    std_train = train_performance.std()
    return NeuralNets, mean_train, std_train, val_losses, STOPs
                           
def fitClassificationModel(X, y, nodes = [10, 15], kernel_sizes = [5], strides = [3],
                          learning_rate = 5e-4, EPOCHS = 100, batch_size = 50,
                          max_val_amount = 400, val_freq = 10):
    
    time_steps = X["val"].shape[2]
    idle_trials = X["train"]["idle"].shape[0]
    active_trials = X["train"]["active"].shape[0]
    
    Net = CNN(time_steps, nodes, kernel_sizes, strides)
    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.BCELoss()
    
    Validator = Validation(max_val_amount, loss_function, X["val"], y["val"])
    balanced_idle_indexes = createBatches(np.arange(idle_trials), active_trials)
    for epoch in range(EPOCHS):
        for balanced_idle_index in balanced_idle_indexes:
            X_balanced = torch.cat((X["train"]["active"], X["train"]["idle"][balanced_idle_index,:,:]))
            y_balanced = torch.cat((y["train"]["active"], y["train"]["idle"][balanced_idle_index,:]))
            shuffled_indexes = np.arange(X_balanced.shape[0])
            random.shuffle(shuffled_indexes)
            X_balanced = X_balanced[shuffled_indexes,:,:]
            y_balanced = y_balanced[shuffled_indexes,:]
            
            batches = createBatches(np.arange(len(balanced_idle_index)), batch_size)
            for i, batch in enumerate(batches):
                updateNet(Net, X_balanced, y_balanced, batch, loss_function, optimizer)
                
                if (i % val_freq) == 0: 
                    Net, STOP = Validator.update(Net)                
                    if STOP != "": 
                        train_performance = np.float(loss_function(Net(X_balanced), y_balanced))
                        return Net, train_performance, Validator.val_losses, STOP
                    
    train_performance = np.float(loss_function(Net(X_balanced), y_balanced))              
    return Net, train_performance, Validator.val_losses, "Max Epochs"                 
    