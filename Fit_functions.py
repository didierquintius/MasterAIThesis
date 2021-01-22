# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:48:20 2020

@author: didie
"""
import torch
import numpy as np
import os, random
from copy import deepcopy
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NeuralNetwork_functions import NeuralNet, CNN
from DataSplit_functions import prepareProjectionData, prepareClassificationData

def calc_loss(Net, X, y, loss_function):
    with torch.no_grad():
        loss = loss_function(Net(X), y)
    return np.float(loss)

class Validation():
    def __init__(self,max_val_amount, min_increment, loss_function, X, y):
        self.loss_function = loss_function
        self.val_counter = 0
        self.max_val_amount = max_val_amount
        self.min_val = np.inf
        self.min_Net = None
        self.val_losses = []
        self.X_val = X
        self.y_val = y
        self.min_increment = min_increment
        
    def update(self, Net):
        
        self.val_losses += [calc_loss(Net, self.X_val, self.y_val, self.loss_function)]
        if (self.min_val - self.val_losses[-1]  < self.min_increment):
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

def fitProjectionModel(EEG, sources,noisy_sources, activity, brain_area, architecture = [20], learning_rate = 1e-4, EPOCHS = 8,
                       batch_size = 50, max_val_amount = 5, val_freq = 5, preportion = 1, min_increment = 1e-6):
    X_train, y_train, X_val, y_val = prepareProjectionData(EEG, sources, noisy_sources, activity, brain_area)

    active_measurements , electrodes = X_train['active'].shape
    silent_measurements = X_train['silent'].shape[0]
    

    Net = NeuralNet(electrodes, architecture, 1)


    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)
    loss_function = nn.MSELoss()
    
    Validator = Validation(max_val_amount, min_increment, loss_function, X_val, y_val)

    balanced_chunks = createBatches(np.arange(silent_measurements), active_measurements * preportion)
    
    for epoch in range(EPOCHS):
        for chunk in balanced_chunks:

            X_train_chunk = torch.cat((X_train['active'], X_train['silent'][chunk, :]))
            y_train_chunk = torch.cat((y_train['active'], y_train['silent'][chunk, :]))
            shuffled_indexes = list(range(len(chunk)))
            random.shuffle(shuffled_indexes)
            X_train_chunk = X_train_chunk[shuffled_indexes, :]
            y_train_chunk = y_train_chunk[shuffled_indexes, :]
            batches = createBatches(shuffled_indexes, batch_size)

            
            for i, batch in enumerate(batches):
                
                updateNet(Net, X_train_chunk, y_train_chunk, batch, loss_function, optimizer)

                if (i % val_freq) == 0:
                    Net, STOP = Validator.update(Net)   
                    if STOP != "": 
                        X_train = torch.cat((X_train['active'], X_train['silent']))
                        y_train = torch.cat((y_train['active'], y_train['silent']))   
                        train_performance = np.float(loss_function(Net(X_train), y_train))
                        return Net, train_performance, Validator.val_losses, STOP


    X_train = torch.cat((X_train['active'], X_train['silent']))
    y_train = torch.cat((y_train['active'], y_train['silent']))            
    train_performance = np.float(loss_function(Net(X_train), y_train))              
    return Net, train_performance, Validator.val_losses, "Max Epochs"

def fitProjectionModels(EEG, sources,noisy_sources, activity, no_brain_areas, architecture = [20], learning_rate = 1e-4, EPOCHS = 20,
                       batch_size = 50, max_val_amount = 50, val_freq = 100, preportion = 1, min_increment = 1e-7):
    NeuralNets = {}
    train_performances = {}
    val_losses = {}
    STOPs = {} 
    for brain_area in tqdm(range(no_brain_areas)):
        NeuralNets[brain_area], train_performances[brain_area], val_losses[brain_area], STOPs[brain_area]= fitProjectionModel(EEG, sources, noisy_sources, activity, brain_area, architecture, learning_rate , EPOCHS, batch_size , max_val_amount, val_freq, preportion, min_increment)


    train_performance = np.array(list(train_performances.values()))
    STOPs = list(STOPs.values())
    return NeuralNets, train_performance, val_losses, STOPs
                           
def fitClassificationModel(EEG_data_trainval, sources_trainval, brain_area, NeuralNet, nodes = [10, 15], kernel_sizes = [5], strides = [3],
                          learning_rate = 1e-4, EPOCHS = 100, batch_size = 50,
                          max_val_amount = 200, val_freq = 5, preportion = 1, min_increment = 1e-9):
    X, y = prepareClassificationData(EEG_data_trainval,sources_trainval, brain_area, NeuralNet)
    time_steps = X["val"].shape[2]
    idle_trials = X["train"]["idle"].shape[0]
    active_trials = X["train"]["active"].shape[0]
    
    Net = CNN(time_steps, nodes, kernel_sizes, strides)

    optimizer = optim.Adam(Net.parameters(), lr =  learning_rate)

    loss_function = nn.BCELoss()
    
    Validator = Validation(max_val_amount, min_increment, loss_function, X["val"], y["val"])
    balanced_idle_indexes = createBatches(np.arange(idle_trials), active_trials * preportion)
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

def fitClassificationModels(EEG_data_trainval,sources_trainval,brain_areas,NeuralNets, nodes = [10, 15], kernel_sizes = [5], strides = [3],
                        learning_rate = 1e-4, EPOCHS = 100, batch_size = 50,
                        max_val_amount = 25, val_freq = 4, preportion = 1, min_increment = 1e-9, prev_nets = None):
    CNNs = {}
    train_performances = {}
    val_losses = {}
    STOPs = {}
    no_brain_areas = len(NeuralNets)
    for brain_area in tqdm(range(no_brain_areas)):
        hi = fitClassificationModel(EEG_data_trainval,sources_trainval,brain_area, NeuralNets[brain_area], nodes,kernel_sizes,strides,learning_rate,EPOCHS,batch_size,max_val_amount,val_freq, preportion, min_increment)
        CNNs[brain_area], train_performances[brain_area], val_losses[brain_area], STOPs[brain_area] = hi

    train_performance = np.array(list(train_performances.values()))
    STOPs = list(STOPs.values())
    return CNNs, train_performance, val_losses, STOPs