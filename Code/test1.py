# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:40:23 2021

@author: Quintius
"""

from DataSimulation_functions import Balanced_EEG
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork_functions import NeuralNet, CNN
from Fit_functions import Validation, createBatches, updateNet

#%%

def train_brain_area(brain_area, param_values, val_perc = 0.1):
     
    def setNNFormat(data, nn_input):
        data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
        return data
    
    def filterActivityData(data, sources, brain_area = brain_area):
    
        relevant_data_indexes = [[neuron == brain_area for neuron in source] for source in sources]
        data = data.swapaxes(1,0)
        data = np.sum(data * np.array(relevant_data_indexes).T, axis = 1)
        data = setNNFormat(data, 1)
        return data
        
    # train neural network
    def train_PredictionModel(param_values, EEG_Data, source_activity):
        
        electrodes = EEG_Data.shape[1]
        
        # shuffle data
        shuffled_indexes = np.random.permutation(EEG_Data.shape[0])
        EEG_Data = EEG_Data[shuffled_indexes, :]
        source_activity = source_activity[shuffled_indexes, :]
        
        data_elements  = EEG_Data.shape[0] 
        
        #split data into validation and training sets
        X_val = EEG_Data[:int(data_elements* val_perc),:]
        y_val = source_activity[:int(data_elements* val_perc),:]
        X_train = EEG_Data[int(data_elements* val_perc):,:]
        y_train = source_activity[int(data_elements* val_perc):,:]
        
        # initialize Neuralnetwork and relevant parameters
        Net = NeuralNet(electrodes, [param_values['nodes_pred']], 1)
        optimizer = optim.Adam(Net.parameters(), lr =  param_values['learning_rate_pred'])
        loss_function = nn.MSELoss()
        Validator = Validation(param_values['max_val_amount_pred'], param_values["val_treshold_pred"], loss_function, X_val, y_val)
        batches = createBatches(np.arange(X_train.shape[0]), param_values['batch_sizes_pred'])
        for epoch in range(param_values['EPOCHS_pred']):
            for i, batch in enumerate(batches):        
                updateNet(Net, X_train, y_train, batch, loss_function, optimizer)        
                if (i % param_values['val_freq_pred']) == 0:
                    Net, STOP = Validator.update(Net)  
                    if STOP != "": 
                        train_performance = np.float(loss_function(Net(X_train), y_train))
                        return train_performance, Validator, Net, STOP
        
        train_performance = np.float(loss_function(Net(X_train), y_train))   
        return train_performance, Validator, Net, "Max_epochs"
    
    def determine_class(active_brain_areas, brain_area):
        trials = active_brain_areas.shape[0]
        class_trial = torch.zeros((trials, 1))
        for trial in range(trials):
            if brain_area in active_brain_areas[trial, :]:
                class_trial[trial] = 1
        return class_trial
    
    def trainClassificationModel(param_values, activity, class_trial):
        
        X_val = activity[:int(param_values['trials']* val_perc),:,:]
        y_val = class_trial[:int(param_values['trials']* val_perc),:]
        X_train = activity[int(param_values['trials']* val_perc):,:,:]
        y_train = class_trial[int(param_values['trials']* val_perc):,:]
        
        Net = CNN(param_values['time_steps'], [param_values['nodes_Conv_clas'], param_values['nodes_Dense_clas']],
                  [param_values['kernel_size']], [param_values['strides']])
    
        optimizer = optim.Adam(Net.parameters(), lr =  param_values['learning_rate_clas'])
    
        loss_function = nn.BCELoss()
        
        Validator = Validation(param_values['max_val_amount_clas'], param_values["val_treshold_clas"], loss_function, X_val, y_val)
        batches = createBatches(np.arange(X_train.shape[0]), param_values['batch_sizes_clas'])
        
        for epoch in range(param_values['EPOCHS_clas']):
            for i, batch in enumerate(batches):
                updateNet(Net, X_train, y_train, batch, loss_function, optimizer)                
                if (i % param_values['val_freq_clas']) == 0: 
                    Net, STOP = Validator.update(Net)                
                    if STOP != "": 
                        train_performance = np.float(loss_function(Net(X_train), y_train))
                        return Net, train_performance, Validator.val_losses, STOP
                        
        train_performance = np.float(loss_function(Net(X_train), y_train))   
          
        return Net, train_performance, Validator.val_losses, "Max Epochs"  
            
    
    # simulate data
    EEG_Data, active_brain_areas, noisy_brain_areas, source_activity, a = Balanced_EEG(param_values, brain_area)
    electrodes, time_steps, _ = EEG_Data.shape
    
    # format  prediction data
    EEG_Data = setNNFormat(EEG_Data, electrodes)
    source_activity = filterActivityData(source_activity, active_brain_areas)
    
    train_performance, Validator, Net, STOP = train_PredictionModel(param_values, EEG_Data, source_activity)
    
    # format classification data
    predicted_source_activity = Net(EEG_Data).reshape((-1, 1, time_steps))
    class_trial= determine_class(active_brain_areas, brain_area)
    
    CNN_Net, train_performance_clas, Validator_clas, STOP_clas = trainClassificationModel(param_values, predicted_source_activity, class_trial)
    
    return train_performance, train_performance_clas
    
    
    
    
    
