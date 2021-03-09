# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 08:40:23 2021

@author: Quintius
"""

from DataSimulation_functions import Balanced_EEG
import torch
from time import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from NeuralNetwork_functions import NeuralNet, CNN
from Fit_functions import Validation, createBatches, updateNet
from Visualize_functions import plot_line

#%%

def train_brain_area(brain_area, param_values, val_perc = 0.1, plot = False):
     
    def setNNFormat(data, nn_input):
        data = torch.Tensor(data.reshape((nn_input, -1),order = "F").transpose())
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
    EEG_Data, activity, source_trials, noisy_trials = Balanced_EEG(param_values, brain_area)
    electrodes, time_steps, _ = EEG_Data.shape
    

    # format  prediction data
    EEG_Data = setNNFormat(EEG_Data, electrodes)
    activity = setNNFormat(activity, 1)
    source_trials = setNNFormat(source_trials, 1)
    train_performance, Validator, Net, STOP = train_PredictionModel(param_values, EEG_Data, activity)
    
    # format classification data
    predicted_source_activity = Net(EEG_Data).reshape((-1, 1, time_steps))
    
    CNN_Net, train_performance_clas, Validator_clas, STOP_clas = trainClassificationModel(param_values, predicted_source_activity, source_trials)
    
    
    EEG_Data, activity, source_trials, _  = Balanced_EEG(param_values, brain_area, seed = 1)
    
    
    EEG_Data = setNNFormat(EEG_Data, electrodes)
    predicted_source_activity = Net(EEG_Data)
    predicted_class = CNN_Net(predicted_source_activity.reshape(-1, 1, time_steps)).detach().numpy().reshape(-1)
    
    
    predicted_source_activity = predicted_source_activity.detach().numpy().reshape(-1, time_steps)
    activity = activity.transpose()
    mse_pred = ((activity - predicted_source_activity)**2).mean()
    mse_clas = ((source_trials - predicted_class)**2).mean()

    truepositive_clas = (predicted_class[source_trials == 1] >= 0.5).mean()
    truenegative_clas = (predicted_class[source_trials == 0] < 0.5).mean()

    if plot:
        activity = activity.reshape((-1))
        predicted_source_activity = predicted_source_activity.reshape((-1))
        plot_line([activity, predicted_source_activity],['Original','Predicted'], title = 'Prediction_Performance')
        source_trials = np.repeat(source_trials, time_steps).reshape((-1))
        predicted_class = np.repeat(predicted_class, time_steps).reshape((-1))
        plot_line([predicted_source_activity, source_trials, predicted_class],['Predicted Activity', 'Original State', 'Predicted State'], dash = ['solid', 'dash', 'dot'],title = 'Classification Performance')
   
    return (mse_pred, mse_clas, truepositive_clas, truenegative_clas, STOP, STOP_clas)
    
    
    
    
    
