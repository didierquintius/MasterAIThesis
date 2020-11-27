# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:52:54 2020

@author: didie
"""
from DataSplit_functions import setNNFormat
import numpy as np
import torch
from Visualize_functions import plot_line

def test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Net, plot = False):
    electrodes, time_steps, trials = EEG_data_test.shape
    prediction_input = setNNFormat(EEG_data_test, electrodes)
    prediction_output = np.zeros((brain_areas, trials * time_steps)) 
    
    source_test = np.zeros((trials, brain_areas))
    for trial in range(trials):
        source_test[trial, sources_test[trial,:]] = 1
    
    for brain_area in range(brain_areas):
        output = NeuralNets[brain_area](prediction_input)
        prediction_output[brain_area, :] = output.view(-1).detach().numpy()
        if plot: plot_line([prediction_output[brain_area, :], np.repeat(source_test[:, brain_area], time_steps)], str(brain_area))
    time_series = prediction_output.reshape(brain_areas, trials, time_steps)
    
    classification_input = torch.Tensor(prediction_output.reshape((-1, 1, time_steps)))
    classification_output = CNN_Net(classification_input)
    
    source_prediction_values = classification_output.reshape((trials, brain_areas))
    source_prediction_values = source_prediction_values.detach().numpy()
    source_prediction = source_prediction_values > 0.5      
    
    predicted_active_areas = np.zeros((trials, 3))
    for trial in range(trials):
        predicted_active_areas[trial, :] = source_prediction_values[trial, :].argsort()[-3:]
    
    correct_area_predictions = (predicted_active_areas == sources_test).sum()
    area_accuracy = correct_area_predictions / predicted_active_areas.size
    
        
    true_positive = ((source_prediction == 1) & (source_test == 1)).sum() / (source_test == 1).sum()
    true_negative = ((source_prediction == 0) & (source_test == 0)).sum() / (source_test == 0).sum()
    
    mse = []
    reals = np.zeros((brain_areas, trials * time_steps))
    for trial, active_neurons in enumerate(sources_test):
        for i, neuron in enumerate(active_neurons):
            real = activity_test[i, :, trial]
            pred = time_series[neuron, trial, :]
            mse += [((real - pred)**2).mean()]
            reals[neuron, (trial * time_steps):((trial + 1) * time_steps)] = real
    
    if plot:
        for brain_area in range(brain_areas):
            plot_line([reals[brain_area,:], prediction_output[brain_area, :]], str(brain_area))  
                
    mse = np.array(mse)
    mean_mse = mse.mean()
    std_mse = mse.std()
       
    return area_accuracy, true_positive, true_negative, mean_mse, std_mse
        