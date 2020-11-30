# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:52:54 2020

@author: didie
"""

import numpy as np
from Visualize_functions import plot_line
from DataSplit_functions import setNNFormat

def most_likely_sources(data, source_test, n = 3):
    trials, brain_areas  = data.shape
    predicted_active_areas = np.zeros((trials, brain_areas))
    for trial in range(trials):
        sorted_trial_data = np.sort(data[trial, :])
        predicted_active_areas[trial, :] = data[trial, :] >= sorted_trial_data[-3]   
    
    wrong_predictions = ((trials * brain_areas)  - np.equal(predicted_active_areas, source_test).sum()) / 2 
    area_accuracy = 1 - wrong_predictions / (trials * 3) 
    
    binary_prediction_data = data > 0.5  
        
    true_positive = ((binary_prediction_data == 1) & (source_test == 1)).sum() / (source_test == 1).sum()
    true_negative = ((binary_prediction_data== 0) & (source_test == 0)).sum() / (source_test == 0).sum()
    return area_accuracy, true_negative, true_positive

def Model(EEG, NeuralNets, CNNs):
    electrodes, time_steps, trials = EEG.shape
    prediction_input = setNNFormat(EEG, electrodes)
    brain_areas = len(NeuralNets)
    
    time_series = np.zeros((brain_areas, trials, time_steps))
    source_prediction_values = np.zeros((trials, brain_areas))
    
    for brain_area, NeuralNet in NeuralNets.items():
        output = NeuralNet(prediction_input)
        time_series[brain_area, :, :] = output.view(-1, trials, time_steps).detach().numpy()
    
        classification_input = output.view(-1, 1, time_steps)
        classification_output = CNNs[brain_area](classification_input)
        source_prediction_values[:, brain_area] = classification_output.view(-1).detach().numpy()
    
    return time_series, source_prediction_values


    
def test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Nets, plot = False):
    
    
    electrodes, time_steps, trials = EEG_data_test.shape
    
    def plot_performance(time_series, source_prediction_values, reals, source_test):
        for brain_area in range(brain_areas):
            time_serie = time_series[brain_area, :, :].reshape(-1)
            plot_line([time_serie, np.repeat(source_test[:, brain_area], time_steps), np.repeat(source_prediction_values[:, brain_area],time_steps)], str(brain_area))
            plot_line([reals[brain_area,:], time_serie], "fit" + str(brain_area))  
        
    source_test = np.zeros((trials, brain_areas))
    for trial in range(trials):
        source_test[trial, sources_test[trial,:]] = 1
        
    time_series, source_prediction_values = Model(EEG_data_test, NeuralNets, CNN_Nets)     
    
    mse = []
    reals = np.zeros((brain_areas, trials * time_steps))
    for trial, active_neurons in enumerate(sources_test):
        for i, neuron in enumerate(active_neurons):
            real = activity_test[i, :, trial]
            pred = time_series[neuron, trial, :]
            mse += [((real - pred)**2).mean()]
            reals[neuron, (trial * time_steps):((trial + 1) * time_steps)] = real
            
    area_accuracy, true_negative, true_positive = most_likely_sources(source_prediction_values, source_test)
    
    if plot: plot_performance(time_series, source_prediction_values, reals, source_test)
                
    mse = np.array(mse)
    mean_mse = mse.mean()
    std_mse = mse.std()
       
    return area_accuracy, true_positive, true_negative, mean_mse, std_mse, time_series
        