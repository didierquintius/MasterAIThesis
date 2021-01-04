# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 23:52:54 2020

@author: didie
"""

import numpy as np
from Visualize_functions import plot_line
from DataSplit_functions import setNNFormat
from eLoreta import eLoretaModel

def most_likely_sources(data, source_test, n = 3):
    trials, brain_areas  = data.shape
    predicted_active_areas = np.zeros((trials, brain_areas))
    for trial in range(trials):
        sorted_trial_data = np.sort(data[trial, :])
        predicted_active_areas[trial, :] = data[trial, :] >= sorted_trial_data[-3]   
    
    wrong_predictions = ((trials * brain_areas)  - np.equal(predicted_active_areas, source_test).sum()) / 2 
    area_accuracy = 1 - wrong_predictions / (trials * 3) 
    
    binary_prediction_data = data > 0.5  
    
    true_positive_data = (binary_prediction_data == 1) & (source_test == 1)
    true_negative_data = (binary_prediction_data == 0) & (source_test == 0)
    
    true_positive_area = true_positive_data.sum(axis = 0) / (source_test == 1).sum(axis = 0)
    true_negative_area = true_negative_data.sum(axis = 0) / (source_test == 0).sum(axis = 0)
    
    true_positive = true_positive_data.sum() / (source_test == 1).sum()
    true_negative = (true_negative_data).sum() / (source_test == 0).sum()    
   
    return area_accuracy, true_negative, true_positive, true_positive_area, true_negative_area

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

def varModel(data, axis):

    data = data.std(axis = axis)
    trials = data.shape[0]
    predicted_active_areas = np.zeros(data.shape)
    for trial in range(trials):
        sorted_trial_data = np.sort(data[trial, :])
        predicted_active_areas[trial, :] = (data[trial, :] >= sorted_trial_data[-3]) * 1

    return predicted_active_areas

    
def test_performance(EEG_data_test, sources_test, activity_test, brain_areas, NeuralNets, CNN_Nets,gamma = 100, plot = False):
    
    
    electrodes, time_steps, trials = EEG_data_test.shape
    
    def plot_performance(time_series, source_prediction_values, reals, source_test):
        for brain_area in [46, 51]:
            time_serie = time_series[brain_area, :, :].reshape(-1)
            plot_line([time_serie, np.repeat(source_test[:, brain_area], time_steps), np.repeat(source_prediction_values[:, brain_area],time_steps)],["Predicted Activity","Real State","Predicted State"],  str(brain_area))
            plot_line([reals[brain_area,:], time_serie],['Real Activity', 'Predicted Activity'], "fit" + str(brain_area))  
    etime_series = eLoretaModel(EEG_data_test,100, gamma)[:, :, :100]
    
    source_test = np.zeros((trials, brain_areas))
    for trial in range(trials):
        source_test[trial, sources_test[trial,:]] = 1
        
    time_series, source_prediction_values = Model(EEG_data_test, NeuralNets, CNN_Nets)     
    
    mse = np.zeros((trials, 3))
    emse = np.zeros((trials, 3))
    
    reals = np.zeros((brain_areas, trials * time_steps))
    
    for trial, active_neurons in enumerate(sources_test):
        for i, neuron in enumerate(active_neurons):
            real = activity_test[i, :, trial]
            pred = time_series[neuron, trial, :]
            epred = etime_series[trial,:, neuron]
            mse[trial, i] = ((real - pred)**2).mean()
            emse[trial, i]= ((real - epred)**2).mean()
            reals[neuron, (trial * time_steps):((trial + 1) * time_steps)] = real
            
    var_pred = varModel(time_series.T, 0)
    var_epred = varModel(etime_series, 1)
    
        
    area_accuracy, true_negative, true_positive, true_positive_area, true_negative_area = most_likely_sources(source_prediction_values, source_test)
    vparea_accuracy, vptrue_negative, vptrue_positive, vptrue_positive_area, vptrue_negative_area = most_likely_sources(var_pred, source_test)
    vearea_accuracy, vetrue_negative, vetrue_positive, vetrue_positive_area, vetrue_negative_area = most_likely_sources(var_epred, source_test)
    
    if plot: plot_performance(time_series, source_prediction_values, reals, source_test)
                
    mean_mse = mse.mean()
    std_mse = mse.std()
    emean_mse = emse.mean()
    estd_mse = emse.std()
       
    return area_accuracy, true_positive, true_negative, vparea_accuracy,  vptrue_positive, vptrue_negative, vearea_accuracy,  vetrue_positive, vetrue_negative,  mean_mse, std_mse, emean_mse, estd_mse, time_series, mse, emse, source_prediction_values, true_positive_area, true_negative_area, vptrue_positive_area, vptrue_negative_area, vetrue_positive_area, vetrue_negative_area
        