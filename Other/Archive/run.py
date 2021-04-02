# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""

import os, pickle, torch
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NNfitting_functions import fit_NeuralNets, fitBalancedNeuralNets
from DataSplit_functions import splitData
from DataSimulation_functions import EEG_signal
from Visualize_functions import plot_line
#%%
def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials):
    data_file = "../../Data/EEG/data_" + str(snr) + "_" + str(cnr) + "_" + str(noisy_areas) + "_" + str(brain_areas) + "_" + str(time_steps) + "_" + str(trials) + ".pkl"
    
    if os.path.exists(data_file):            
        EEG_data, sources, activity = pickle.load(open(data_file, "rb"))
    else:
        EEG_data, sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                  snr, cnr, noisy_areas)
    
    return EEG_data, sources, activity

def changeDataFormat(EEG_data, sources, brain_areas):
    
    X = EEG_data.reshape((EEG_data.shape[0],-1), order = 'F')
    X = torch.Tensor(X).transpose(0,1)

    y = torch.Tensor(np.zeros((len(sources), brain_areas)))
    for trial, active_areas in enumerate(sources):
        y[trial, active_areas] = 1
        
    return X, y  

#%%
snr, cnr, noisy_areas, brain_areas, time_steps, trials = (0.9, 0.9, 5, 10, 50, 60)
EEG_data, sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials)
#%%
X_prediction, y_prediction = splitData(EEG_data, sources, activity, brain_areas)
#%%
NeuralNets, results_pred = fit_NeuralNets(X_prediction, y_prediction, val_freq=100, architecture=[20])
#%%
plot_line([y_prediction["0"]["test"].transpose(0,1)[0], results_pred[0]["test_prediction"]])
#%%
plot_line([NeuralNets[0](X_classification).detach().transpose(0,1)[0], np.repeat(y_classification[:,0],300)])
#%%
X_classification, y_classification = changeDataFormat(EEG_data,sources, brain_areas)
#%%
#results_clas = fitBalancedNeuralNets(NeuralNets, X_classification, y_classification, time_steps, architecture= ([5, 4, 10], [10, 3], [2,1]), EPOCHS=100, val_freq = 50, max_val_amount = 30)   