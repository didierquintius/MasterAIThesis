# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:10:28 2020

@author: didie
"""

import os, pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from NNfitting_functions import fit_NeuralNets, fitBalancedNeuralNets
from DataLoadandSplit_functions import splitData
from DataSimulation_functions import EEG_signal
#%%
def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials):
    data_file = "../../Data/EEG/data_" + str(snr) + "_" + str(cnr) + "_" + str(noisy_areas) + "_" + str(brain_areas) + "_" + str(time_steps) + "_" + str(trials) + ".pkl"
    
    if os.path.exists(data_file):            
        EEG_data, sources, activity = pickle.load(open(data_file, "rb"))
    else:
        EEG_data, sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                  snr, cnr, noisy_areas)
    
    return EEG_data, sources, activity

#%%
snr, cnr, noisy_areas, brain_areas, time_steps, trials = (0.9, 0.9, 5, 20, 50, 100)
EEG_data, sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials)
#%%
X, y = splitData(EEG_data, sources, activity, brain_areas)
#%%
NeuralNets = fit_NeuralNets(X, y)
#%%
EEG_data = EEG_data.reshape((electrodes, timesteps * trials), order = 'F')
EEG_data = torch.Tensor(EEG_data).transpose(0,1)
#%%
source_matrix = torch.Tensor(np.zeros((trials, timesteps)))
for trial, active_areas in enumerate(sources):
    source_matrix[trial, active_areas] = 1
#%%
def fitBalancedNeuralNets(NeuralNets, EEG_data, source_matrix, timesteps)