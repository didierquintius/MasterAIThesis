# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 08:42:14 2020

@author: didie
"""
import pickle, os, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from DataSimulation_functions import EEG_signal
from DataSplit_functions import splitTestData, prepareProjectionData, setNNFormat, prepareClassificationData
from NeuralNetwork_functions import NeuralNet, CNN

def importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials):
    data_file = "../../Data/EEG/data_" + str(snr) + "_" + str(cnr) + "_" + str(noisy_areas) + "_" + str(brain_areas) + "_" + str(time_steps) + "_" + str(trials) + ".pkl"
    
    if os.path.exists(data_file):            
        EEG_data, sources, activity = pickle.load(open(data_file, "rb"))
    else:
        EEG_data, sources, activity =  EEG_signal(time_steps, trials, brain_areas,
                                                  snr, cnr, noisy_areas)
    
    return EEG_data, sources, activity
#%%
snr, cnr, noisy_areas, brain_areas, time_steps, trials = (0.9, 0.9, 5, 10, 50, 60)
EEG_data, sources, activity =  importData(snr, cnr, noisy_areas, brain_areas, time_steps, trials)
EEG_data_test, sources_test, activity_test, EEG_data_trainval, sources_trainval, activity_trainval = splitTestData(EEG_data, sources, activity)
del EEG_data, sources, activity

#%%
a,b,c,d = prepareProjectionData(EEG_data_trainval, sources_trainval, activity_trainval, 0)

#%%
NeuralNets = {}
for brain_area in range(brain_areas):NeuralNets[brain_area] = NeuralNet(108, [5], 1)

X, y= prepareClassificationData(EEG_data_trainval, sources_trainval, brain_areas, NeuralNets)
    
    