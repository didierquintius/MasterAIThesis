# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:03:13 2021

@author: didie
"""
import os, pickle
import numpy as np
from tqdm import tqdm
from evaluation import evaluateResults
#%%
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from DataSimulation_functions import simulateData, setNNFormat
#%%
def analyzeDipole(EEG_Data, dipole, NNs, CNNs):
    electrodes, time_steps, _ = EEG_Data.shape
    EEG_Data = setNNFormat(EEG_Data, electrodes)
    dipole_activity = NNs[dipole](EEG_Data)
    predicted_class = CNNs[dipole](dipole_activity.reshape(-1, 1, time_steps)).detach().numpy().reshape(-1)
    dipole_activity = dipole_activity.detach().numpy().reshape((-1, time_steps))
    return (dipole_activity, predicted_class)
#%%
param_values = dict(trials = 750, time_steps = 10, brain_areas = 1000)

def evaluateML(param_values):

    EEG_Data, activity, source_trials, noisy_trials  = simulateData(param_values, 'close' , seed = 1)
    results = pickle.load(open("C:/Users/didie/Documents/MasterAIThesis/Code/Results-17-3/TrainResults.pkl",'rb'))
    
    NNs, CNNs = results['NNs'], results['CNNs']
    dipoles = len(NNs)
    electrodes, time_steps, trials = EEG_Data.shape
    real_activity = np.zeros((dipoles, trials, time_steps))
    source_likelihood = np.zeros((dipoles, trials))
    
    for dipole in tqdm(range(dipoles)):
        real_activity[dipole,:], source_likelihood[dipole,:] = analyzeDipole(EEG_Data, dipole, NNs, CNNs)
    
    source_dipoles = np.zeros((dipoles, trials)).astype('int')
    noisy_dipoles = np.zeros((dipoles, trials)).astype('int')
    
    for trial in range(trials): 
        source_dipoles[source_trials[trial],trial] = 1
        noisy_dipoles[noisy_trials[trial,:], trial] = 1
    
    mse_pred = ((activity- real_activity)**2).mean(axis = 2)
    mse_clas = (source_likelihood - source_dipoles)**2
    accuracy = ((source_likelihood >= 0.5) == source_dipoles)
    #%%
    estimated_sources = [ np.argsort(source_likelihood[:, trial])[-3:] for trial in range(750)]
    source_accuracy = [[source in source_trials[trial] for source in estimated_sources[trial]] for trial in range(750)]
    source_accuracy = np.array(source_accuracy).mean(axis =1)
    #%%
    evaluateResults(mse_pred, mse_clas, accuracy, source_dipoles, noisy_dipoles, source_accuracy)

def clasmodel(neural_activity):
    variances = neural_activity.std(axis = 2)
    variances = variances / variances.max()
    center_variance = np.sort(variances.reshape(-1))[-int(3 * neural_activity.shape[0])]
    variances = np.exp(np.log(0.5)/np.log(center_variance) * np.log(variances))
    return variances
    
          